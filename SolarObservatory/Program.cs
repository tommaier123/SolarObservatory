using ImageMagick;
using System.Globalization;
using System.Text.RegularExpressions;

namespace SolarObservatory
{
    internal class Program
    {
        // Limit heavy CPU processing to the number of physical cores available (usually 2 on standard GitHub Actions)
        static readonly SemaphoreSlim _magickSemaphore = new SemaphoreSlim(Environment.ProcessorCount);

        // Set to false to skip saving debug images
        const bool SAVE_DEBUG_IMAGES = false;

        // Use a persistent HTTP client to prevent socket exhaustion and repeated connection overhead
        static readonly HttpClient client = new HttpClient();

        static async Task<int> Main(string[] args)
        {
            try
            {
                // Disable Magick.NET parallelism for individual image operations.
                // GitHub Actions runners usually only have 2-4 vCPUs. 
                // Because we are already processing wavelengths concurrently using Task.WhenAll, 
                // ImageMagick trying to internally multithread EACH resize operation causes massive CPU thread thrashing and slows everything down.
                Environment.SetEnvironmentVariable("MAGICK_THREAD_LIMIT", "1");

                Console.WriteLine("Starting JSOC/AIA+HMI collector (C# port of solar_observatory.py)");

                // wavelengths of interest (AIA): 131,171,193,304,1700
                var aiaWavelengths = new[] { 131, 171, 193, 304, 1700 };

                // Build candidate lists in parallel (over -1..+1 days)
                var lists = new Dictionary<string, List<(DateTime ts, string url)>>();

                var tasks = new List<Task>();
                foreach (var wl in aiaWavelengths)
                {
                    tasks.Add(Task.Run(async () =>
                    {
                        var res = await ListAiaCandidates(client, wl, 1);
                        lock (lists) { lists[wl.ToString()] = res; }
                    }));
                }

                tasks.Add(Task.Run(async () =>
                {
                    var res = await ListHmiCandidates(client, 1);
                    lock (lists) { lists["HMI"] = res; }
                }));

                await Task.WhenAll(tasks);

                // collect candidate reference times: take latest N from each list
                var REF_CANDIDATES = new HashSet<DateTime>();
                const int N = 3;
                foreach (var kv in lists)
                {
                    var lst = kv.Value;
                    foreach (var item in lst.Skip(Math.Max(0, lst.Count - N)))
                        REF_CANDIDATES.Add(item.ts);
                }

                if (!REF_CANDIDATES.Any())
                    throw new Exception("No candidates found for HMI/AIA");

                // nearest selection function
                (DateTime? ts, string? url, double diff) NearestInList(List<(DateTime ts, string url)> lst, DateTime refTs)
                {
                    if (lst == null || lst.Count == 0) return (null, null, double.PositiveInfinity);
                    var best = lst.OrderBy(x => Math.Abs((x.ts - refTs).TotalSeconds)).First();
                    return (best.ts, best.url, Math.Abs((best.ts - refTs).TotalSeconds));
                }

                DateTime? bestRef = null;
                Dictionary<string, (DateTime? ts, string? url)> bestSelection = null;
                double bestMaxDiff = double.PositiveInfinity;

                foreach (var refCandidate in REF_CANDIDATES.OrderByDescending(x => x))
                {
                    var selection = new Dictionary<string, (DateTime? ts, string? url)>();
                    double maxdiff = 0.0;

                    var hmiNearest = NearestInList(lists.GetValueOrDefault("HMI", new List<(DateTime, string)>()), refCandidate);
                    selection["HMI"] = (hmiNearest.ts, hmiNearest.url);
                    maxdiff = Math.Max(maxdiff, hmiNearest.diff);

                    foreach (var wl in aiaWavelengths)
                    {
                        var lst = lists.GetValueOrDefault(wl.ToString(), new List<(DateTime, string)>());
                        var n = NearestInList(lst, refCandidate);
                        selection[wl.ToString()] = (n.ts, n.url);
                        maxdiff = Math.Max(maxdiff, n.diff);
                    }

                    if ((maxdiff < bestMaxDiff) || (Math.Abs(maxdiff - bestMaxDiff) < 1e-6 && (bestRef == null || refCandidate > bestRef)))
                    {
                        bestMaxDiff = maxdiff;
                        bestRef = refCandidate;
                        bestSelection = selection;
                    }
                }

                if (bestSelection == null)
                    throw new Exception("Failed to choose selection");

                // Download selected images in parallel
                var downloads = new List<(byte[] data, DateTime ts, string key, int w, int h)>();

                var dlTasks = new List<Task>();

                foreach (var wl in aiaWavelengths)
                {
                    var key = wl.ToString();
                    var (ts, url) = bestSelection[key];
                    if (!string.IsNullOrEmpty(url))
                    {
                        dlTasks.Add(Task.Run(async () =>
                        {
                            var res = await DownloadAiaByUrl(client, url!, wl);
                            lock (downloads) { if (res.data != null) downloads.Add((res.data, res.ts, key, res.w, res.h)); else downloads.Add((null, res.ts, key, 0, 0)); }
                        }));
                    }
                    else
                    {
                        downloads.Add((null, DateTime.MinValue, key, 0, 0));
                    }
                }

                var h = bestSelection["HMI"];
                if (!string.IsNullOrEmpty(h.url))
                {
                    dlTasks.Add(Task.Run(async () =>
                    {
                        var res = await DownloadHmiByUrl(client, h.url!);
                        lock (downloads) { if (res.data != null) downloads.Add((res.data, res.ts, "HMI", res.w, res.h)); else downloads.Add((null, res.ts, "HMI", 0, 0)); }
                    }));
                }

                await Task.WhenAll(dlTasks);

                // Use average timestamp of successful downloads
                var successfulItems = downloads.Where(d => d.data != null && d.ts != DateTime.MinValue).ToList();
                DateTime containerTs;
                if (successfulItems.Any())
                {
                    var secs = successfulItems.Select(d => d.ts.ToUniversalTime().Subtract(DateTime.UnixEpoch).TotalSeconds);
                    var avg = secs.Average();
                    containerTs = DateTime.UnixEpoch.AddSeconds(avg).ToUniversalTime();
                }
                else
                {
                    containerTs = DateTime.UtcNow;
                }

                // Create container file (expects order: 131,171,193,304,1700,HMI)
                await CreateContainerFileAsync(containerTs, downloads);

                // write timestamp.txt
                var tsPath = Path.Combine(AppContext.BaseDirectory, "timestamp.txt");
                await File.WriteAllTextAsync(tsPath, containerTs.ToString("yyyy-MM-dd HH:mm:ss", CultureInfo.InvariantCulture));

                // Print summary
                var containerPath = Path.Combine(AppContext.BaseDirectory, "solar.dat");
                var fi = new FileInfo(containerPath);

                Console.WriteLine(new string('=', 60));
                Console.WriteLine("TIMESTAMP SUMMARY");
                Console.WriteLine(new string('=', 60));
                Console.WriteLine($"Current time: {DateTime.UtcNow:yyyy-MM-dd HH:mm:ss 'UTC'}");
                var age = DateTime.UtcNow - containerTs;
                var ageSeconds = (int)age.TotalSeconds;
                var hrs = ageSeconds / 3600; var rem = ageSeconds % 3600; var mins = rem / 60; var secs2 = rem % 60;
                string ageHuman = hrs > 0 ? $"{hrs}h{mins}m{secs2}s" : (mins > 0 ? $"{mins}m{secs2}s" : $"{secs2}s");
                Console.WriteLine($"Data age: {ageHuman} ({age.TotalMinutes:F1} min)");
                Console.WriteLine(new string('=', 60));

                foreach (var key in new[] { "131", "171", "193", "304", "1700", "HMI" })
                {
                    var item = downloads.FirstOrDefault(d => d.key == key && d.data != null);
                    if (item.data != null)
                    {
                        var delta = (item.ts - containerTs).TotalMinutes;
                        Console.WriteLine($"{key.PadLeft(5)}: {item.ts.ToString("yyyy-MM-dd HH:mm:ss 'UTC'")} (Δ {delta:+0.0} min)");
                    }
                }

                Console.WriteLine(new string('=', 60));
                Console.WriteLine($"Container: {fi.Length:N0} bytes ({fi.Length / 1024.0 / 1024.0:F2} MB)");
                Console.WriteLine(new string('=', 60));

                return 0;
            }
            catch (Exception ex)
            {
                Console.Error.WriteLine($"Error: {ex.Message}");
                Console.Error.WriteLine(ex.StackTrace);
                return 2;
            }
        }

        static async Task<List<(DateTime ts, string url)>> ListHmiCandidates(HttpClient client, int maxDays = 1)
        {
            var now = DateTime.UtcNow;
            var outList = new List<(DateTime ts, string url)>();
            for (int dayOffset = -maxDays; dayOffset <= maxDays; dayOffset++)
            {
                var d = now.AddDays(dayOffset);
                var baseUrl = $"https://jsoc1.stanford.edu/data/hmi/images/{d.Year}/{d.Month:00}/{d.Day:00}/";
                try
                {
                    var txt = await client.GetStringAsync(baseUrl);
                    var matches = Regex.Matches(txt, "href=\"([0-9_]+_M_4k\\.jpg)\"");
                    foreach (Match m in matches)
                    {
                        var fn = m.Groups[1].Value;
                        try
                        {
                            var parts = fn.Split('_');
                            var ts = DateTime.ParseExact($"{parts[0]}_{parts[1]}", "yyyyMMdd_HHmmss", CultureInfo.InvariantCulture, DateTimeStyles.AssumeUniversal | DateTimeStyles.AdjustToUniversal);
                            outList.Add((ts, baseUrl + fn));
                        }
                        catch { continue; }
                    }
                }
                catch { continue; }
            }
            outList.Sort((a, b) => a.ts.CompareTo(b.ts));
            return outList;
        }

        static async Task<List<(DateTime ts, string url)>> ListAiaCandidates(HttpClient client, int wavelength, int maxDays = 1)
        {
            var now = DateTime.UtcNow;
            var outList = new List<(DateTime ts, string url)>();
            for (int dayOffset = -maxDays; dayOffset <= maxDays; dayOffset++)
            {
                var h = now.AddDays(dayOffset);
                var baseUrl = $"http://jsoc.stanford.edu/data/aia/images/{h.Year}/{h.Month:00}/{h.Day:00}/{wavelength}/";
                try
                {
                    var txt = await client.GetStringAsync(baseUrl);
                    var matches = Regex.Matches(txt, "href=\"([^\"]+\\.jp2)\"");
                    foreach (Match m in matches)
                    {
                        var fn = m.Groups[1].Value;
                        var parts = fn.Split(new string[] { "__" }, StringSplitOptions.None);
                        if (parts.Length < 2) continue;
                        try
                        {
                            var dateStr = parts[0].Replace('_', '-');
                            var timeStr = parts[1].Substring(0, 8).Replace('_', ':');
                            var ts = DateTime.ParseExact($"{dateStr} {timeStr}", "yyyy-MM-dd HH:mm:ss", CultureInfo.InvariantCulture, DateTimeStyles.AssumeUniversal | DateTimeStyles.AdjustToUniversal);
                            outList.Add((ts, baseUrl + fn));
                        }
                        catch { continue; }
                    }
                }
                catch { continue; }
            }
            outList.Sort((a, b) => a.ts.CompareTo(b.ts));
            return outList;
        }

        static async Task<(byte[] data, DateTime ts, int w, int h)> DownloadAiaByUrl(HttpClient client, string url, int wavelength)
        {
            try
            {
                using var resp = await client.GetAsync(url);
                resp.EnsureSuccessStatusCode();
                using var ms = new MemoryStream();
                await resp.Content.CopyToAsync(ms);
                ms.Position = 0;

                DateTime ts;
                var fn = url.Split('/').Last();
                var parts = fn.Split(new string[] { "__" }, StringSplitOptions.None);
                if (parts.Length >= 2)
                {
                    var dateStr = parts[0].Replace('_', '-');
                    var timeStr = parts[1].Substring(0, 8).Replace('_', ':');
                    if (!DateTime.TryParseExact($"{dateStr} {timeStr}", "yyyy-MM-dd HH:mm:ss", CultureInfo.InvariantCulture, DateTimeStyles.AssumeUniversal | DateTimeStyles.AdjustToUniversal, out ts))
                        ts = DateTime.UtcNow;
                }
                else ts = DateTime.UtcNow;

                await _magickSemaphore.WaitAsync();
                try
                {
                    using var img = new MagickImage(ms);
                    img.FilterType = FilterType.Lanczos;
                    img.Resize(2048, 2048);
                    img.ColorSpace = ColorSpace.Gray;
                    img.Format = MagickFormat.Gray;
                    img.Depth = 8;
                    var raw = img.ToByteArray();

                    if (SAVE_DEBUG_IMAGES)
                    {
                        SaveDebugImage(raw, 2048, 2048, $"AIA_{wavelength}_{ts:yyyyMMdd_HHmmss}.png");
                    }

                    return (raw, ts, (int)img.Width, (int)img.Height);
                }
                finally
                {
                    _magickSemaphore.Release();
                }
            }
            catch (Exception ex)
            {
                Console.Error.WriteLine($"Error downloading AIA by URL {url}: {ex.Message}");
                return (null, DateTime.MinValue, 0, 0);
            }
        }

        static async Task<(byte[] data, DateTime ts, int w, int h)> DownloadHmiByUrl(HttpClient client, string url)
        {
            try
            {
                using var resp = await client.GetAsync(url);
                resp.EnsureSuccessStatusCode();
                using var ms = new MemoryStream();
                await resp.Content.CopyToAsync(ms);
                ms.Position = 0;

                DateTime ts;
                var fn = url.Split('/').Last();
                var parts = fn.Split('_');
                if (parts.Length >= 2 && DateTime.TryParseExact($"{parts[0]}_{parts[1]}", "yyyyMMdd_HHmmss", CultureInfo.InvariantCulture, DateTimeStyles.AssumeUniversal | DateTimeStyles.AdjustToUniversal, out ts))
                {
                }
                else ts = DateTime.UtcNow;

                await _magickSemaphore.WaitAsync();
                try
                {
                    using var img = new MagickImage(ms);
                    img.FilterType = FilterType.Lanczos;
                    img.Resize(2048, 2048);
                    img.ColorSpace = ColorSpace.Gray;
                    img.Format = MagickFormat.Gray;
                    img.Depth = 8;
                    var raw = img.ToByteArray();

                    if (SAVE_DEBUG_IMAGES)
                    {
                        SaveDebugImage(raw, 2048, 2048, $"HMI_{ts:yyyyMMdd_HHmmss}.png");
                    }

                    return (raw, ts, (int)img.Width, (int)img.Height);
                }
                finally
                {
                    _magickSemaphore.Release();
                }
            }
            catch (Exception ex)
            {
                Console.Error.WriteLine($"Error downloading HMI by URL {url}: {ex.Message}");
                return (null, DateTime.MinValue, 0, 0);
            }
        }

        static void SaveDebugImage(byte[] rawGray, int width, int height, string filename)
        {
            if (!SAVE_DEBUG_IMAGES) return;
            try
            {
                var debugDir = Path.Combine(AppContext.BaseDirectory, "debug_images");
                Directory.CreateDirectory(debugDir);
                var path = Path.Combine(debugDir, filename);
                var settings = new MagickReadSettings { Width = (uint)width, Height = (uint)height, Format = MagickFormat.Gray, Depth = 8 };
                using var img = new MagickImage();
                img.Read(rawGray, settings);
                img.Write(path);
            }
            catch (Exception ex) { Console.Error.WriteLine("SaveDebugImage error: " + ex); }
        }

        static void SaveDebugRgbImage(byte[] rawRgb, int width, int height, string filename)
        {
            if (!SAVE_DEBUG_IMAGES) return;
            try
            {
                var debugDir = Path.Combine(AppContext.BaseDirectory, "debug_images");
                Directory.CreateDirectory(debugDir);
                var path = Path.Combine(debugDir, filename);
                var settings = new MagickReadSettings { Width = (uint)width, Height = (uint)height, Format = MagickFormat.Rgb, Depth = 8 };
                using var img = new MagickImage();
                img.Read(rawRgb, settings);
                img.Write(path);
            }
            catch (Exception ex) { Console.Error.WriteLine("SaveDebugRgbImage error: " + ex); }
        }

        static async Task CreateContainerFileAsync(DateTime timestamp, List<(byte[] data, DateTime ts, string key, int w, int h)> downloads)
        {
            string containerPath = Path.Combine(AppContext.BaseDirectory, "solar.dat");
            // Order required: 131,171,193,304,1700,HMI
            var order = new[] { "131", "171", "193", "304", "1700", "HMI" };
            var map = downloads.ToDictionary(d => d.key, d => d);

            var successful = order.Select(k => map.ContainsKey(k) ? map[k] : (data: (byte[])null, ts: DateTime.MinValue, key: k, w: 0, h: 0)).ToList();

            if (successful.Count != 6)
                throw new Exception($"Missing channels when creating container");

            int width = 2048, height = 2048;

            byte[] rgb0 = CreateRgbImage(successful[0].data, successful[1].data, successful[2].data, width * height);
            byte[] rgb1 = CreateRgbImage(successful[3].data, successful[4].data, successful[5].data, width * height);

            SaveDebugRgbImage(rgb0, width, height, $"packed_131_171_193_{timestamp:yyyyMMdd_HHmmss}.png");
            SaveDebugRgbImage(rgb1, width, height, $"packed_304_1700_hmi_{timestamp:yyyyMMdd_HHmmss}.png");

            using var fs = new FileStream(containerPath, FileMode.Create, FileAccess.Write, FileShare.None);
            using var writer = new BinaryWriter(fs);
            writer.Write((byte)2);
            var tsStr = timestamp.ToString("yyyy-MM-dd HH:mm:ss", CultureInfo.InvariantCulture);
            var tsBytes = System.Text.Encoding.ASCII.GetBytes(tsStr);
            writer.Write(tsBytes);

            writer.Write((ushort)width);
            writer.Write((ushort)height);
            writer.Write(rgb0);

            writer.Write((ushort)width);
            writer.Write((ushort)height);
            writer.Write(rgb1);
        }

        static byte[] CreateRgbImage(byte[] rChannel, byte[] gChannel, byte[] bChannel, int pixelCount)
        {
            var empty = Enumerable.Repeat((byte)0, pixelCount).ToArray();
            if (rChannel == null) rChannel = empty;
            if (gChannel == null) gChannel = empty;
            if (bChannel == null) bChannel = empty;

            if (rChannel.Length < pixelCount || gChannel.Length < pixelCount || bChannel.Length < pixelCount)
                throw new Exception("Channel length mismatch or missing data");

            byte[] rgb = new byte[pixelCount * 3];
            for (int i = 0; i < pixelCount; i++)
            {
                rgb[i * 3] = rChannel[i];
                rgb[i * 3 + 1] = gChannel[i];
                rgb[i * 3 + 2] = bChannel[i];
            }
            return rgb;
        }
    }
}
