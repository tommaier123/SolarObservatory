using ImageMagick;
using System.Net;
using System.Net.Mime;
using System.Web;

namespace SolarVideo
{
    internal class Program
    {
        static async Task Main(string[] args)
        {
            try
            {
                DateTime requestedTimestamp = DateTime.UtcNow;
                Console.WriteLine($"Get: {requestedTimestamp:yyyy-MM-dd HH:mm:ss}");

                // Define wavelengths to download
                var wavelengths = new[] { 10, 11, 12 }; // AIA 171, 193, 211
                
                var downloadTasks = wavelengths.Select(sourceId => DownloadAndProcessAsync(requestedTimestamp, sourceId)).ToArray();
                var results = await Task.WhenAll(downloadTasks);
                
                var successfulDownloads = results.Where(r => r.qoiData != null).ToList();
                
                if (successfulDownloads.Count > 0)
                {
                    DateTime latestTimestamp = results.Select(r => r.timestamp).Max();
                    
                    // Create container file with all wavelengths
                    await CreateContainerFileAsync(latestTimestamp, successfulDownloads);
                }
                else
                {
                    Console.WriteLine("No images downloaded.");
                }
            }
            catch (Exception e)
            {
                Console.WriteLine($"Error: {e.Message}");
            }
        }

        static async Task<(byte[]? qoiData, DateTime timestamp, int sourceId)> DownloadAndProcessAsync(DateTime timestamp, int sourceId)
        {
            string url = $"https://api.helioviewer.org/v2/getJP2Image/?date={timestamp:yyyy-MM-ddTHH:mm:ssZ}&sourceId={sourceId}";

            try
            {
                using (HttpClient client = new HttpClient())
                using (HttpResponseMessage response = await client.GetAsync(url))
                {
                    response.EnsureSuccessStatusCode();

                    string? contentDispositionFilename = response.Content.Headers.ContentDisposition?.FileName?.Trim('"');
                    DateTime imageTimestamp = timestamp;

                    if (!string.IsNullOrEmpty(contentDispositionFilename))
                    {
                        string originalFilename = HttpUtility.UrlDecode(contentDispositionFilename);

                        // Parse timestamp from filename (format: 2025_12_30__14_01_21_349__SDO_AIA_AIA_171.jp2)
                        string[] parts = originalFilename.Split("__");
                        if (parts.Length >= 2)
                        {
                            string datePart = parts[0].Replace('_', '-');
                            string timePart = parts[1].Replace('_', ':').Substring(0, 8); // Take only HH:mm:ss, ignore milliseconds
                            imageTimestamp = DateTime.ParseExact($"{datePart} {timePart}", "yyyy-MM-dd HH:mm:ss", System.Globalization.CultureInfo.InvariantCulture);
                        }
                    }

                    Console.WriteLine($"Downloaded sourceId {sourceId}: {imageTimestamp:yyyy-MM-dd HH:mm:ss}");

                    using (Stream contentStream = await response.Content.ReadAsStreamAsync())
                    using (var image = new MagickImage(contentStream))
                    {
                        // Apply 50% downscaling with Lanczos filter
                        image.FilterType = FilterType.Lanczos;
                        image.Resize(image.Width / 2, image.Height / 2);

                        // Convert to QOI format in memory
                        image.Format = MagickFormat.Qoi;
                        byte[] qoiData = image.ToByteArray();
                        
                        Console.WriteLine($"Processed sourceId {sourceId}: {qoiData.Length:N0} bytes");
                        return (qoiData, imageTimestamp, sourceId);
                    }
                }
            }
            catch (Exception ex)
            {
                Console.WriteLine($"Failed to download/process sourceId {sourceId}: {ex.Message}");
                return (null, timestamp, sourceId);
            }
        }

        static async Task CreateContainerFileAsync(DateTime timestamp, List<(byte[]? qoiData, DateTime timestamp, int sourceId)> downloads)
        {
            string containerPath = Path.Combine(AppContext.BaseDirectory, "solar.dat");
            
            using (var fs = new FileStream(containerPath, FileMode.Create, FileAccess.Write, FileShare.None, bufferSize: 4096, useAsync: true))
            using (var writer = new BinaryWriter(fs))
            {
                // Header:
                // - Image count (1 byte)
                // - Timestamp string (19 bytes fixed: "yyyy-MM-dd HH:mm:ss")
                
                writer.Write((byte)downloads.Count);
                
                string timestampStr = timestamp.ToString("yyyy-MM-dd HH:mm:ss");
                byte[] timestampBytes = System.Text.Encoding.ASCII.GetBytes(timestampStr);
                writer.Write(timestampBytes); // Always 19 bytes
                
                // Image entries:
                // For each image:
                // - Source ID (1 byte)
                // - Data length (4 bytes, uint32)
                // - QOI data (variable)
                
                foreach (var (qoiData, _, sourceId) in downloads)
                {
                    if (qoiData != null)
                    {
                        writer.Write((byte)sourceId);
                        writer.Write((uint)qoiData.Length);
                        writer.Write(qoiData);
                    }
                }
            }
            
            var fileInfo = new FileInfo(containerPath);
            Console.WriteLine($"Created container of size: {fileInfo.Length:N0} bytes ({fileInfo.Length / 1024.0 / 1024.0:F2} MB)");
        }
    }
}