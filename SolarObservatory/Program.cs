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
                Console.WriteLine($"Get: {requestedTimestamp:yyyy-MM-dd HH:mm:ss.fff}");

                const int sourceId = 10; // AIA 171
                var (image, receivedTimestamp) = await DownloadWavelengthAsync(requestedTimestamp, sourceId);
                
                Console.WriteLine($"Got: {receivedTimestamp:yyyy-MM-dd HH:mm:ss.fff}");
                
                if (image != null)
                {
                    using (image)
                    {
                        string baseFilename = $"{receivedTimestamp:yyyy-MM-dd_HH-mm-ss-fff}";
                        await DownscaleAndSaveAsync(image, baseFilename);
                    }
                }
            }
            catch (Exception e)
            {
                Console.WriteLine($"Error: {e.Message}");
            }
        }

        static async Task<(MagickImage? image, DateTime timestamp)> DownloadWavelengthAsync(DateTime timestamp, int sourceId)
        {
            string url = $"https://api.helioviewer.org/v2/getJP2Image/?date={timestamp:yyyy-MM-ddTHH:mm:ssZ}&sourceId={sourceId}";

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
                        string timePart = parts[1].Replace('_', ':');
                        imageTimestamp = DateTime.ParseExact($"{datePart} {timePart}", "yyyy-MM-dd HH:mm:ss:fff", System.Globalization.CultureInfo.InvariantCulture);
                    }
                }

                using (Stream contentStream = await response.Content.ReadAsStreamAsync())
                {
                    var image = new MagickImage(contentStream);
                    return (image, imageTimestamp);
                }
            }
        }

        static async Task DownscaleAndSaveAsync(MagickImage image, string baseFilename)
        {
            // Apply 50% downscaling with Lanczos filter
            image.FilterType = FilterType.Lanczos;
            image.Resize(image.Width / 2, image.Height / 2);

            string qoiPath = Path.Combine(AppContext.BaseDirectory, baseFilename + ".qoi");
            image.Format = MagickFormat.Qoi;
            await image.WriteAsync(qoiPath);

#if DEBUG
            string pngPath = Path.Combine(AppContext.BaseDirectory, baseFilename + ".png");
            image.Format = MagickFormat.Png;
            await image.WriteAsync(pngPath);
#endif
        }
    }
}