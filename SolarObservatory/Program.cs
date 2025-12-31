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
            DateTime requestedTimestamp = DateTime.UtcNow;
            Console.WriteLine($"Get: {requestedTimestamp:yyyy-MM-dd HH:mm:ss}");

            // Define wavelengths to download
            var wavelengths = new[] { 10, 11, 12 }; // AIA 171, 193, 211
            
            var downloadTasks = wavelengths.Select(sourceId => DownloadAndProcessAsync(requestedTimestamp, sourceId)).ToArray();
            var results = await Task.WhenAll(downloadTasks);
            
            var successfulDownloads = results.Where(r => r.rawData != null).ToList();
            
            if (successfulDownloads.Count > 0)
            {
                DateTime latestTimestamp = results.Select(r => r.timestamp).Max();
                
                await CreateContainerFileAsync(latestTimestamp, successfulDownloads);
            }
            else
            {
                throw new Exception("No images downloaded.");
            }
        }

        static async Task<(byte[]? rawData, DateTime timestamp, int sourceId, uint width, uint height)> DownloadAndProcessAsync(DateTime timestamp, int sourceId)
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

                    image.ColorSpace = ColorSpace.Gray;
                    image.Format = MagickFormat.Gray;
                    
                    byte[] rawPixels = image.ToByteArray();
                    
                    // Validate dimensions match data size
                    int expectedSize = (int)(image.Width * image.Height);
                    if (rawPixels.Length != expectedSize)
                    {
                        throw new Exception($"Data size mismatch: expected {expectedSize} bytes ({image.Width}x{image.Height}), got {rawPixels.Length} bytes");
                    }
                    
                    Console.WriteLine($"Processed sourceId {sourceId}: {image.Width}x{image.Height}, {rawPixels.Length:N0} bytes");
                    return (rawPixels, imageTimestamp, sourceId, image.Width, image.Height);
                }
            }
        }

        static async Task CreateContainerFileAsync(DateTime timestamp, List<(byte[]? rawData, DateTime timestamp, int sourceId, uint width, uint height)> downloads)
        {
            string containerPath = Path.Combine(AppContext.BaseDirectory, "solar.dat");
            
            using (var fs = new FileStream(containerPath, FileMode.Create, FileAccess.Write, FileShare.None, bufferSize: 4096, useAsync: true))
            using (var writer = new BinaryWriter(fs))
            {
                // Header:
                // - Image count (1 byte)
                // - Timestamp string (19 bytes: "yyyy-MM-dd HH:mm:ss")
                
                writer.Write((byte)downloads.Count);
                
                string timestampStr = timestamp.ToString("yyyy-MM-dd HH:mm:ss");
                byte[] timestampBytes = System.Text.Encoding.ASCII.GetBytes(timestampStr);
                writer.Write(timestampBytes); // Always 19 bytes
                
                // Image entries:
                // For each image:
                // - Source ID (1 byte)
                // - Width (2 bytes, uint16)
                // - Height (2 bytes, uint16)
                // - Raw grayscale bitmap data (width * height bytes)
                
                foreach (var (rawData, _, sourceId, width, height) in downloads)
                {
                    if (rawData != null)
                    {
                        writer.Write((byte)sourceId);
                        writer.Write((ushort)width);
                        writer.Write((ushort)height);
                        writer.Write(rawData);
                    }
                }
            }
            
            var fileInfo = new FileInfo(containerPath);
            Console.WriteLine($"Created container of size: {fileInfo.Length:N0} bytes ({fileInfo.Length / 1024.0 / 1024.0:F2} MB)");
        }
    }
}