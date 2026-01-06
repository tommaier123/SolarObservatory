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

            var hmiResult = await DownloadAndProcessAsync(requestedTimestamp, 19);
            
            if (hmiResult.rawData == null)
            {
                throw new Exception("Failed to download HMI Magnetogram.");
            }
            
            DateTime referenceTimestamp = hmiResult.timestamp;
            Console.WriteLine($"Reference timestamp from HMI: {referenceTimestamp:yyyy-MM-dd HH:mm:ss}");
            
            // Now download AIA wavelengths using the HMI timestamp for consistency
            var aiaWavelengths = new[] { 9, 10, 11, 13, 16 }; // 131, 171, 193, 304, 1700
            
            var downloadTasks = aiaWavelengths.Select(sourceId => DownloadAndProcessAsync(referenceTimestamp, sourceId)).ToArray();
            var aiaResults = await Task.WhenAll(downloadTasks);
            
            var successfulDownloads = aiaResults.Where(r => r.rawData != null).ToList();
            successfulDownloads.Add(hmiResult); // Add HMI result to the list
            
            if (successfulDownloads.Count > 0)
            {
                await CreateContainerFileAsync(referenceTimestamp, successfulDownloads);
                await SaveTimestampFileAsync(referenceTimestamp);
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
                    // Apply downscaling to 2048x2048 with Lanczos filter
                    image.FilterType = FilterType.Lanczos;
                    image.Resize(2048, 2048);

                    image.ColorSpace = ColorSpace.Gray;
                    image.Format = MagickFormat.Gray;
                    
                    byte[] rawPixels = image.ToByteArray();

                    Console.WriteLine($"Processed sourceId {sourceId}: {image.Width}x{image.Height}, {rawPixels.Length:N0} bytes");
                    return (rawPixels, imageTimestamp, sourceId, image.Width, image.Height);
                }
            }
        }

        static async Task CreateContainerFileAsync(DateTime timestamp, List<(byte[]? rawData, DateTime timestamp, int sourceId, uint width, uint height)> downloads)
        {
            string containerPath = Path.Combine(AppContext.BaseDirectory, "solar.dat");
            
            // Expected order: sourceId 9 (131), 10 (171), 11 (193), 13 (304), 16 (1700), 19 (HMI Mag)
            // RGB Image 0: R=131, G=171, B=193
            // RGB Image 1: R=304, G=1700, B=HMI Mag
            
            var orderedDownloads = downloads.OrderBy(d => d.sourceId).ToList();
            
            if (orderedDownloads.Count != 6)
            {
                throw new Exception($"Expected 6 images, got {orderedDownloads.Count}");
            }
            
            // All images are 2048x2048 after downscaling
            uint width = 2048;
            uint height = 2048;
            
            // Create two RGB images by interleaving channels
            byte[] rgbImage0 = CreateRgbImage(orderedDownloads[0].rawData!, orderedDownloads[1].rawData!, orderedDownloads[2].rawData!);
            byte[] rgbImage1 = CreateRgbImage(orderedDownloads[3].rawData!, orderedDownloads[4].rawData!, orderedDownloads[5].rawData!);
            
            using (var fs = new FileStream(containerPath, FileMode.Create, FileAccess.Write, FileShare.None, bufferSize: 4096, useAsync: true))
            using (var writer = new BinaryWriter(fs))
            {
                // Header:
                // - Image count (1 byte) - always 2
                // - Timestamp string (19 bytes: "yyyy-MM-dd HH:mm:ss")
                
                writer.Write((byte)2);
                
                string timestampStr = timestamp.ToString("yyyy-MM-dd HH:mm:ss");
                byte[] timestampBytes = System.Text.Encoding.ASCII.GetBytes(timestampStr);
                writer.Write(timestampBytes); // Always 19 bytes
                
                // Image entries:
                // For each RGB image:
                // - Width (2 bytes, uint16)
                // - Height (2 bytes, uint16)
                // - RGB interleaved bitmap data (width * height * 3 bytes)
                
                writer.Write((ushort)width);
                writer.Write((ushort)height);
                writer.Write(rgbImage0);
                
                writer.Write((ushort)width);
                writer.Write((ushort)height);
                writer.Write(rgbImage1);
            }
            
            var fileInfo = new FileInfo(containerPath);
            Console.WriteLine($"Created container of size: {fileInfo.Length:N0} bytes ({fileInfo.Length / 1024.0 / 1024.0:F2} MB)");
        }
        
        static byte[] CreateRgbImage(byte[] rChannel, byte[] gChannel, byte[] bChannel)
        {
            if (rChannel.Length != gChannel.Length || gChannel.Length != bChannel.Length)
            {
                throw new Exception("All channels must have the same length");
            }
            
            int pixelCount = rChannel.Length;
            byte[] rgbData = new byte[pixelCount * 3];
            
            for (int i = 0; i < pixelCount; i++)
            {
                rgbData[i * 3] = rChannel[i];     // R
                rgbData[i * 3 + 1] = gChannel[i]; // G
                rgbData[i * 3 + 2] = bChannel[i]; // B
            }
            
            return rgbData;
        }
        
        static async Task SaveTimestampFileAsync(DateTime timestamp)
        {
            string timestampPath = Path.Combine(AppContext.BaseDirectory, "timestamp.txt");
            string timestampStr = timestamp.ToString("yyyy-MM-dd HH:mm:ss");
            
            await File.WriteAllTextAsync(timestampPath, timestampStr);
            
            Console.WriteLine($"Saved timestamp file: {timestampStr}");
        }
    }
}