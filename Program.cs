using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.IO;
using System.Net.Http;
using System.Net.Http.Headers;

namespace GVision
{
	partial class Program
	{

		private static string predictionKey;
		private static string predictionEndpointUrl;

		public static string directoryPath;
		public static string[] pictures;
		public static bool goodPath;
		static void Main(string[] args)
		{

			do
			{
				Console.WriteLine("Introdu cale poze:");

				directoryPath = Console.ReadLine();

				goodPath = true;
				try
				{
					pictures = Directory.GetFiles(directoryPath);
				}
				catch
				{
					goodPath = false;
				}
			} while (!goodPath);

			Console.WriteLine();


			foreach( string path in pictures)
			{
				byte[] pictureBytes = GetPictureBytes(path);
				MakeRequest(pictureBytes).Wait();
			}
		}



		static byte[] GetPictureBytes(string path)
		{
			FileStream fileStream = new FileStream(path, FileMode.Open, FileAccess.Read);
			BinaryReader binaryReader = new BinaryReader(fileStream);
			return binaryReader.ReadBytes((int)fileStream.Length);
		}
	
		static async Task MakeRequest (byte[] byteData)
		{
			using (var content = new ByteArrayContent(byteData))
			{
				var client = new HttpClient();
				client.DefaultRequestHeaders.Add("Prediction-Key", predictionKey);
				content.Headers.ContentType = new MediaTypeHeaderValue("application/octet-stream");
				HttpResponseMessage response = await client.PostAsync(_predictionEndpointUrl, content);
				Console.WriteLine(await response.Content.ReadAsStringAsync());
			}
		}
	}
}
