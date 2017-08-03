using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.IO;


namespace GVision
{
	class Program
	{

		public static string directoryPath;
		public static string[] pictures;

		static void Main(string[] args)
		{

			Console.WriteLine("Introdu cale poze:");
	
			directoryPath = Console.ReadLine();

			pictures = Directory.GetFiles(directoryPath);

		}
	}
}
