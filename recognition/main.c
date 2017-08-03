#include <vector>
#include <algorithm>
#include <functional>
#include <map>
#include <set>
#include <fstream>
#include <string.h>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/ml/ml.hpp>

std::vector<std::string> getFilesFromList(char* list)
{
	std::vector<std::string> files;
	char *p1, *p2;
	p1 = list;
	while ((p2 = strchr(p1, ','))) {
		std::string path(p1, p2 - p1);
		files.push_back(path);
	}
	files.push_back(p1);
	return files;
}


int main(int argc, char** argv)
{
	if (argc != 4)
	{
			std::cerr << "Usage: <IMAGES> <NETWORK_INPUT_LAYER_SIZE> <TRAIN_SPLIT_RATIO>" << std::endl;
			exit(-1);
	}
	char* imagesList = argv[1];
	int networkInputSize = atoi(argv[2]);
	float trainSplitSize = atof(argv[3]);
	
	std::cout << "Reading training set..." << std::endl;
	double start = (double) cv::getTickCount();
	std::vector<std::string> files = getFilesFromList(imagesList);
	std::random_shuffle(files.begin(), files.end());
}