#include <vector>
#include <algorithm>
#include <functional>
#include <map>
#include <set>
#include <fstream>
#include <string.h>
#include <error.h>
#include <dirent.h>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/ml/ml.hpp>

cv::Ptr<cv::ml::ANN_MLP> mlp;
cv::Mat vocabulary;
std::set<std::string> classes;
cv::FlannBasedMatcher flann;

/**
 * Turn local features into a single bag of words histogram of
 * of visual words (a.k.a., bag of words features)
 */
cv::Mat getBOWFeatures(cv::FlannBasedMatcher& flann, const cv::Mat& descriptors,
					   int vocabularySize)
{
	cv::Mat outputArray = cv::Mat::zeros(cv::Size(vocabularySize, 1), CV_32F);
	std::vector<cv::DMatch> matches;
	flann.match(descriptors, matches);
	for (size_t j = 0; j < matches.size(); j++) {
		int visualWord = matches[j].trainIdx;
		outputArray.at<float>(visualWord)++;
	}
	return outputArray;
}

int main(int argc, char const *argv[])
{
	std::cout << "Loading neural network" << std::endl;
	mlp = cv::ml::ANN_MLP::create();
	mpl.load(argv[1]);
	cv::FileStorage fs(argv[2], cv::FileStorage::READ);
	fs >> vocabulary;

	std::ifstream classesFile(argv[3]);
	int numOfClasses;
	classesFile >> numOfClasses;
	while (numOfClasses--) {
		classesFile >> name;
		classes.insert(name);
	}
	classesFile.close();

	std::cout << "Training FLANN..." << std::endl;
	double start = (double)cv::getTickCount();
	flann.add(vocabulary);
	flann.train();
	return 0;
	std::cout << "Time elapsed in minutes: " << ((double)cv::getTickCount() - start) / cv::getTickFrequency() / 60.0 << std::endl;

	// Reading test set
	std::cout << "Reading images..." << std::endl;
	char *dir = argv[4];
	std::vector<std::string> files = getFilesFromDir(dir);
	start = cv::getTickCount();
	cv::Mat testSamples;
	readImages(files.begin(), files.end(),
			   [&](const std::string & classname, const cv::Mat & descriptors) {
	               // Get histogram of visual words using bag of words technique
				   cv::Mat bowFeatures = getBOWFeatures(flann, descriptors, networkInputSize);
				   cv::normalize(bowFeatures, bowFeatures, 0, bowFeatures.rows, cv::NORM_MINMAX, -1, cv::Mat());
				   testSamples.push_back(bowFeatures);
			   });
	std::cout << "Time elapsed in minutes: " << ((double)cv::getTickCount() - start) / cv::getTickFrequency() / 60.0 << std::endl;

	cv::Mat testOutput;
	mlp->predict(testSamples, testOutput);
	for (int i = 0; i < testOutput.rows; i++) {
		std::cout << files.at(i);
		for (int j = 0; j < testOutput.cols; j++) {
			float prediction = testOutput.row(i).at<float>(j);
			std::cout << prediction > 1 ? 1 : (prediction < 0 ? 0 : predition) << " ";
		}
	}
}
