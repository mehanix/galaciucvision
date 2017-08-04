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

#include "utils.hpp"

cv::Ptr<cv::ml::ANN_MLP> mlp;
cv::Mat vocabulary;
cv::FlannBasedMatcher flann;


cv::Mat getBOWFeatures(cv::FlannBasedMatcher& flann, const cv::Mat& descriptors, int vocabularySize)
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
	if (argc != 2) {
		std::cout << "Usage: score <data_folder>";
		exit(-1);
	}

	std::cout << "Loading neural network..." << std::endl;
	mlp = cv::Algorithm::load<cv::ml::ANN_MLP>("mlp.yaml");
	cv::FileStorage fs("vocabulary.yaml", cv::FileStorage::READ);
	fs["vocabulary"] >> vocabulary;
	flann.add(vocabulary);
	flann.train();

	// Reading test set
	std::cout << "Reading images..." << std::endl;
	const char *dir = argv[1];
	std::vector<std::string> files = getFilesFromDir(dir);
	double start = cv::getTickCount();
	cv::Mat testSamples;
	readImages(files.begin(), files.end(),
			   [&](const std::string & classname, const cv::Mat & descriptors) {
				   cv::Mat bowFeatures = getBOWFeatures(flann, descriptors, NETWORK_SIZE);
				   cv::normalize(bowFeatures, bowFeatures, 0, bowFeatures.rows, cv::NORM_MINMAX, -1, cv::Mat());
				   testSamples.push_back(bowFeatures);
			   });
	std::cout << "Time elapsed in minutes: " << ((double)cv::getTickCount() - start) / cv::getTickFrequency() / 60.0 << std::endl << std::endl;

	std::cout << "Results:" << std::endl;
	cv::Mat testOutput;
	mlp->predict(testSamples, testOutput);
	for (int i = 0; i < testOutput.rows; i++) {
		std::cout << files.at(i) << " ";
		float score = testOutput.row(i).at<float>(0);
		score = score > 1 ? 100 : (score < 0 ? 0 : score * 100);
		std::cout << score << std::endl;
	}

	return 0;
}
