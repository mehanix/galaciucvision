#include <vector>
#include <functional>
#include <fstream>
#include <string.h>
#include <dirent.h>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/ml/ml.hpp>

#include "utils.hpp"

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
	bool singleFile = false;
	if (argc != 2 && argc != 3) {
		std::cout << "Usage: score <data_folder>" << std::endl;
		std::cout << "Usage: score -f <file>" << std::endl;
		exit(-1);
	}

	cv::Ptr<cv::ml::ANN_MLP> mlp = cv::Algorithm::load<cv::ml::ANN_MLP>("mlp.yaml");
	cv::FileStorage fs("vocabulary.yaml", cv::FileStorage::READ);

	cv::Mat vocabulary;
	fs["vocabulary"] >> vocabulary;
	cv::FlannBasedMatcher flann;
	flann.add(vocabulary);
	flann.train();

	std::vector<std::string> files;
	if (argc == 3 && !strcmp(argv[1], "-f")) {
		files.push_back(std::string(argv[2]));
		singleFile = true;
	} else {
		files = getFilesFromDir(argv[1]);
	}

	cv::Mat samples;
	readImages(files.begin(), files.end(),
			   [&](const std::string & classname, const cv::Mat & descriptors) {
				   cv::Mat bowFeatures = getBOWFeatures(flann, descriptors, NETWORK_SIZE);
				   cv::normalize(bowFeatures, bowFeatures, 0, bowFeatures.rows, cv::NORM_MINMAX, -1, cv::Mat());
				   samples.push_back(bowFeatures);
			   }, false);

	cv::Mat scores;
	float score;
	mlp->predict(samples, scores);
	for (int i = 0; i < scores.rows; i++) {
		if (!singleFile) std::cout << files.at(i) << " ";
		score = scores.row(i).at<float>(0);
		score = score > 1 ? 100 : (score < 0 ? 0 : score * 100);
		std::cout << score << std::endl;
	}

	return 0;
}
