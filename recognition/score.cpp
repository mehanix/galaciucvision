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
	cv::Mat outputArray = cv::Mat::zeros(1, vocabularySize, CV_32F);
	std::vector<cv::DMatch> matches;
	flann.match(descriptors, matches);
	for (size_t i = 0; i < matches.size(); i++) outputArray.at<float>(matches[i].trainIdx)++;
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

	cv::Ptr<cv::ml::ANN_MLP> mlp = cv::Algorithm::load<cv::ml::ANN_MLP>("mlp.yml");
	cv::FileStorage fs("vocabulary.yml", cv::FileStorage::READ);

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
		try {
			files = getFilesFromDir(argv[1]);
		} catch (std::string e) {
			std::cout << "Exception caught: " << e << std::endl;
			exit(-1);
		}
	}

	cv::Mat samples;
	std::vector<std::string>::const_iterator it;
	int i;
	for (it = files.begin(), i = 0; it != files.end(); it++, i++) {
		std::cout << "\r" << i * 100 / files.size() << "%" << std::flush;
		cv::Mat descriptors;
		try {
			descriptors = readImageDescriptors(*it);
		} catch (std::string e) {
			std::cout << e << std::endl;
			continue;
		}
		cv::Mat bowFeatures = getBOWFeatures(flann, descriptors, NETWORK_SIZE);
		cv::Mat normalizedHist;
		cv::normalize(bowFeatures, normalizedHist, 0, 1, cv::NORM_MINMAX);
		samples.push_back(normalizedHist);
	}
	std::cout << "\r     \r";

	cv::Mat scores;
	float score;
	int good = 0, bad = 0;

	mlp->predict(samples, scores);
	for (int i = 0; i < scores.rows; i++) {
		if (!singleFile) std::cout << files.at(i) << " ";
		score = scores.at<float>(i);
		score = score > 1 ? 100 : (score < -1 ? 0 : (score + 1) * 50);

		std::cout << "\e[1m"; // Bold
		if (score < 100 / 3.0 && ++bad) std::cout << "\e[31m";  // Red
		else if (score < 200 / 3.0) std::cout << "\e[33m";  // Yellow
		else if (++good) std::cout << "\e[32m";  // Green

		std::cout << score << std::endl;

		std::cout << "\e[0m"; // Reset
	}

	// Print summary
	std::cout << "Summary:" << std::endl;
	std::cout << "\t\e[1m\e[32m" << good / (float)scores.rows * 100 << "% good" << std::endl;
	std::cout << "\t\e[31m" << bad / (float)scores.rows * 100 << "% bad" << std::endl;
	std::cout << "\t\e[33m" << (scores.rows - bad - good) / (float)scores.rows * 100 << "% unsure\e[0m" << std::endl;

	return 0;
}
