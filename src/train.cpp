#include <iostream>
#include <vector>
#include <functional>
#include <set>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/ml/ml.hpp>

#include "utils.hpp"

struct Image {
	cv::Mat descriptors;
	cv::Mat bowFeatures;
	bool isGood;
};

cv::Ptr<cv::ml::ANN_MLP> getTrainedNeuralNetwork(const cv::Mat& trainSamples, const cv::Mat& trainResponses)
{
	int networkInputSize = trainSamples.cols;
	int networkOutputSize = trainResponses.cols;
	int sizes[] = { networkInputSize, networkInputSize / 2, networkInputSize / 4, networkOutputSize };
	cv::Ptr<cv::ml::ANN_MLP> mlp = cv::ml::ANN_MLP::create();
	mlp->setTrainMethod(cv::ml::ANN_MLP::BACKPROP);
	mlp->setTermCriteria(cv::TermCriteria(cv::TermCriteria::MAX_ITER + cv::TermCriteria::EPS, 1000, 0.01));
	mlp->setLayerSizes(std::vector<int>(sizes, sizes + sizeof(sizes) / sizeof(sizes[0])));
	mlp->setActivationFunction(cv::ml::ANN_MLP::SIGMOID_SYM, 1, 1);
	mlp->train(trainSamples, cv::ml::ROW_SAMPLE, trainResponses);
	return mlp;
}

void saveModels(cv::Ptr<cv::ml::ANN_MLP> mlp, const cv::Mat& vocabulary)
{
	mlp->save("mlp.yml");
	cv::FileStorage fs("vocabulary.yml", cv::FileStorage::WRITE);
	fs << "vocabulary" << vocabulary;
	fs.release();
}


int main(int argc, char **argv)
{
	if (argc != 3) {
		std::cout << "Usage: train <good_data> <bad_data>" << std::endl;
		exit(-1);
	}

	std::cout << "Reading training set..." << std::endl;
	double start = (double)cv::getTickCount();

	std::vector<std::string> good_files;
	std::vector<std::string> bad_files;
	try {
		good_files = getFilesFromDir(argv[1]);
		bad_files = getFilesFromDir(argv[2]);
	} catch (std::string e) {
		std::cout << "Exception caught: " << e << std::endl;
		exit(-1);
	}

	std::vector<std::string> files(good_files);
	files.insert(files.end(), bad_files.begin(), bad_files.end());
	std::random_shuffle(files.begin(), files.end());

	std::vector<Image *> images;
	cv::Mat allDescriptors, descriptors;
	std::vector<int> indexes;
	std::vector<std::string>::const_iterator it;

	int i;
	for (it = files.begin(), i = 0; it != files.end(); it++, i++) {
		std::cout << "\rReading " << *it << std::endl;
		std::cout << i * 100 / files.size() << "%" << std::flush;
		try {
			descriptors = readImageDescriptors(*it);
		} catch (std::string e) {
			std::cout << e << std::endl;
			continue;
		}

		Image *img = new Image;
		img->isGood = !it->compare(0, std::string(argv[1]).size(), argv[1]);
		img->descriptors = descriptors;
		img->bowFeatures = cv::Mat::zeros(1, NETWORK_SIZE, CV_32F);
		images.push_back(img);
		allDescriptors.push_back(img->descriptors);
		for (int j = 0; j < img->descriptors.rows; j++) indexes.push_back(i);
	}
	std::cout << "\r     \r";

	std::cout << "Time elapsed: " << ((double)cv::getTickCount() - start) / cv::getTickFrequency() << std::endl;

	std::cout << "Creating vocabulary..." << std::endl;
	start = (double)cv::getTickCount();
	cv::Mat labels;
	cv::Mat vocabulary;
	cv::kmeans(allDescriptors, NETWORK_SIZE, labels, cv::TermCriteria(cv::TermCriteria::MAX_ITER + cv::TermCriteria::EPS, 10, 0.01), 1, cv::KMEANS_PP_CENTERS, vocabulary);
	std::cout << "Time elapsed: " << ((double)cv::getTickCount() - start) / cv::getTickFrequency() << std::endl;

	std::cout << "Getting histograms of visual words..." << std::endl;
	for (int i = 0; i < labels.rows; i++) {
		Image *img = images.at(indexes.at(i));
		img->bowFeatures.at<float>(labels.at<int>(i))++;
	}

	std::cout << "Preparing neural network..." << std::endl;
	cv::Mat trainSamples;
	cv::Mat trainResponses;
	std::random_shuffle(images.begin(), images.end());
	for (std::vector<Image *>::const_iterator it = images.begin(); it != images.end(); it++) {
		Image *img = *it;
		cv::Mat normalizedHist;
		cv::normalize(img->bowFeatures, normalizedHist, 0, 1, cv::NORM_MINMAX);
		trainSamples.push_back(normalizedHist);
		trainResponses.push_back(img->isGood ? 1.0f : -1.0f);
		delete *it;
	}

	std::cout << "Training neural network..." << std::endl;
	start = cv::getTickCount();
	cv::Ptr<cv::ml::ANN_MLP> mlp = getTrainedNeuralNetwork(trainSamples, trainResponses);
	std::cout << "Time elapsed: " << ((double)cv::getTickCount() - start) / cv::getTickFrequency() << std::endl;

	std::cout << "Saving models..." << std::endl;
	saveModels(mlp, vocabulary);

	std::cout << "Done" << std::endl;

	return 0;
}
