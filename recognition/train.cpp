#include <iostream>
#include <vector>
#include <functional>
#include <set>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/ml/ml.hpp>

#include "utils.hpp"

int getClassId(const std::set<std::string>& classes, const std::string& classname)
{
	int index = 0;
	for (auto it = classes.begin(); it != classes.end(); ++it) {
		if (*it == classname) break;
		++index;
	}
	return index;
}

cv::Mat getClassCode(const std::set<std::string>& classes, const std::string& classname)
{
	cv::Mat code = cv::Mat::zeros(cv::Size((int)classes.size(), 1), CV_32F);
	int index = getClassId(classes, classname);
	code.at<float>(index) = 1;
	return code;
}

cv::Ptr<cv::ml::ANN_MLP> getTrainedNeuralNetwork(const cv::Mat& trainSamples, const cv::Mat& trainResponses)
{
	int networkInputSize = trainSamples.cols;
	int networkOutputSize = trainResponses.cols;
	cv::Ptr<cv::ml::ANN_MLP> mlp = cv::ml::ANN_MLP::create();
	std::vector<int> layerSizes = { networkInputSize, networkInputSize / 2, networkOutputSize };
	mlp->setLayerSizes(layerSizes);
	mlp->setActivationFunction(cv::ml::ANN_MLP::SIGMOID_SYM);
	mlp->train(trainSamples, cv::ml::ROW_SAMPLE, trainResponses);
	return mlp;
}

void saveModels(cv::Ptr<cv::ml::ANN_MLP> mlp, const cv::Mat& vocabulary, const std::set<std::string>& classes)
{
	mlp->save("mlp.yaml");
	cv::FileStorage fs("vocabulary.yaml", cv::FileStorage::WRITE);
	fs << "vocabulary" << vocabulary;
	fs.release();
}


int main(int argc, char **argv)
{
	if (argc != 2) {
		std::cout << "Usage: train <train_data_folder>" << std::endl;
		exit(-1);
	}
	std::cout << "Reading training set..." << std::endl;
	double start = (double)cv::getTickCount();
	std::vector<std::string> files = getFilesFromDir(argv[1]);
	std::random_shuffle(files.begin(), files.end());

	cv::Mat descriptorsSet;
	std::vector<ImageData *> descriptorsMetadata;
	std::set<std::string> classes;
	readImages(files.begin(), files.end(),
			   [&](const std::string & classname, const cv::Mat & descriptors) {
				   classes.insert(classname);
				   descriptorsSet.push_back(descriptors);
				   ImageData *data = new ImageData;
				   data->classname = classname;
				   data->bowFeatures = cv::Mat::zeros(cv::Size(NETWORK_SIZE, 1), CV_32F);
				   for (int j = 0; j < descriptors.rows; j++)
					   descriptorsMetadata.push_back(data);
			   });
	std::cout << "Time elapsed in minutes: " << ((double)cv::getTickCount() - start) / cv::getTickFrequency() / 60.0 << std::endl;

	std::cout << "Creating vocabulary..." << std::endl;
	start = (double)cv::getTickCount();
	cv::Mat labels;
	cv::Mat vocabulary;
	cv::kmeans(descriptorsSet, NETWORK_SIZE, labels, cv::TermCriteria(cv::TermCriteria::EPS + cv::TermCriteria::MAX_ITER, 10, 0.01), 1, cv::KMEANS_PP_CENTERS, vocabulary);
	descriptorsSet.release();
	std::cout << "Time elapsed in minutes: " << ((double)cv::getTickCount() - start) / cv::getTickFrequency() / 60.0 << std::endl;

	std::cout << "Getting histograms of visual words..." << std::endl;
	int *ptrLabels = (int *)(labels.data);
	int size = labels.rows * labels.cols;
	for (int i = 0; i < size; i++) {
		int label = *ptrLabels++;
		ImageData *data = descriptorsMetadata[i];
		data->bowFeatures.at<float>(label)++;
	}

	std::cout << "Preparing neural network..." << std::endl;
	cv::Mat trainSamples;
	cv::Mat trainResponses;
	std::set<ImageData *> uniqueMetadata(descriptorsMetadata.begin(), descriptorsMetadata.end());
	for (auto it = uniqueMetadata.begin(); it != uniqueMetadata.end(); it++) {
		ImageData *data = *it;
		cv::Mat normalizedHist;
		cv::normalize(data->bowFeatures, normalizedHist, 0, data->bowFeatures.rows, cv::NORM_MINMAX, -1, cv::Mat());
		trainSamples.push_back(normalizedHist);
		trainResponses.push_back(getClassCode(classes, data->classname));
		delete *it;
	}
	descriptorsMetadata.clear();

	std::cout << "Training neural network..." << std::endl;
	start = cv::getTickCount();
	cv::Ptr<cv::ml::ANN_MLP> mlp = getTrainedNeuralNetwork(trainSamples, trainResponses);
	std::cout << "Time elapsed in minutes: " << ((double)cv::getTickCount() - start) / cv::getTickFrequency() / 60.0 << std::endl;

	trainSamples.release();
	trainResponses.release();

	std::cout << "Saving models..." << std::endl;
	saveModels(mlp, vocabulary, classes);

	std::cout << "Done" << std::endl;

	return 0;
}
