#include <vector>
#include <algorithm>
#include <functional>
#include <map>
#include <set>
#include <fstream>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/ml/ml.hpp>

#include "utils.hpp"

/**
 * Transform a class name into an id
 */
int getClassId(const std::set<std::string>& classes, const std::string& classname)
{
	int index = 0;
	for (auto it = classes.begin(); it != classes.end(); ++it) {
		if (*it == classname) break;
		++index;
	}
	return index;
}

/**
 * Get a binary code associated to a class
 */
cv::Mat getClassCode(const std::set<std::string>& classes, const std::string& classname)
{
	cv::Mat code = cv::Mat::zeros(cv::Size((int)classes.size(), 1), CV_32F);
	int index = getClassId(classes, classname);
	code.at<float>(index) = 1;
	return code;
}

/**
 * Get a trained neural network according to some inputs and outputs
 */
cv::Ptr<cv::ml::ANN_MLP> getTrainedNeuralNetwork(const cv::Mat& trainSamples, const cv::Mat& trainResponses)
{
	int networkInputSize = trainSamples.cols;
	int networkOutputSize = trainResponses.cols;
	cv::Ptr<cv::ml::ANN_MLP> mlp = cv::ml::ANN_MLP::create();
	std::vector<int> layerSizes = { networkInputSize, networkInputSize / 2,
									networkOutputSize };
	mlp->setLayerSizes(layerSizes);
	mlp->setActivationFunction(cv::ml::ANN_MLP::SIGMOID_SYM);
	mlp->train(trainSamples, cv::ml::ROW_SAMPLE, trainResponses);
	return mlp;
}

/**
 * Save our obtained models (neural network, bag of words vocabulary
 * and class names) to use it later
 */
void saveModels(cv::Ptr<cv::ml::ANN_MLP> mlp, const cv::Mat& vocabulary,
				const std::set<std::string>& classes)
{
	mlp->save("mlp.yaml");
	cv::FileStorage fs("vocabulary.yaml", cv::FileStorage::WRITE);
	fs << vocabulary;
	fs.release();
	std::ofstream classesOutput("classes.txt");
	classesOutput << classes.size();
	for (auto it = classes.begin(); it != classes.end(); ++it)
		classesOutput << *it;
	classesOutput << std::endl;
	classesOutput.close();
}

int main(int argc, char **argv)
{
	if (argc != 3) {
		std::cerr << "Usage: <IMAGES> <NETWORK_INPUT_LAYER_SIZE>" << std::endl;
		exit(-1);
	}
	char *dir = argv[1];
	int networkInputSize = atoi(argv[2]);

	std::cout << "Reading training set..." << std::endl;
	double start = (double)cv::getTickCount();
	std::vector<std::string> files = getFilesFromDir(dir);
	std::random_shuffle(files.begin(), files.end());

	cv::Mat descriptorsSet;
	std::vector<ImageData *> descriptorsMetadata;
	std::set<std::string> classes;
	readImages(files.begin(), files.end(),
			   [&](const std::string & classname, const cv::Mat & descriptors) {
	               // Append to the set of classes
				   classes.insert(classname);
	               // Append to the list of descriptors
				   descriptorsSet.push_back(descriptors);
	               // Append metadata to each extracted feature
				   ImageData *data = new ImageData;
				   data->classname = classname;
				   data->bowFeatures = cv::Mat::zeros(cv::Size(networkInputSize, 1), CV_32F);
				   for (int j = 0; j < descriptors.rows; j++)
					   descriptorsMetadata.push_back(data);
			   });
	std::cout << "Time elapsed in minutes: " << ((double)cv::getTickCount() - start) / cv::getTickFrequency() / 60.0 << std::endl;

	std::cout << "Creating vocabulary..." << std::endl;
	start = (double)cv::getTickCount();
	cv::Mat labels;
	cv::Mat vocabulary;
	// Use k-means to find k centroids (the words of our vocabulary)
	cv::kmeans(descriptorsSet, networkInputSize, labels, cv::TermCriteria(cv::TermCriteria::EPS +
																		  cv::TermCriteria::MAX_ITER, 10, 0.01), 1, cv::KMEANS_PP_CENTERS, vocabulary);
	// No need to keep it on memory anymore
	descriptorsSet.release();
	std::cout << "Time elapsed in minutes: " << ((double)cv::getTickCount() - start) / cv::getTickFrequency() / 60.0 << std::endl;

	// Convert a set of local features for each image in a single descriptors
	// using the bag of words technique
	std::cout << "Getting histograms of visual words..." << std::endl;
	int *ptrLabels = (int *)(labels.data);
	int size = labels.rows * labels.cols;
	for (int i = 0; i < size; i++) {
		int label = *ptrLabels++;
		ImageData *data = descriptorsMetadata[i];
		data->bowFeatures.at<float>(label)++;
	}

	// Filling matrixes to be used by the neural network
	std::cout << "Preparing neural network..." << std::endl;
	cv::Mat trainSamples;
	cv::Mat trainResponses;
	std::set<ImageData *> uniqueMetadata(descriptorsMetadata.begin(), descriptorsMetadata.end());
	for (auto it = uniqueMetadata.begin(); it != uniqueMetadata.end(); ) {
		ImageData *data = *it;
		cv::Mat normalizedHist;
		cv::normalize(data->bowFeatures, normalizedHist, 0, data->bowFeatures.rows, cv::NORM_MINMAX, -1, cv::Mat());
		trainSamples.push_back(normalizedHist);
		trainResponses.push_back(getClassCode(classes, data->classname));
		delete *it;   // clear memory
		it++;
	}
	descriptorsMetadata.clear();

	// Training neural network
	std::cout << "Training neural network..." << std::endl;
	start = cv::getTickCount();
	cv::Ptr<cv::ml::ANN_MLP> mlp = getTrainedNeuralNetwork(trainSamples, trainResponses);
	std::cout << "Time elapsed in minutes: " << ((double)cv::getTickCount() - start) / cv::getTickFrequency() / 60.0 << std::endl;

	// We can clear memory now
	trainSamples.release();
	trainResponses.release();

	// Save models
	std::cout << "Saving models..." << std::endl;
	saveModels(mlp, vocabulary, classes);

	std::cout << "Done" << std::endl;

	return 0;
}
