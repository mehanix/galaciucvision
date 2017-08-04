#include "utils.hpp"

#include <iostream>
#include <fstream>
#include <dirent.h>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#define IMAGE_WIDTH 1024

std::vector<std::string> getFilesFromDir(const char *dir)
{
	DIR *dp;
	std::vector<std::string> files;
	struct dirent *dirp;
	if ((dp = opendir(dir)) == NULL) {
		std::cout << "Could not open " << dir << std::endl;
		exit(-1);
	}

	while ((dirp = readdir(dp)) != NULL)
		if (strcmp(dirp->d_name, "..") && strcmp(dirp->d_name, "."))
			files.push_back(std::string(dir) + "/" + std::string(dirp->d_name));
	closedir(dp);
	return files;
}


/**
 * Extract the class name from a file name
 */
std::string getClassName(const std::string& filename)
{
	return filename.substr(filename.find_last_of('/') + 1, 2);
}

/**
 * Read images from a list of file names and returns, for each read image,
 * its class name and its local descriptors
 */
void readImages(vec_iter begin, vec_iter end, std::function<void (const std::string&, const cv::Mat&)> callback)
{
	for (auto it = begin; it != end; ++it) {
		std::string filename = *it;
		std::cout << "Reading " << filename << std::endl;
		cv::Mat img = cv::imread(filename, 0);
		if (img.empty()) {
			std::cerr << "WARNING: Could not read image." << std::endl;
			continue;
		}

		// Resize image
		int w = img.size().width;
		int h = img.size().height * IMAGE_WIDTH / (float)w;
		w = IMAGE_WIDTH;
		cv::Size size(w, h);
		cv::Mat smallImg;
		cv::resize(img, smallImg, size);

		std::string classname = getClassName(filename);
		cv::Mat descriptors = getDescriptors(smallImg);
		callback(classname, descriptors);
	}
}

/**
 * Extract local features for an image
 */
cv::Mat getDescriptors(const cv::Mat& img)
{
	cv::Ptr<cv::KAZE> kaze = cv::KAZE::create();
	std::vector<cv::KeyPoint> keypoints;
	cv::Mat descriptors;
	kaze->detectAndCompute(img, cv::noArray(), keypoints, descriptors);
	return descriptors;
}
