#include "utils.hpp"

#include <iostream>
#include <fstream>
#include <dirent.h>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/imgproc/imgproc.hpp>

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

std::string getClassName(const std::string& filename)
{
	return filename.substr(filename.find_last_of('/') + 1, 2);
}

void readImages(vector_iterator begin, vector_iterator end, std::function<void (const std::string&, const cv::Mat&)> callback, bool printProgress)
{
	vector_iterator it;
	for (it = begin; it != end; ++it) {
		std::string file = *it;
		if (printProgress) std::cout << "Reading " << file << std::endl;
		cv::Mat img = cv::imread(file, 0);
		if (img.empty()) {
			std::cout << "Could not read image." << std::endl;
			continue;
		}

		std::string classname = getClassName(file);
		cv::Mat descriptors;

		// Resize image if needed
		int w, h;
		w = img.size().width;
		h = img.size().height;
		if (w > MAX_IMAGE_WIDTH) {
			h = img.size().height * MAX_IMAGE_WIDTH / (float)w;
			w = MAX_IMAGE_WIDTH;
			cv::Size size(w, h);
			cv::Mat resizedImg;
			cv::resize(img, resizedImg, size);
			descriptors = getDescriptors(resizedImg);
		} else {
			descriptors = getDescriptors(img);
		}

		callback(classname, descriptors);
	}
}

cv::Mat getDescriptors(const cv::Mat& img)
{
	cv::Ptr<cv::KAZE> kaze = cv::KAZE::create();
	std::vector<cv::KeyPoint> keypoints;
	cv::Mat descriptors;
	kaze->detectAndCompute(img, cv::noArray(), keypoints, descriptors);
	return descriptors;
}
