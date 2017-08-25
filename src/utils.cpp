#include "utils.hpp"

#include <iostream>
#include <fstream>
#include <dirent.h>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/imgproc/imgproc.hpp>


bool isPathValid(const char *path)
{
	int len = strlen(path), extlen;
	const char *ext[] = { ".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG" };
	for (size_t i = 0; i < sizeof(ext) / sizeof(ext[0]); i++) {
		extlen = strlen(ext[i]);
		if (len >= extlen && !strcmp(path + len - extlen, ext[i])) return true;
	}
	return false;
}

std::vector<std::string> getFilesFromDir(const char *dir)
{
	DIR *dp;
	std::vector<std::string> files;
	struct dirent *dirp;
	if ((dp = opendir(dir)) == NULL) throw "Could not open " + std::string(dir);

	while ((dirp = readdir(dp)) != NULL)
		if (strcmp(dirp->d_name, "..") && strcmp(dirp->d_name, "."))
			if (isPathValid(dirp->d_name))
				files.push_back(std::string(dir) + "/" + std::string(dirp->d_name));

	closedir(dp);
	return files;
}

cv::Mat getDescriptors(const cv::Mat& img)
{
	cv::Ptr<cv::AKAZE> akaze = cv::AKAZE::create(cv::AKAZE::DESCRIPTOR_KAZE, 0, 1);
	std::vector<cv::KeyPoint> keypoints;
	cv::Mat descriptors;
	akaze->detectAndCompute(img, cv::noArray(), keypoints, descriptors);
	return descriptors;
}

cv::Mat readImageDescriptors(const std::string& file)
{
	// Check for cache
	std::string cachePath = file + ".cache.yml";
	cv::FileStorage cachefs(cachePath, cv::FileStorage::READ);
	cv::Mat descriptors;
	cachefs["descriptors"] >> descriptors;
	if (descriptors.empty()) {
		cv::Mat img = cv::imread(file, 0);
		if (img.empty()) throw "Could not read image";

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
			img = resizedImg;
		}

		descriptors = getDescriptors(img);
		cachefs.open(cachePath, cv::FileStorage::WRITE);
		cachefs << "descriptors" << descriptors;
	}
	cachefs.release();
	return descriptors;
}
