#ifndef __UTILS_HPP
#define __UTILS_HPP

#include <string>
#include <functional>

#include <opencv2/core/core.hpp>

#define NETWORK_SIZE 64

struct ImageData {
	std::string classname;
	cv::Mat bowFeatures;
};

typedef std::vector<std::string>::const_iterator vec_iter;

std::vector<std::string> getFilesFromDir(const char *dir);

std::string getClassName(const std::string& filename);

cv::Mat getDescriptors(const cv::Mat& img);

void readImages(vec_iter begin, vec_iter end, std::function<void (const std::string&, const cv::Mat&)> callback);

#endif
