#ifndef __UTILS_HPP
#define __UTILS_HPP

#include <string>
#include <functional>

#include <opencv2/core/core.hpp>

#define NETWORK_SIZE 64
#define MAX_IMAGE_WIDTH 1024

struct ImageData {
	std::string classname;
	cv::Mat bowFeatures;
};

typedef std::vector<std::string>::const_iterator vector_iterator;

std::vector<std::string> getFilesFromDir(const char *dir);

std::string getClassName(const std::string& filename);

cv::Mat getDescriptors(const cv::Mat& img);

void readImages(vector_iterator begin, vector_iterator end, std::function<void (const std::string&, const cv::Mat&)> callback, bool printProgress = true);

#endif // !__UTILS_HPP
