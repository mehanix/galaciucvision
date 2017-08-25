#ifndef __UTILS_HPP
#define __UTILS_HPP

#include <string>
#include <functional>

#include <opencv2/core/core.hpp>

#define NETWORK_SIZE 128
#define MAX_IMAGE_WIDTH 1024

std::vector<std::string> getFilesFromDir(const char *dir);

cv::Mat readImageDescriptors(const std::string& file);

#endif // !__UTILS_HPP
