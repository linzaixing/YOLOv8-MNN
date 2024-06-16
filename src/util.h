#pragma once

#include <vector>
#include <opencv2/opencv.hpp>

void MatToCHW(const cv::Mat &img, cv::Mat &dst);

void LetterBox(std::vector<float> &ratio, std::vector<float> &dwh, cv::Mat &img, cv::Size new_shape, int color, bool scale_fill, bool scale_up);

void Yolov8DetectPostprocess(std::vector<std::vector<float>> &boxes, float nms_thresh, float conf_thresh);
