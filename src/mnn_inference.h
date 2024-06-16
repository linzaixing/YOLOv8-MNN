#pragma once

#include <string>
#include <vector>

#include <MNN/Interpreter.hpp>

struct Detection {
    int cls_id;
    std::string cls_name;
    float confidence;
    cv::Rect2d box;
    std::vector<float> scores;
};

class MNNInference {
public:
    MNNInference(const std::string &model_path, const cv::Size &modelInputShape = {640, 640}, int engine_type = 0);
    std::vector<Detection> MNNRunYolov8Detect(const cv::Mat &src);

private:
    std::shared_ptr<MNN::Interpreter> mnn_net_ = nullptr;
    MNN::Tensor * mnn_tensor_ = nullptr;
}