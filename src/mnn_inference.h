#pragma once

#include <string>
#include <vector>
#include <opencv2/opencv.hpp>
#include <MNN/Interpreter.hpp>
#include "util.h"

struct ObjectDetection {
    int cls_id;
    std::string cls_name;
    float confidence;
    cv::Rect2d box;
    std::vector<float> scores;
};

class MNNInference {
public:
    MNNInference(const std::string &model_path, const cv::Size &modelInputShape = {640, 640}, int engine_type = 0);
    std::vector<ObjectDetection> MNNRunYolov8Detect(const cv::Mat &src, const cv::Size &modelInputShape);

private:
    std::shared_ptr<MNN::Interpreter> mnn_net_ = nullptr;
    MNN::Tensor * mnn_tensor_ = nullptr;
    MNN::Session * session_ = nullptr;
    std::vector<std::string> det_coco_classes_ {
        "person",         "bicycle",    "car",           "motorcycle",    "airplane",     "bus",           "train",
        "truck",          "boat",       "traffic light", "fire hydrant",  "stop sign",    "parking meter", "bench",
        "bird",           "cat",        "dog",           "horse",         "sheep",        "cow",           "elephant",
        "bear",           "zebra",      "giraffe",       "backpack",      "umbrella",     "handbag",       "tie",
        "suitcase",       "frisbee",    "skis",          "snowboard",     "sports ball",  "kite",          "baseball bat",
        "baseball glove", "skateboard", "surfboard",     "tennis racket", "bottle",       "wine glass",    "cup",
        "fork",           "knife",      "spoon",         "bowl",          "banana",       "apple",         "sandwich",
        "orange",         "broccoli",   "carrot",        "hot dog",       "pizza",        "donut",         "cake",
        "chair",          "couch",      "potted plant",  "bed",           "dining table", "toilet",        "tv",
        "laptop",         "mouse",      "remote",        "keyboard",      "cell phone",   "microwave",     "oven",
        "toaster",        "sink",       "refrigerator",  "book",          "clock",        "vase",          "scissors",
        "teddy bear",     "hair drier", "toothbrush"};
};