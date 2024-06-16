#include "mnn_inference.h"

MNNInference::MNNInference(const std::string &model_path, const cv::Size &modelInputShape = {640, 640}, int engine_type = 0) {
    // load MNN model
    mnn_net_ = std::shared_ptr<MNN::Interpreter>(MNN::Interpreter::createFromFile(model_path.c_str()));
    MNN::ScheduleConfig config;
    if (engine_type == 0) {
        config.type = MNN_FORWARD_CPU;
        config.numThread = 4;
    } else if (engine_type == 1) {
        config.type = MNN_FORWARD_OPENCL;
    } else {
        printf("current engine type not support!");
        return;
    }

    // create session
    auto session = mnn_net_->createSession(config); 
    mnn_tensor_ = mnn_net_->getSessionInput(session, "images");
    printf("input b:%d, w:%d, h:%d, c:%d\n", input_tensor->batch(), input_tensor->width(), input_tensor->height(), input_tensor->channel());
}

std::vector<Detection> MNNInference::MNNRunYolov8Detect(const cv::Mat &src) {

}