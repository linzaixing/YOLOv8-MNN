//
//  detect_demo.cc
//
//  Created by fanghuilin on 2024/05/16.
//  Copyright © 2024 fanghuilin. All rights reserved.
//

#include <iostream>
#include <string>

#include "mnn_inference.h"

using namespace cv;

int main(int argc, const char* argv[]) {
    const String keys =
        "{help h usage ?   |      | print this message   }"
        "{img   i          |<none>| img path }"
        "{model  m         |<none>| model path}";
    CommandLineParser parser(argc, argv, keys);
    if (!parser.check()) {
        parser.printErrors();
        return -1;
    }
    if (!parser.has("img")) {
        printf("error, invalid img param\n");
        return -1;
    }
    if (!parser.has("model")) {
        printf("error, invalid model param\n");
        return -1;
    }

    std::string img_path = parser.get<String>("img");
    std::string model_path = parser.get<String>("model");

    cv::Mat img = cv::imread(img_path, cv::IMREAD_UNCHANGED);
    if (img.empty()) {
        printf("input image is empty!\n");
        return -1;
    }

    cv::namedWindow("src_nchw", cv::WINDOW_AUTOSIZE);
    cv::imshow("src_nchw", img);
    cv::waitKey(1000);

    cv::Size new_shape(640, 480);
    int engine_type = 0;

    std::shared_ptr<MNNInference> mnn_inference = std::make_shared<MNNInference>(model_path, new_shape, engine_type);

    // // load MNN model
    // auto mnn_net_ = std::shared_ptr<MNN::Interpreter>(MNN::Interpreter::createFromFile(model_path.c_str()));
    // MNN::ScheduleConfig config;
    // if (engine_type == 0) {
    //     config.type = MNN_FORWARD_CPU;
    //     config.numThread = 4;
    // } else if (engine_type == 1) {
    //     config.type = MNN_FORWARD_OPENCL;
    // } else {
    //     printf("current engine type not support!");
    //     return -1;
    // }

    // // create session
    // auto session = mnn_net_->createSession(config); 
    // auto input_tensor = mnn_net_->getSessionInput(session, "images");
    // printf("input b:%d, w:%d, h:%d, c:%d\n", input_tensor->batch(), input_tensor->width(), input_tensor->height(), input_tensor->channel());
    
    // std::vector<std::string> det_coco_classes{
    //     "person",         "bicycle",    "car",           "motorcycle",    "airplane",     "bus",           "train",
    //     "truck",          "boat",       "traffic light", "fire hydrant",  "stop sign",    "parking meter", "bench",
    //     "bird",           "cat",        "dog",           "horse",         "sheep",        "cow",           "elephant",
    //     "bear",           "zebra",      "giraffe",       "backpack",      "umbrella",     "handbag",       "tie",
    //     "suitcase",       "frisbee",    "skis",          "snowboard",     "sports ball",  "kite",          "baseball bat",
    //     "baseball glove", "skateboard", "surfboard",     "tennis racket", "bottle",       "wine glass",    "cup",
    //     "fork",           "knife",      "spoon",         "bowl",          "banana",       "apple",         "sandwich",
    //     "orange",         "broccoli",   "carrot",        "hot dog",       "pizza",        "donut",         "cake",
    //     "chair",          "couch",      "potted plant",  "bed",           "dining table", "toilet",        "tv",
    //     "laptop",         "mouse",      "remote",        "keyboard",      "cell phone",   "microwave",     "oven",
    //     "toaster",        "sink",       "refrigerator",  "book",          "clock",        "vase",          "scissors",
    //     "teddy bear",     "hair drier", "toothbrush"};

    // // create temp Tensor
    // MNN::Tensor givenTensor(new MNN::Tensor(input_tensor, MNN::Tensor::CAFFE));

    // // write data to input_tensor
    // memcpy(givenTensor.host<float>(), (float *)src_nchw.data, sizeof(float) * src_nchw.rows * src_nchw.cols * src_nchw.channels());

    // // copy to session
    // input_tensor->copyFromHostTensor(&givenTensor);

    // // run session
    // int error_code = mnn_net_->runSession(session); 
    // if (error_code) {
    //     return -1;
    // }

    // // get output data from tensor
    // MNN::Tensor *feature = mnn_net_->getSessionOutput(session, "output0");

    // MNN::Tensor outputUser(feature, feature->getDimensionType());
    // feature->copyToHostTensor(&outputUser);
    // printf("outputUser b:%d, w:%d, h:%d, c:%d\n", outputUser.batch(), outputUser.width(), outputUser.height(), outputUser.channel());

    // float *data = outputUser.host<float>();

    
    std::vector<ObjectDetection> results = mnn_inference->MNNRunYolov8Detect(img, new_shape);

    for (int i = 0; i < results.size(); i++) {
        auto &det = results[i];
        float left = det.box.x; 
        float top = det.box.y;
        float width = det.box.width;
        float height = det.box.height;
        printf("class: %s, score: %f\n", det.cls_name.c_str(), det.confidence);
        printf("box coordinate left, top, width, height: [%lf, %lf, %lf, %lf]\n", left, top, width, height);

        cv::rectangle(img, cv::Point(left, top), cv::Point(left + width, top + height), {250, 50, 50}, 2, cv::LINE_AA);
        std::stringstream text;
        text << det.cls_name << " " << std::setiosflags(std::ios::fixed)<<std::setprecision(1) << det.confidence * 100.f << "%";
        cv::putText(img, text.str(), cv::Point(left - 20.0, top - 10.0), cv::FONT_HERSHEY_SIMPLEX, 0.5, {50, 50, 255}, 1, cv::LINE_AA);
    }

    // cv::imwrite(save_path, img);
    std::cout << "Press any key to exit" << std::endl;
    cv::imshow("Result of Detection", img);
    cv::waitKey(0);
    cv::destroyAllWindows();

    return 0;
}
