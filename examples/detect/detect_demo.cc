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
        "{help h usage ?   |      | print this message }"
        "{img   i          |      | img path }"
        "{model  m         |      | model path}";
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

    cv::Mat img = cv::imread(img_path);
    if (img.empty()) {
        printf("input image is empty!\n");
        printf("img_path: %s\n", img_path.c_str());
        return -1;
    }

    cv::namedWindow("src_nchw", cv::WINDOW_AUTOSIZE);
    cv::imshow("src_nchw", img);
    cv::waitKey(1000);

    cv::Size new_shape(640, 480);
    int engine_type = 0;

    std::shared_ptr<MNNInference> mnn_inference = std::make_shared<MNNInference>(model_path, new_shape, engine_type);
    
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
