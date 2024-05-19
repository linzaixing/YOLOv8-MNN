//
//  detect_demo.cc
//
//  Created by fanghuilin on 2024/05/16.
//  Copyright © 2024 fanghuilin. All rights reserved.
//

#include <iostream>
#include <string>
#include <opencv2/opencv.hpp>
#include <MNN/Interpreter.hpp>

using namespace cv;

void MatToCHW(const cv::Mat &img, cv::Mat &dst) {
    std::vector<cv::Mat> channels;
    cv::split(img, channels);
    for (auto &img : channels) {
        img = img.reshape(1, 1);
    }
    cv::hconcat(channels, dst);
}

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

    // uint8转浮点
    cv::Mat dst, src_nchw;

    cv::cvtColor(img, img, cv::COLOR_BGR2RGB);
    
    img.convertTo(dst, CV_32FC3, 1 / 255.0);
    cv::resize(dst, dst, cv::Size(480, 640));

    MatToCHW(dst, src_nchw);
    cv::namedWindow("src_nchw", cv::WINDOW_AUTOSIZE);
    cv::imshow("src_nchw", dst);
    cv::waitKey(1000);

    // 加载模型
    // auto net = std::shared_ptr<MNN::Interpreter>(MNN::Interpreter::createFromFile(model_path.c_str()));
    std::shared_ptr<MNN::Interpreter> net(MNN::Interpreter::createFromFile(model_path.c_str()));
    MNN::ScheduleConfig config;
    config.type = MNN_FORWARD_CPU;
    config.numThread = 4;

    // 创建session
    auto session = net->createSession(config); 

    auto input_tensor = net->getSessionInput(session, "images");

    // 创建临时Tensor
    MNN::Tensor givenTensor(new MNN::Tensor(input_tensor, MNN::Tensor::CAFFE));

    // // 数据写入input_tensor
    memcpy(givenTensor.host<float>(), (float *)src_nchw.data, sizeof(float) * src_nchw.rows * src_nchw.cols * src_nchw.channels());

    // 拷贝到到session
    input_tensor->copyFromHostTensor(&givenTensor);

    // 运行模型
    net->runSession(session); 

    // 获取输出数据
    auto detect_output_tensors_ = net->getSessionOutput(session, "output0");
    MNN::Tensor outputUser(detect_output_tensors_, net->getSessionOutput(session, nullptr)->getDimensionType());
    net->getSessionOutput(session, nullptr)->copyToHostTensor(&outputUser);

    auto score = outputUser.host<float>()[0];
    auto index = outputUser.host<float>()[1];

    return 0;
}
