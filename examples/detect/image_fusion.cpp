
#include <MNN/Interpreter.hpp>
#include <MNN/ImageProcess.hpp>
#include <opencv2/opencv.hpp>
#include <iostream>

void MatToCHW(const cv::Mat &img, cv::Mat &dst) {
    std::vector<cv::Mat> channels;
    cv::split(img, channels);
    for (auto &img : channels) {
        img = img.reshape(1, 1);
    }
    cv::hconcat(channels, dst);
}

void TensorToMat(MNN::Tensor &tensor, cv::Mat& img) {
    int height = img.rows;
    int width = img.cols;
    int channel = img.channels();
    int single_channel_size = height * width;
    std::vector<cv::Mat> bgr;
    for (int i = 0; i < channel; ++i) {
        cv::Mat tmp = cv::Mat::zeros(height, width, CV_32FC1);
        std::memcpy((float *)tmp.data, tensor.host<float>() + i * single_channel_size, sizeof(float) * single_channel_size);
        bgr.emplace_back(tmp);
    }
    cv::merge(bgr, img);
}

int main() {
    // 读取图像
    cv::Mat fore_img = cv::imread("E:/CProject/YOLOv8-MNN/input/jump.jpg", cv::IMREAD_UNCHANGED);
    cv::Mat back_img = cv::imread("E:/CProject/YOLOv8-MNN/input/seaside.jpg", cv::IMREAD_UNCHANGED);
    cv::Mat alpha_img = cv::imread("E:/CProject/YOLOv8-MNN/input/mask.jpg", cv::IMREAD_GRAYSCALE);

    // uint8转浮点
    fore_img.convertTo(fore_img, CV_32FC3, 1 / 255.0);  //BGR
    back_img.convertTo(back_img, CV_32FC3, 1 / 255.0);
    alpha_img.convertTo(alpha_img, CV_32FC1, 1 / 255.0);

    // NHWC转NCHW
    cv::Mat fg_chw, bg_chw, alpha_chw;
    MatToCHW(fore_img, fg_chw);
    MatToCHW(back_img, bg_chw);
    MatToCHW(alpha_img, alpha_chw);

    // 加载模型
    std::shared_ptr<MNN::Interpreter> net(MNN::Interpreter::createFromFile("E:/CProject/YOLOv8-MNN/input/model.mnn"));
    MNN::ScheduleConfig config;
    config.type = MNN_FORWARD_CPU;

    // 创建session
    auto session = net->createSession(config); 

    auto input_tensor1_ = net->getSessionInput(session, "fg_input");
    auto input_tensor2_ = net->getSessionInput(session, "bg_input");
    auto input_tensor3_ = net->getSessionInput(session, "alpha_input");

    // 创建临时Tensor
    MNN::Tensor fg_nhwc_tensor_tmp(new MNN::Tensor(input_tensor1_, MNN::Tensor::CAFFE));
    MNN::Tensor bg_nhwc_tensor_tmp(new MNN::Tensor(input_tensor2_, MNN::Tensor::CAFFE));
    MNN::Tensor alpha_nhwc_tensor_tmp(new MNN::Tensor(input_tensor3_, MNN::Tensor::CAFFE));

    // 数据写入input_tensor
    memcpy(fg_nhwc_tensor_tmp.host<float>(), (float *)fg_chw.data, sizeof(float) * fg_chw.rows * fg_chw.cols * fg_chw.channels());
    memcpy(bg_nhwc_tensor_tmp.host<float>(), (float *)bg_chw.data, sizeof(float) * bg_chw.rows * bg_chw.cols * bg_chw.channels());
    memcpy(alpha_nhwc_tensor_tmp.host<float>(),
           (float *)alpha_chw.data,
           sizeof(float) * alpha_chw.rows * alpha_chw.cols * alpha_chw.channels());

    // 拷贝到到session
    input_tensor1_->copyFromHostTensor(&fg_nhwc_tensor_tmp);
    input_tensor2_->copyFromHostTensor(&bg_nhwc_tensor_tmp);
    input_tensor3_->copyFromHostTensor(&alpha_nhwc_tensor_tmp);
    
    net->resizeTensor(input_tensor1_, {3, 512, 288});
    net->resizeTensor(input_tensor2_, {3, 512, 288});
    net->resizeTensor(input_tensor3_, {1, 512, 288});
    net->resizeSession(session);
    net->releaseModel();
    
    // 运行模型
    net->runSession(session); 
    
    // 获取输出数据
    auto detect_output_tensors_ = net->getSessionOutput(session, "fusion_output");
    MNN::Tensor outputUser(detect_output_tensors_, net->getSessionOutput(session, nullptr)->getDimensionType());
    net->getSessionOutput(session, nullptr)->copyToHostTensor(&outputUser);

    // Tensor数据转回NHWC
    cv::Mat output(fore_img.size(), CV_32FC3);
    TensorToMat(outputUser, output);
    output.convertTo(output, CV_8UC3, 255);

    cv::imwrite("fusion.jpg", output); 

    return 0;
}

