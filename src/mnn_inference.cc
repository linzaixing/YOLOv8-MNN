#include "mnn_inference.h"

MNNInference::MNNInference(const std::string &model_path, const cv::Size &modelInputShape, int engine_type) {
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
    }

    // create session
    session_ = mnn_net_->createSession(config); 
    mnn_tensor_ = mnn_net_->getSessionInput(session_, "images");
    printf("input b:%d, w:%d, h:%d, c:%d\n", mnn_tensor_->batch(), mnn_tensor_->width(), mnn_tensor_->height(), mnn_tensor_->channel());
}

std::vector<ObjectDetection> MNNInference::MNNRunYolov8Detect(const cv::Mat &src, const cv::Size &modelInputShape) {

    std::vector<float> ratio, dwh;
    cv::Mat origin_img = src.clone();
    LetterBox(ratio, dwh, origin_img, modelInputShape, 117, false, true);

    // uint8 image
    cv::Mat dst, reize_img, rgb_img, src_nchw;
    cv::cvtColor(origin_img, rgb_img, cv::COLOR_BGR2RGB);
    rgb_img.convertTo(rgb_img, CV_32FC3, 1 / 255.0);

    MatToCHW(rgb_img, src_nchw);
    // create temp Tensor
    MNN::Tensor givenTensor(new MNN::Tensor(mnn_tensor_, MNN::Tensor::CAFFE));

    // write data to mnn_tensor_
    memcpy(givenTensor.host<float>(), (float *)src_nchw.data, sizeof(float) * src_nchw.rows * src_nchw.cols * src_nchw.channels());

    // copy to session
    mnn_tensor_->copyFromHostTensor(&givenTensor);

    // run session
    int error_code = mnn_net_->runSession(session_); 
    if (error_code) {
        printf("runSession error!");
    }

    // get output data from tensor
    MNN::Tensor *feature = mnn_net_->getSessionOutput(session_, "output0");

    MNN::Tensor outputUser(feature, feature->getDimensionType());
    feature->copyToHostTensor(&outputUser);
    printf("outputUser b:%d, w:%d, h:%d, c:%d\n", outputUser.batch(), outputUser.width(), outputUser.height(), outputUser.channel());

    float *data = outputUser.host<float>();

    int strideNum = outputUser.height(); //6300
    int signalResultNum = outputUser.channel(); //84

    std::vector<std::vector<float>> boxes;

    for (int i = 0; i < feature->height(); i++) {
        std::vector<float> tmp_data(feature->channel());
        for (int j = 0; j < feature->channel(); j++) {
            int offset = j * feature->height() + i;
            tmp_data[j] = data[offset];
        }
        boxes.emplace_back(tmp_data);
    }

    float rectConfidenceThreshold = 0.5;
    float iouThreshold = 0.6;

    Yolov8DetectPostprocess(boxes, iouThreshold, rectConfidenceThreshold);
    std::vector<ObjectDetection> results;
    for (int i = 0; i < static_cast<int>(boxes.size()); i++) {
        ObjectDetection ob;
        // [x1, y1, x2, y2, conf, class]
        float left = static_cast<float>((boxes[i][0] - dwh[0]) / ratio[0]); 
        float top = static_cast<float>((boxes[i][1] - dwh[1]) / ratio[1]); 
        float w = static_cast<float>((boxes[i][2] - boxes[i][0]) / ratio[0]);
        float h = static_cast<float>((boxes[i][3] - boxes[i][1]) / ratio[1]);
        ob.box = cv::Rect2d(left, top, w, h);
        ob.cls_name = det_coco_classes_[boxes[i][5]];
        ob.confidence = boxes[i][4];
        ob.cls_id = boxes[i][5];
        results.emplace_back(ob);
    }

    return results;

}