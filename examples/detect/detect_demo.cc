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

struct ObjectDetection {
    int cls_id;
    std::string cls_name;
    float confidence;
    cv::Rect2d box;
    std::vector<float> scores;
};

void MatToCHW(const cv::Mat &img, cv::Mat &dst) {
    if (img.empty()) {
        printf("error, img is empty!");
        return;
    }
    std::vector<cv::Mat> channels;
    cv::split(img, channels);
    for (auto &img : channels) {
        // reshape images to 1 row (rows*cols) cols
        img = img.reshape(0, 1);
    }
    // The horizontal stitching 3 channels to 1 row
    cv::hconcat(channels, dst);
}

void LetterBox(std::vector<float> &ratio, std::vector<float> &dwh, cv::Mat &img, cv::Size new_shape, int color, bool scale_fill, bool scale_up) {
    float r = std::fmin(static_cast<float>(new_shape.width) / img.cols, static_cast<float>(new_shape.height) / img.rows);
    if (!scale_up)
        r = std::fmin(r, 1.f);
    ratio.resize(2);
    ratio[0] = r;
    ratio[1] = r;
    std::vector<int> new_unpad = {static_cast<int>(std::round(img.cols * r)), 
        static_cast<int>(std::round(img.rows * r))};  

    dwh.resize(2); //[dw, dh]
    dwh[0] = new_shape.width - new_unpad[0];
    dwh[1] = new_shape.height - new_unpad[1];

    if (scale_fill) {
        dwh[0] = 0.f;
        dwh[1] = 0.f;
        new_unpad[0] = new_shape.height;
        new_unpad[1] = new_shape.width;
        ratio[0] = new_shape.width / img.cols;
        ratio[1] = new_shape.height / img.rows;
    }

    dwh[0] /= 2.f;
    dwh[1] /= 2.f;

    if (img.cols != new_unpad[0] || img.rows != new_unpad[1]) {
        cv::resize(img, img, cv::Size(new_unpad[0], new_unpad[1]), cv::INTER_LINEAR);
    }

    int left = static_cast<int>(std::round(dwh[0] - 0.1));
    int right = static_cast<int>(std::round(dwh[0] + 0.1));
    int top = static_cast<int>(std::round(dwh[1] - 0.1));
    int bottom = static_cast<int>(std::round(dwh[1] + 0.1));

    cv::copyMakeBorder(img, img, top, bottom, left, right, cv::BORDER_CONSTANT, cv::Scalar(color, color, color));

    return;
}

void Yolov8DetectPostprocess(std::vector<std::vector<float>> &boxes, float nms_thresh, float conf_thresh) {
    std::vector<std::vector<float>> tmp_boxes = boxes;
    std::vector<float> confs;
    std::vector<float> classIds;
    std::vector<cv::Rect> rectBoxes;

    boxes.clear();

    for (int i = 0; i < static_cast<int>(tmp_boxes.size()); i++) {
        std::vector<float> tmp_box = tmp_boxes[i];
        float bestConf = 0;
        int bestClassId = 0;
        for (int j = 4; j < static_cast<int>(tmp_box.size()); j++){
            if (tmp_box[j] > bestConf){
                bestConf = tmp_box[j];
                bestClassId = j - 4;
            }
        }
        if (bestConf > conf_thresh){
            int centerX = (int)(tmp_box[0]);
            int centerY = (int)(tmp_box[1]);
            int width = (int)(tmp_box[2]);
            int height = (int)(tmp_box[3]);
            int left = centerX - width / 2;
            int top = centerY - height / 2;

            confs.emplace_back(bestConf);
            classIds.emplace_back(bestClassId);
            rectBoxes.emplace_back(left, top, width, height);
        }
    }

    std::vector<int> indices;
    cv::dnn::NMSBoxes(rectBoxes, confs, conf_thresh, nms_thresh, indices);

    std::vector<std::vector<float>> outputBoxes;
    for (int idx : indices){

        float x1 = static_cast<float>(rectBoxes[idx].x);
        float y1 = static_cast<float>(rectBoxes[idx].y);
        float x2 = static_cast<float>(x1 + rectBoxes[idx].width);
        float y2 = static_cast<float>(y1 + rectBoxes[idx].height);

        std::vector<float> currOutBox= {x1, y1, x2, y2, confs[idx], static_cast<float>(classIds[idx])};

        outputBoxes.emplace_back(currOutBox);
    }
    swap(boxes, outputBoxes);
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

    cv::namedWindow("src_nchw", cv::WINDOW_AUTOSIZE);
    cv::imshow("src_nchw", img);
    cv::waitKey(1000);

    cv::Size new_shape(640, 480);
    int engine_type = 0;

    // load MNN model
    auto mnn_net_ = std::shared_ptr<MNN::Interpreter>(MNN::Interpreter::createFromFile(model_path.c_str()));
    MNN::ScheduleConfig config;
    if (engine_type == 0) {
        config.type = MNN_FORWARD_CPU;
        config.numThread = 4;
    } else if (engine_type == 1) {
        config.type = MNN_FORWARD_OPENCL;
    } else {
        printf("current engine type not support!");
        return -1;
    }

    // create session
    auto session = mnn_net_->createSession(config); 
    auto input_tensor = mnn_net_->getSessionInput(session, "images");
    printf("input b:%d, w:%d, h:%d, c:%d\n", input_tensor->batch(), input_tensor->width(), input_tensor->height(), input_tensor->channel());
    
    std::vector<std::string> det_coco_classes{
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

    std::vector<float> ratio, dwh;
    cv::Mat origin_img = img.clone();
    LetterBox(ratio, dwh, img, new_shape, 117, false, true);

    // uint8 image
    cv::Mat dst, reize_img, rgb_img, src_nchw;

    cv::cvtColor(img, rgb_img, cv::COLOR_BGR2RGB);
    rgb_img.convertTo(rgb_img, CV_32FC3, 1 / 255.0);

    MatToCHW(rgb_img, src_nchw);

    // create temp Tensor
    MNN::Tensor givenTensor(new MNN::Tensor(input_tensor, MNN::Tensor::CAFFE));

    // write data to input_tensor
    memcpy(givenTensor.host<float>(), (float *)src_nchw.data, sizeof(float) * src_nchw.rows * src_nchw.cols * src_nchw.channels());

    // copy to session
    input_tensor->copyFromHostTensor(&givenTensor);

    // run session
    int error_code = mnn_net_->runSession(session); 
    if (error_code) {
        return -1;
    }

    // get output data from tensor
    MNN::Tensor *feature = mnn_net_->getSessionOutput(session, "output0");

    MNN::Tensor outputUser(feature, feature->getDimensionType());
    feature->copyToHostTensor(&outputUser);
    printf("outputUser b:%d, w:%d, h:%d, c:%d\n", outputUser.batch(), outputUser.width(), outputUser.height(), outputUser.channel());

    float *data = outputUser.host<float>();

    std::vector<std::vector<float>> boxes;

    for (int i = 0; i < feature->height(); i++) {
        std::vector<float> tmp_data(feature->channel());
        for (int j = 0; j < feature->channel(); j++) {
            int offset = j * feature->height() + i;
            tmp_data[j] = data[offset];
        }
        boxes.emplace_back(tmp_data);
    }

    double rectConfidenceThreshold = 0.5;
    double iouThreshold = 0.1;

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
        ob.cls_name = det_coco_classes[boxes[i][5]];
        ob.confidence = boxes[i][4];
        ob.cls_id = boxes[i][5];
        results.emplace_back(ob);
    }

    int strideNum = outputUser.height(); //6300
    int signalResultNum = outputUser.channel(); //84


    for (int i = 0; i < results.size(); i++) {
        auto &det = results[i];
        float left = det.box.x; 
        float top = det.box.y;
        float width = det.box.width;
        float height = det.box.height;
        printf("class: %s, score: %f\n", det.cls_name.c_str(), det.confidence);
        printf("box coordinate left, top, width, height: [%lf, %lf, %lf, %lf]\n", left, top, width, height);

        cv::rectangle(origin_img, cv::Point(left, top), cv::Point(left + width, top + height), {250, 50, 50}, 2, cv::LINE_AA);
        std::stringstream text;
        text << det.cls_name << " " << std::setiosflags(std::ios::fixed)<<std::setprecision(1) << det.confidence * 100.f << "%";
        cv::putText(origin_img, text.str(), cv::Point(left - 20.0, top - 10.0), cv::FONT_HERSHEY_SIMPLEX, 0.5, {50, 50, 255}, 1, cv::LINE_AA);
    }

    // cv::imwrite(save_path, img);
    std::cout << "Press any key to exit" << std::endl;
    cv::imshow("Result of Detection", origin_img);
    cv::waitKey(0);
    cv::destroyAllWindows();

    return 0;
}
