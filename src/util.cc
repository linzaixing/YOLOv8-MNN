#include "util.h"

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
