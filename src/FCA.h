#ifndef FACEDETECTION_FCA_H
#define FACEDETECTION_FCA_H

#include <map>
#include <vector>
#include <string>
#include <iostream>
#include <fstream>

#include "opencv2/opencv.hpp"
#include "../include/json.hpp"

using json = nlohmann::json;

const std::map<std::string, int> str2backend{
        {"opencv", cv::dnn::DNN_BACKEND_OPENCV}, {"cuda", cv::dnn::DNN_BACKEND_CUDA},
        {"timvx",  cv::dnn::DNN_BACKEND_TIMVX},  {"cann", cv::dnn::DNN_BACKEND_CANN}
};
const std::map<std::string, int> str2target{
        {"cpu", cv::dnn::DNN_TARGET_CPU}, {"cuda", cv::dnn::DNN_TARGET_CUDA},
        {"npu", cv::dnn::DNN_TARGET_NPU}, {"cuda_fp16", cv::dnn::DNN_TARGET_CUDA_FP16}
};


void saveEmbedding (const std::string& name, const cv::Mat& embedding);
float cosineSimilarity(const cv::Mat& a, const cv::Mat& b);
std::string findMostSimilar(const cv::Mat& embedding);
cv::Mat visualize(const cv::Mat& image, const cv::Mat& faces, const std::string mode, float fps = -1.f);

#endif //FACEDETECTION_FCA_H
