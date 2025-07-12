#include <iostream>
#include <opencv2/opencv.hpp>
#include "libfacedetection/src/facedetectcnn.h"

int main() {
    cv::VideoCapture cap(0);
    if (!cap.isOpened()) {
        std::cerr << "Камера не открылась" << std::endl;
        return -1;
    }

    std::cout << "Камера успешно открыта" << std::endl;

    cv::Mat frame;
    while (true) {
        cap >> frame;
        if (frame.empty()) {
            std::cerr << "Пустой кадр!" << std::endl;
            continue;
        }

        cv::imshow("Camera", frame);
        if (cv::waitKey(30) == 'q') break;
    }

    return 0;
}
