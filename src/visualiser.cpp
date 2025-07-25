// Copyright 2025 Elena Grigoreva
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
#include "face_detection.h"

cv::Mat visualize(const std::string &vmodel, cv::Mat &image,
                  const cv::Mat &faces, const std::string &mode, Database &db,
                  float fps) {
  static cv::Scalar box_color{0, 255, 0};
  static std::vector<cv::Scalar> landmark_color{
      cv::Scalar(255, 0, 0),   // right eye
      cv::Scalar(0, 0, 255),   // left eye
      cv::Scalar(0, 255, 0),   // nose tip
      cv::Scalar(255, 0, 255), // right mouth corner
      cv::Scalar(0, 255, 255)  // left mouth corner
  };
  static cv::Scalar text_color{0, 255, 0};

  cv::Mat output_image = image.clone();

  if (fps >= 0) {
    cv::putText(output_image, cv::format("FPS: %.2f", fps), cv::Point(0, 15),
                cv::FONT_HERSHEY_SIMPLEX, 0.5, text_color, 2);
  }

  for (int i = 0; i < faces.rows; ++i) {
    // Draw bounding boxes
    int x1 = static_cast<int>(faces.at<float>(i, 0));
    int y1 = static_cast<int>(faces.at<float>(i, 1));
    int w = static_cast<int>(faces.at<float>(i, 2));
    int h = static_cast<int>(faces.at<float>(i, 3));
    cv::rectangle(output_image, cv::Rect(x1, y1, w, h), box_color, 2);

    // Confidence as text
    float conf = faces.at<float>(i, 14);
    cv::putText(output_image, cv::format("%.4f", conf), cv::Point(x1, y1 + 12),
                cv::FONT_HERSHEY_DUPLEX, 0.5, text_color);

    // Draw landmarks
    for (int j = 0; j < landmark_color.size(); ++j) {
      int x = static_cast<int>(faces.at<float>(i, 2 * j + 4)),
          y = static_cast<int>(faces.at<float>(i, 2 * j + 5));
      cv::circle(output_image, cv::Point(x, y), 2, landmark_color[j], 2);
    }

    // ROI with size: 112x112
    cv::Rect rect(x1, y1, w, h);
    rect &= cv::Rect(0, 0, output_image.cols, output_image.rows); // ограничение

    if (rect.width <= 0 || rect.height <= 0)
      continue;

    cv::Mat faceROI = output_image(rect).clone();
    cv::resize(faceROI, faceROI, cv::Size(112, 112));

    // Make blob for MobileFaceNet
    cv::Mat blob =
        cv::dnn::blobFromImage(faceROI, 1.0 / 128.0, cv::Size(112, 112),
                               cv::Scalar(127.5, 127.5, 127.5), true, false);

    // recognize from MobileFaceNet
    static cv::dnn::Net recognizer;
    static bool net_loaded = false;
    if (!net_loaded) {
      try {
        recognizer = cv::dnn::readNet(vmodel);
        net_loaded = true;
      } catch (const cv::Exception &e) {
        std::cerr << "Warning: Could not load model: " << e.what() << std::endl;
        // Для теста можно вернуть image без изменений
        return output_image;
      }
    }
    recognizer.setInput(blob);
    cv::Mat embedding = recognizer.forward();

    if (mode == "register") {
      std::string name;
      std::cout << "Введите имя: ";
      std::cin >> name;
      db.saveEmbedding(name, embedding);
    } else if (mode == "identify") {
      // bool match = db.findMostSimilar(embedding);
      if (db.findMostSimilar(embedding)) {
        // std::cout << "Access allowed\n";
        cv::putText(output_image, "Access allowed", cv::Point(x1, y1 - 5),
                    cv::FONT_HERSHEY_SIMPLEX, 0.6, text_color, 2);
      } else {
        // std::cout << "Access denied\n";
        cv::putText(output_image, "Access denied", cv::Point(x1, y1 - 5),
                    cv::FONT_HERSHEY_SIMPLEX, 0.6, text_color, 2);
      }
    }
  }
  return output_image;
}
