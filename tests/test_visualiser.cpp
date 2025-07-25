#include "../src/database.cpp"
#include "../src/face_detection.h"
#include <cstdio>
#include <gtest/gtest.h>
#include <opencv2/opencv.hpp>

TEST(VisualiserTest, HandlesEmptyFaces) {
  std::string test_db = "test_db_vis.json";
  std::remove(test_db.c_str());
  Database db(test_db);

  // Пустое изображение 224x224
  cv::Mat image = cv::Mat::zeros(224, 224, CV_8UC3);
  // Пустая матрица faces
  cv::Mat faces;
  // Путь к несуществующей модели (т.к. не будет инференса)
  std::string vmodel = "models/MobileFaceNet.onnx";
  std::string mode = "identify";

  // Не должно падать
  cv::Mat result = visualize(vmodel, image, faces, mode, db, 0.0f);

  EXPECT_EQ(result.rows, image.rows);
  EXPECT_EQ(result.cols, image.cols);
  EXPECT_EQ(result.type(), image.type());

  std::remove(test_db.c_str());
}

TEST(VisualiserTest, HandlesSingleFace) {
  std::string test_db = "test_db_vis2.json";
  std::remove(test_db.c_str());
  Database db(test_db);

  cv::Mat image = cv::Mat::zeros(224, 224, CV_8UC3);
  // Один "фейковый" face: x, y, w, h (остальные поля не используются)
  cv::Mat faces = (cv::Mat_<float>(1, 15) << 50, 50, 100, 100, 60, 60, 70, 70,
                   80, 80, 90, 90, 100, 100, 0.99);
  std::string vmodel = "models/MobileFaceNet.onnx";
  std::string mode = "identify";

  // Не должно падать
  cv::Mat result = visualize(vmodel, image, faces, mode, db, 0.0f);

  EXPECT_EQ(result.rows, image.rows);
  EXPECT_EQ(result.cols, image.cols);
  EXPECT_EQ(result.type(), image.type());

  std::remove(test_db.c_str());
}

TEST(VisualiserTest, HandlesMultipleFaces) {
  std::string test_db = "test_db_vis3.json";
  std::remove(test_db.c_str());
  Database db(test_db);

  cv::Mat image = cv::Mat::zeros(224, 224, CV_8UC3);
  // Два "фейковых" лица
  cv::Mat faces = (cv::Mat_<float>(2, 15) << 10, 10, 50, 50, 12, 12, 14, 14, 16,
                   16, 18, 18, 20, 20, 0.95, 100, 100, 60, 60, 110, 110, 120,
                   120, 130, 130, 140, 140, 150, 150, 0.98);
  std::string vmodel = "models/MobileFaceNet.onnx";
  std::string mode = "identify";

  cv::Mat result = visualize(vmodel, image, faces, mode, db, 0.0f);

  EXPECT_EQ(result.rows, image.rows);
  EXPECT_EQ(result.cols, image.cols);
  EXPECT_EQ(result.type(), image.type());

  std::remove(test_db.c_str());
}
