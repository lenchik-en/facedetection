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
#include "YuNet.cpp"
#include "database.cpp"
#include "face_detection.h"

int main(int argc, char **argv) {
  cv::CommandLineParser parser(
      argc, argv,
      "{help  h           |                                   | Print this "
      "message}"
      "{database  db           | face_db.json                       | Set path "
      "to the database}"
      "{model m           | face_detection_yunet_2023mar.onnx  | Set path to "
      "the model}"
      "{v-model vm        | MobileFaceNet.onnx             | Set path to the "
      "model for vectoring}"
      "{mode             | identify                           | Mode: register "
      "or identify}"
      "{backend b         | opencv                            | Set DNN "
      "backend}"
      "{target t          | cpu                               | Set DNN target}"
      /* model params below*/
      "{conf_threshold    | 0.9                               | Set the "
      "minimum confidence for "
      "the model to identify a face. Filter out faces of conf < conf_threshold}"
      "{nms_threshold     | 0.3                               | Set the "
      "threshold to suppress "
      "overlapped boxes. Suppress boxes if IoU(box1, box2) >= nms_threshold, "
      "the one of higher "
      "score is kept.}"
      "{top_k             | 5000                              | Keep top_k "
      "bounding boxes before "
      "NMS. Set a lower value may help speed up postprocessing.}");
  if (parser.has("help")) {
    parser.printMessage();
    return 0;
  }

  Database db(parser.get<std::string>("database"));
  std::string model_path = parser.get<std::string>("model");
  std::string vmodel_path = parser.get<std::string>("v-model");
  std::string backend = parser.get<std::string>("backend");
  std::string target = parser.get<std::string>("target");
  std::string mode = parser.get<std::string>("mode");
  std::cout << "Mode:" << mode << std::endl;

  // model params
  float conf_threshold = parser.get<float>("conf_threshold");
  float nms_threshold = parser.get<float>("nms_threshold");
  int top_k = parser.get<int>("top_k");
  const int backend_id = str2backend.at(backend);
  const int target_id = str2target.at(target);

  // Instantiate YuNet
  YuNet model(model_path, cv::Size(320, 320), conf_threshold, nms_threshold,
              top_k, backend_id, target_id);

  int device_id = 0;
  auto cap = cv::VideoCapture(device_id);
  if (!cap.isOpened()) {
    std::cerr << "Error: Cannot open camera with device_id " << device_id
              << "\n";
    return -1;
  }
  int w = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_WIDTH));
  int h = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_HEIGHT));
  model.setInputSize(cv::Size(w, h));

  auto tick_meter = cv::TickMeter();
  cv::Mat frame;

  while (cv::waitKey(1) < 0) {

    bool has_frame = cap.read(frame);

    if (!has_frame) {
      std::cout << "No frames grabbed! Exiting ...\n";
      break;
    }

    // Inference
    tick_meter.start();
    cv::Mat faces = model.infer(frame);
    tick_meter.stop();

    // Draw results on the input image
    auto res_image = visualize(vmodel_path, frame, faces, mode, db,
                               (float)tick_meter.getFPS());

    // Visualize in a new window
    cv::imshow("Face detection for Avrora", res_image);

    tick_meter.reset();
  }

  return 0;
}