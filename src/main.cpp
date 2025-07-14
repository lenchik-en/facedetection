#include "YuNet.h"
#include "database.h"
#include "face_detection.h"

int main(int argc, char** argv)
{
    cv::CommandLineParser parser(
        argc, argv,
        "{help  h           |                                   | Print this message}"
        "{database  db           | face_db.json                       | Set path to the database}"
        "{input i           |                                   | Set input to a certain image, "
        "omit if using camera}"
        "{model m           | face_detection_yunet_2023mar.onnx  | Set path to the model}"
        "{mode             | identify                           | Mode: register or identify}"
        "{backend b         | opencv                            | Set DNN backend}"
        "{target t          | cpu                               | Set DNN target}"
        "{save s            | false                             | Whether to save result image or "
        "not}"
        "{vis v             | false                             | Whether to visualize result "
        "image or not}"
        /* model params below*/
        "{conf_threshold    | 0.9                               | Set the minimum confidence for "
        "the model to identify a face. Filter out faces of conf < conf_threshold}"
        "{nms_threshold     | 0.3                               | Set the threshold to suppress "
        "overlapped boxes. Suppress boxes if IoU(box1, box2) >= nms_threshold, the one of higher "
        "score is kept.}"
        "{top_k             | 5000                              | Keep top_k bounding boxes before "
        "NMS. Set a lower value may help speed up postprocessing.}");
    if (parser.has("help"))
    {
        parser.printMessage();
        return 0;
    }

    Database    db(parser.get<std::string>("database"));
    std::string input_path = parser.get<std::string>("input");
    std::string model_path = parser.get<std::string>("model");
    std::string backend    = parser.get<std::string>("backend");
    std::string target     = parser.get<std::string>("target");
    std::string mode       = parser.get<std::string>("mode");
    bool        save_flag  = parser.get<bool>("save");
    bool        vis_flag   = parser.get<bool>("vis");

    // model params
    float     conf_threshold = parser.get<float>("conf_threshold");
    float     nms_threshold  = parser.get<float>("nms_threshold");
    int       top_k          = parser.get<int>("top_k");
    const int backend_id     = str2backend.at(backend);
    const int target_id      = str2target.at(target);

    // Instantiate YuNet
    YuNet model(model_path, cv::Size(320, 320), conf_threshold, nms_threshold, top_k, backend_id,
                target_id);

    // If input is an image
    if (!input_path.empty())
    {
        auto image = cv::imread(input_path);

        // Inference
        model.setInputSize(image.size());
        auto faces = model.infer(image);

        // Print faces
        std::cout << cv::format("%d faces detected:\n", faces.rows);
        for (int i = 0; i < faces.rows; ++i)
        {
            int   x1   = static_cast<int>(faces.at<float>(i, 0));
            int   y1   = static_cast<int>(faces.at<float>(i, 1));
            int   w    = static_cast<int>(faces.at<float>(i, 2));
            int   h    = static_cast<int>(faces.at<float>(i, 3));
            float conf = faces.at<float>(i, 14);
            std::cout << cv::format("%d: x1=%d, y1=%d, w=%d, h=%d, conf=%.4f\n", i, x1, y1, w, h,
                                    conf);
        }

        // Draw reults on the input image
        if (save_flag || vis_flag)
        {
            auto res_image = visualize(image, faces, mode, db);
            if (save_flag)
            {
                std::cout << "Results are saved to result.jpg\n";
                cv::imwrite("result.jpg", res_image);
            }
            if (vis_flag)
            {
                cv::namedWindow(input_path, cv::WINDOW_AUTOSIZE);
                cv::imshow(input_path, res_image);
                cv::waitKey(0);
            }
        }
    }
    else // Call default camera
    {
        int  device_id = 0;
        auto cap       = cv::VideoCapture(device_id);
        int  w         = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_WIDTH));
        int  h         = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_HEIGHT));
        model.setInputSize(cv::Size(w, h));

        auto    tick_meter = cv::TickMeter();
        cv::Mat frame;
        while (cv::waitKey(1) < 0)
        {
            bool has_frame = cap.read(frame);
            if (!has_frame)
            {
                std::cout << "No frames grabbed! Exiting ...\n";
                break;
            }

            // Inference
            tick_meter.start();
            cv::Mat faces = model.infer(frame);
            tick_meter.stop();

            // Draw results on the input image
            auto res_image = visualize(frame, faces, mode, db, (float)tick_meter.getFPS());
            // Visualize in a new window
            cv::imshow("Face detection for Avrora", res_image);

            tick_meter.reset();
        }
    }

    return 0;
}