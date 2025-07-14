#include "FCA.h"

cv::Mat visualize(const cv::Mat& image, const cv::Mat& faces, const std::string mode, float fps)
{
    static cv::Scalar box_color{0, 255, 0};
    static std::vector<cv::Scalar> landmark_color{
            cv::Scalar(255,   0,   0), // right eye
            cv::Scalar(  0,   0, 255), // left eye
            cv::Scalar(  0, 255,   0), // nose tip
            cv::Scalar(255,   0, 255), // right mouth corner
            cv::Scalar(  0, 255, 255)  // left mouth corner
    };
    static cv::Scalar text_color{0, 255, 0};

    cv::Mat output_image = image.clone();

    if (fps >= 0)
    {
        cv::putText(output_image, cv::format("FPS: %.2f", fps), cv::Point(0, 15), cv::FONT_HERSHEY_SIMPLEX, 0.5, text_color, 2);
    }

    for (int i = 0; i < faces.rows; ++i)
    {
        // Draw bounding boxes
        int x1 = static_cast<int>(faces.at<float>(i, 0));
        int y1 = static_cast<int>(faces.at<float>(i, 1));
        int w = static_cast<int>(faces.at<float>(i, 2));
        int h = static_cast<int>(faces.at<float>(i, 3));
        cv::rectangle(output_image, cv::Rect(x1, y1, w, h), box_color, 2);

        // Confidence as text
        float conf = faces.at<float>(i, 14);
        cv::putText(output_image, cv::format("%.4f", conf), cv::Point(x1, y1+12), cv::FONT_HERSHEY_DUPLEX, 0.5, text_color);

        // Draw landmarks
        for (int j = 0; j < landmark_color.size(); ++j)
        {
            int x = static_cast<int>(faces.at<float>(i, 2*j+4)), y = static_cast<int>(faces.at<float>(i, 2*j+5));
            cv::circle(output_image, cv::Point(x, y), 2, landmark_color[j], 2);
        }

        // ROI with size: 112x112
        cv::Mat faceROI = output_image(cv::Rect(x1, y1, w, h)).clone();
        cv::resize(faceROI, faceROI, cv::Size(112, 112));

        // Make blob for MobileFaceNet
        cv::Mat blob = cv::dnn::blobFromImage(faceROI, 1.0 / 128.0, cv::Size(112, 112),
                                              cv::Scalar(127.5, 127.5, 127.5), true, false);

        //recognize from MobileFaceNet
        static cv::dnn::Net recognizer = cv::dnn::readNet("/Users/elenagrigoreva/CLionProjects/facedetection/models/MobileFaceNet.onnx");
        recognizer.setInput(blob);
        cv::Mat embedding = recognizer.forward();

        if (mode == "register") {
            std::string name;
            std::cout << "Введите имя: ";
            std::cin >> name;
            saveEmbedding(name, embedding);
        } else if (mode == "identify") {
            std::string match = findMostSimilar(embedding);
            cv::putText(output_image, match, cv::Point(x1, y1 - 5), cv::FONT_HERSHEY_SIMPLEX, 0.6, text_color, 2);
        }

    }
    return output_image;
}


void saveEmbedding (const std::string& name, const cv::Mat& embedding) {
    json entry;
    entry["name"] = name;
    for (int i = 0; i < embedding.cols; i++) {
        entry["embedding"].push_back(embedding.at<float>(0, i));
    }

    std::ifstream in("face_db.json");
    json db = in ? json::parse(in) : json::array();
    db.push_back(entry);
    std::ofstream out("face_db.json");
    out << db.dump(2);
}

// Cosine similarity between two embeddings
float cosineSimilarity(const cv::Mat& a, const cv::Mat& b) {
    return a.dot(b) / (cv::norm(a) * cv::norm(b));
}

//TODO: переделать под булевую функцию
std::string findMostSimilar(const cv::Mat& embedding) {
    //TODO: не корректная работа с базой данный
    std::ifstream in("face_db.json");
    if (!in) return "no database";

    json db = json::parse(in);


    float best_sim = -1.0;
    std::string best_name = "unknown";

    for (auto& entry : db) {
        std::vector<float> vec = entry["embedding"];
        cv::Mat db_embedding(1, vec.size(), CV_32F, vec.data());
        float sim = cosineSimilarity(embedding, db_embedding);
        if (sim > best_sim) {
            best_sim = sim;
            best_name = entry["name"];
        }
    }

    if (best_sim > 0.55) return best_name;
    return "unknown";
}
