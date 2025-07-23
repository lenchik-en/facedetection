#ifndef FACEDETECTION_DATABASE_H
#define FACEDETECTION_DATABASE_H

#include <fstream>
#include <string>

#include "../include/json.hpp"
#include "opencv2/opencv.hpp"
using json = nlohmann::json;

class Database
{
   public:
    explicit Database(const std::string& path) : db_path(path) {}

    void saveEmbedding(const std::string& name, const cv::Mat& embedding)
    {
        json entry;
        entry["name"] = name;
        for (int i = 0; i < embedding.cols; i++)
        {
            entry["embedding"].push_back(embedding.at<float>(0, i));
        }

        std::ifstream in(db_path);
        json          db = in ? json::parse(in) : json::array();
        db.push_back(entry);
        std::ofstream out(db_path);
        out << db.dump(2);
    }

    bool findMostSimilar(const cv::Mat& embedding)
    {
        std::ifstream in(db_path);
        if (!in) return false;

        json db = json::parse(in);

        float       best_sim  = -1.0;
        //std::string best_name = "unknown";

        for (auto& entry : db)
        {
            std::vector<float> vec = entry["embedding"];
            cv::Mat            db_embedding(1, vec.size(), CV_32F, vec.data());
            float              sim = cosineSimilarity(embedding, db_embedding);
            if (sim > best_sim)
            {
                best_sim  = sim;
                //best_name = entry["name"];
            }
        }

        if (best_sim > 0.55) return true;
        return false;
    }

   private:
    std::string db_path;

    static float cosineSimilarity(const cv::Mat& a, const cv::Mat& b)
    {
        return a.dot(b) / (cv::norm(a) * cv::norm(b));
    }
};

#endif // FACEDETECTION_DATABASE_H
