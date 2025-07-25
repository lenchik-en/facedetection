#include "../src/database.cpp"
#include <cstdio>
#include <gtest/gtest.h>

TEST(DatabaseTest, SaveAndFindMostSimilar) {
  std::string test_db = "test_db.json";
  std::remove(test_db.c_str());

  Database db(test_db);

  cv::Mat emb1 = (cv::Mat_<float>(1, 3) << 1.0, 0.0, 0.0);
  db.saveEmbedding("Alice", emb1);

  cv::Mat emb2 = (cv::Mat_<float>(1, 3) << 0.9, 0.1, 0.0);
  EXPECT_TRUE(db.findMostSimilar(emb2));

  cv::Mat emb3 = (cv::Mat_<float>(1, 3) << 0.0, 1.0, 0.0);
  EXPECT_FALSE(db.findMostSimilar(emb3));

  std::remove(test_db.c_str());
}

TEST(Database, SaveEmbedding) {
  Database db("test.json");
  db.saveEmbedding("test", cv::Mat::zeros(1, 128, CV_32F));
}

TEST(DatabaseTest, EmptyDatabaseFind) {
  std::string test_db = "test_db_empty.json";
  std::remove(test_db.c_str());
  Database db(test_db);

  cv::Mat emb = (cv::Mat_<float>(1, 3) << 1.0, 0.0, 0.0);
  EXPECT_FALSE(db.findMostSimilar(emb));

  std::remove(test_db.c_str());
}

TEST(DatabaseTest, SaveMultipleSameName) {
  std::string test_db = "test_db_multi.json";
  std::remove(test_db.c_str());
  Database db(test_db);

  cv::Mat emb1 = (cv::Mat_<float>(1, 3) << 1.0, 0.0, 0.0);
  cv::Mat emb2 = (cv::Mat_<float>(1, 3) << 0.0, 1.0, 0.0);
  db.saveEmbedding("Alice", emb1);
  db.saveEmbedding("Alice", emb2);

  // Проверяем, что оба embedding сохранены (можно добавить метод size() для
  // проверки) Или проверить, что поиск по emb2 даст true
  EXPECT_TRUE(db.findMostSimilar(emb2));

  std::remove(test_db.c_str());
}

TEST(DatabaseTest, SaveAndReload) {
  std::string test_db = "test_db_reload.json";
  std::remove(test_db.c_str());

  {
    Database db(test_db);
    cv::Mat emb = (cv::Mat_<float>(1, 3) << 1.0, 0.0, 0.0);
    db.saveEmbedding("Alice", emb);
  }
  {
    Database db(test_db);
    cv::Mat emb = (cv::Mat_<float>(1, 3) << 1.0, 0.0, 0.0);
    EXPECT_TRUE(db.findMostSimilar(emb));
  }
  std::remove(test_db.c_str());
}