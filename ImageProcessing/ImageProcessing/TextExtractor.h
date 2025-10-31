#pragma once
#include <opencv2/opencv.hpp>
#include <tesseract/baseapi.h>
#include <memory>
#include <string>

struct TextResult {
    std::string text;
    double conf = 0.0;      // 평균 confidence
    cv::Rect roi;           // 입력 ROI
    int err = 0;            // 0=OK, 음수=에러코드
    std::string errMsg;     // 에러 메시지
};

class TextExtractor {
public:
    TextExtractor();
    ~TextExtractor();

    bool Initialize(const std::string& lang = "eng");
    std::string Extract(const cv::Mat& roi);
    void DebugPrint(const std::string& text) const;

private:
    class Impl;
    Impl* m_impl;
};

