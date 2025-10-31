#pragma once
#include <opencv2/opencv.hpp>
#include <tesseract/baseapi.h>
#include <memory>
#include <string>

struct TextResult {
    std::string text;
    double conf = 0.0;      // ��� confidence
    cv::Rect roi;           // �Է� ROI
    int err = 0;            // 0=OK, ����=�����ڵ�
    std::string errMsg;     // ���� �޽���
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

