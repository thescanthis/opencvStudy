#pragma once
#include "ShapeExtractor.h"
#include "TextExtractor.h"
#include <opencv2/opencv.hpp>

class DebugAnalyzer {
public:
    DebugAnalyzer();

    void AnalyzeAndShow(const cv::Mat& roi);

private:
    ShapeExtractor shape;
    TextExtractor  text;
};