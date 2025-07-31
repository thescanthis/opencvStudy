#pragma once
#include <opencv2/opencv.hpp>
#include <wx/bitmap.h>
#include <wx/image.h>
#include <string>

class OpenCVHandler {
public:
    // 이미지 로드 후 wxBitmap으로 변환
    static wxBitmap LoadImageAsBitmap(const std::string& path);
};

