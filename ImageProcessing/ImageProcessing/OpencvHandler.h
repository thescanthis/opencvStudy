#pragma once
#include <opencv2/opencv.hpp>
#include <wx/bitmap.h>
#include <wx/image.h>
#include <string>

class OpenCVHandler {
public:
    // �̹��� �ε� �� wxBitmap���� ��ȯ
    static wxBitmap LoadImageAsBitmap(const std::string& path);
};

