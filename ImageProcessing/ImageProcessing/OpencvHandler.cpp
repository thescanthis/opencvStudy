#include "pch.h"
#include "OpencvHandler.h"

wxBitmap OpenCVHandler::LoadImageAsBitmap(const std::string& path) {
    cv::Mat img = cv::imread(path, cv::IMREAD_COLOR);
    if (img.empty()) return wxBitmap();

    cv::cvtColor(img, img, cv::COLOR_BGR2RGB);

    int size = img.cols * img.rows * 3;
    unsigned char* buffer = new unsigned char[size];
    std::memcpy(buffer, img.data, size);
    wxImage wxImg(img.cols, img.rows, buffer, true);  // ← wx가 해제 책임
    return wxBitmap(wxImg);
}
