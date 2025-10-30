#pragma once
#include <opencv2/opencv.hpp>
#include <string>
#include <vector>

// PDF 전체 → Mat 벡터
std::vector<cv::Mat> LoadPdfAllPages_Poppler(const std::wstring& wpath, int dpi = 200);

// 단일 페이지(0-based) → Mat
cv::Mat LoadPdfPage_Poppler(const std::wstring& wpath, int pageIndex, int dpi = 200);
