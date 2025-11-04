#include "pch.h"
// ShapeExtractor.cpp
#include "ShapeExtractor.h"
#include <wx/log.h>

ShapeExtractor::ShapeExtractor()
{
}

void ShapeExtractor::Analyze(const cv::Mat& roi)
{
    m_lines.clear();
    if (roi.empty()) return;

    // 1) LineExtractor 호출
    m_lineExtractor.Analyze(roi);
    m_lines = m_lineExtractor.lines();

    // 2) 디버그용 출력
    std::cout << "[ShapeExtractor] Found " << m_lines.size() << " lines\n";

    DrawLines(roi);
}

cv::Mat ShapeExtractor::DrawLines(const cv::Mat& src) const
{
    if (src.empty()) return {};

    cv::Mat vis = src.clone();
    for (const auto& L : m_lines)
    {
        cv::line(vis, L.a, L.b, cv::Scalar(0, 255, 0), 2, cv::LINE_AA);
    }
    return vis;
}
