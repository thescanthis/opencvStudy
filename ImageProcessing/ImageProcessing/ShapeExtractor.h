#pragma once
#include "LineExtractor.h"   // 방금 만든 클래스 include

class ShapeExtractor {
public:
    ShapeExtractor();
    ~ShapeExtractor() = default;

    void Analyze(const cv::Mat& roi);

    cv::Mat DrawLines(const cv::Mat& src) const;
    const std::vector<aind::Line>& GetLines() const { return m_lines; }

private:
    aind::LineExtractor m_lineExtractor;   // 선 검출기
    std::vector<aind::Line> m_lines;       // 결과 캐시
};


