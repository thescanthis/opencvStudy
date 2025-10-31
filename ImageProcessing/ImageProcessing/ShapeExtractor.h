#pragma once
#include <opencv2/opencv.hpp>
#include <vector>

struct DetectedLine {
    cv::Point p1, p2;
    double length;
    double angle;
};

struct DetectedCircle {
    cv::Point center;
    float radius;
};

struct DetectedContour {
    cv::Rect bbox;
    double area;
};

struct DetectedRect { 
    cv::Rect r; 
    double area; 
};

class ShapeExtractor {
public:
    ShapeExtractor() = default;

    void Analyze(const cv::Mat& roi);
    void DebugPrint() const;

    const std::vector<DetectedLine>& GetLines() const { return m_lines; }
    const std::vector<DetectedCircle>& GetCircles() const { return m_circles; }
    const std::vector<DetectedContour>& GetContours() const { return m_contours; }
    const std::vector<DetectedRect>& GetRects() const { return m_rects; }


    cv::Mat BuildMask(const cv::Size& sz, int lineThick = 3, int circleThick = 3, bool fillRects = true) const;

private:
    std::vector<DetectedLine>   m_lines;
    std::vector<DetectedCircle> m_circles;
    std::vector<DetectedContour> m_contours;
    std::vector<DetectedRect>   m_rects;
};

