#pragma once
#include <optional>
#include <vector>

#include <opencv2/core.hpp>

class ShapeExtractor
{
public:
    enum class ShapeType
    {
        Line,
        Circle,
        LineAndCircle
    };

    struct LineSegment
    {
        cv::Point p1;
        cv::Point p2;

        double Length() const;
        cv::Rect BoundingRect(int padding = 0, const cv::Size& limit = {}) const;
    };

    struct Circle
    {
        cv::Point center;
        int radius = 0;

        cv::Rect BoundingRect(int padding = 0, const cv::Size& limit = {}) const;
    };

    struct DetectedShape
    {
        ShapeType type;
        cv::Rect boundingBox;
        std::vector<cv::Point> contour;
        std::optional<LineSegment> line;
        std::optional<Circle> circle;
    };

    struct Params
    {
        double cannyLower = 60.0;
        double cannyUpper = 160.0;
        double minComponentAreaRatio = 0.0002; // relative to image area
        double minLineLengthRatio = 0.18;       // relative to shorter image dimension
        double maxLineGapRatio = 0.02;          // relative to shorter image dimension
        double minCircleRadiusRatio = 0.04;     // relative to shorter image dimension
        double maxCircleRadiusRatio = 0.45;     // relative to shorter image dimension
        double circleAccumulator = 30.0;
        double circleCannyHigh = 150.0;
    };

    explicit ShapeExtractor(const Params& params = Params{});

    std::vector<DetectedShape> Extract(const cv::Mat& input) const;
    cv::Mat RenderResult(const cv::Mat& input, const std::vector<DetectedShape>& shapes) const;

private:
    Params params_;

    static std::vector<cv::Point> BuildCircleContour(const Circle& circle);
    static double ContourCircularity(const std::vector<cv::Point>& contour);
};


