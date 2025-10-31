#include "pch.h"
#include "ShapeExtractor.h"

#include <algorithm>
#include <cmath>
#include <string>
#include <utility>

using Shape = ShapeExtractor::DetectedShape;

namespace
{
    cv::Point ClampPoint(const cv::Point& pt, const cv::Size& size)
    {
        cv::Point clamped = pt;
        if (size.width > 0) {
            clamped.x = std::clamp(clamped.x, 0, size.width - 1);
        }
        if (size.height > 0) {
            clamped.y = std::clamp(clamped.y, 0, size.height - 1);
        }
        return clamped;
    }
}

double ShapeExtractor::LineSegment::Length() const
{
    const double dx = static_cast<double>(p1.x - p2.x);
    const double dy = static_cast<double>(p1.y - p2.y);
    return std::sqrt(dx * dx + dy * dy);
}

cv::Rect ShapeExtractor::LineSegment::BoundingRect(int padding, const cv::Size& limit) const
{
    std::vector<cv::Point> points{ ClampPoint(p1, limit), ClampPoint(p2, limit) };
    cv::Rect rect = cv::boundingRect(points);
    if (padding > 0) {
        rect.x -= padding;
        rect.y -= padding;
        rect.width += padding * 2;
        rect.height += padding * 2;
    }
    if (limit.width > 0) {
        rect.x = std::max(0, std::min(rect.x, limit.width));
        int maxWidth = std::max(0, limit.width - rect.x);
        rect.width = std::clamp(rect.width, 0, maxWidth);
    }
    if (limit.height > 0) {
        rect.y = std::max(0, std::min(rect.y, limit.height));
        int maxHeight = std::max(0, limit.height - rect.y);
        rect.height = std::clamp(rect.height, 0, maxHeight);
    }
    return rect;
}

cv::Rect ShapeExtractor::Circle::BoundingRect(int padding, const cv::Size& limit) const
{
    cv::Rect rect(center.x - radius, center.y - radius, radius * 2, radius * 2);
    if (padding > 0) {
        rect.x -= padding;
        rect.y -= padding;
        rect.width += padding * 2;
        rect.height += padding * 2;
    }
    if (limit.width > 0) {
        int right = std::min(limit.width, rect.x + rect.width);
        rect.x = std::clamp(rect.x, 0, limit.width);
        rect.width = std::max(0, right - rect.x);
    }
    if (limit.height > 0) {
        int bottom = std::min(limit.height, rect.y + rect.height);
        rect.y = std::clamp(rect.y, 0, limit.height);
        rect.height = std::max(0, bottom - rect.y);
    }
    return rect;
}

ShapeExtractor::ShapeExtractor(const Params& params)
    : params_(params)
{
}

std::vector<Shape> ShapeExtractor::Extract(const cv::Mat& input) const
{
    if (input.empty()) {
        return {};
    }

    cv::Mat gray;
    if (input.channels() == 3) {
        cv::cvtColor(input, gray, cv::COLOR_BGR2GRAY);
    }
    else if (input.channels() == 4) {
        cv::cvtColor(input, gray, cv::COLOR_BGRA2GRAY);
    }
    else {
        gray = input.clone();
    }

    cv::Mat blurred;
    cv::GaussianBlur(gray, blurred, cv::Size(5, 5), 0);

    cv::Mat edges;
    cv::Canny(blurred, edges, params_.cannyLower, params_.cannyUpper);

    cv::Mat binary;
    cv::adaptiveThreshold(blurred, binary, 255, cv::ADAPTIVE_THRESH_GAUSSIAN_C, cv::THRESH_BINARY_INV, 35, 5);

    cv::Mat labels, stats, centroids;
    int numLabels = cv::connectedComponentsWithStats(binary, labels, stats, centroids);

    cv::Mat filtered = cv::Mat::zeros(binary.size(), CV_8UC1);
    const double imageArea = static_cast<double>(binary.total());
    const int minArea = std::max(20, static_cast<int>(imageArea * params_.minComponentAreaRatio));
    const int shorterDim = std::min(binary.cols, binary.rows);
    const int minDimThreshold = std::max(8, static_cast<int>(shorterDim * 0.04));

    for (int label = 1; label < numLabels; ++label) {
        int area = stats.at<int>(label, cv::CC_STAT_AREA);
        int width = stats.at<int>(label, cv::CC_STAT_WIDTH);
        int height = stats.at<int>(label, cv::CC_STAT_HEIGHT);

        if (area >= minArea || width >= minDimThreshold || height >= minDimThreshold) {
            cv::Mat mask = (labels == label);
            mask.convertTo(mask, CV_8UC1, 255);
            filtered |= mask;
        }
    }

    if (cv::countNonZero(filtered) > 0) {
        cv::bitwise_and(edges, filtered, edges);
    }

    cv::Mat morphKernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3, 3));
    cv::dilate(edges, edges, morphKernel);
    cv::erode(edges, edges, morphKernel);

    int minLineLength = std::max(20, static_cast<int>(shorterDim * params_.minLineLengthRatio));
    int maxLineGap = std::max(5, static_cast<int>(shorterDim * params_.maxLineGapRatio));
    int houghThreshold = std::max(25, static_cast<int>(shorterDim * 0.08));

    std::vector<cv::Vec4i> rawLines;
    cv::HoughLinesP(edges, rawLines, 1.0, CV_PI / 180.0, houghThreshold, minLineLength, maxLineGap);

    std::vector<LineSegment> lines;
    lines.reserve(rawLines.size());
    for (const cv::Vec4i& l : rawLines) {
        LineSegment seg{ {l[0], l[1]}, {l[2], l[3]} };
        if (seg.Length() >= minLineLength) {
            lines.push_back(seg);
        }
    }

    double minRadiusF = std::max(8.0, shorterDim * params_.minCircleRadiusRatio);
    double maxRadiusF = std::max(minRadiusF + 5.0, shorterDim * params_.maxCircleRadiusRatio);
    double minDist = std::max(25.0, shorterDim * 0.25);

    std::vector<cv::Vec3f> rawCircles;
    cv::HoughCircles(blurred, rawCircles, cv::HOUGH_GRADIENT, 1.0, minDist, params_.circleCannyHigh, params_.circleAccumulator,
        static_cast<int>(minRadiusF), static_cast<int>(maxRadiusF));

    std::vector<Circle> circles;
    circles.reserve(rawCircles.size());
    for (const cv::Vec3f& c : rawCircles) {
        Circle circle{ cv::Point(cvRound(c[0]), cvRound(c[1])), cvRound(c[2]) };
        if (circle.radius <= 0) {
            continue;
        }
        cv::Rect roi = circle.BoundingRect(0, gray.size());
        if (roi.width <= 0 || roi.height <= 0) {
            continue;
        }
        if (cv::countNonZero(filtered) > 0) {
            cv::Mat mask = filtered(roi);
            double filled = cv::countNonZero(mask);
            double expected = CV_PI * circle.radius * circle.radius;
            if (expected > 0 && filled / expected < 0.35) {
                continue;
            }
        }
        circles.push_back(circle);
    }

    std::vector<Shape> results;
    results.reserve(lines.size() + circles.size());

    std::vector<bool> lineUsed(lines.size(), false);
    std::vector<bool> circleUsed(circles.size(), false);

    for (size_t ci = 0; ci < circles.size(); ++ci) {
        const Circle& circle = circles[ci];
        cv::Rect circleRect = circle.BoundingRect(4, gray.size());

        for (size_t li = 0; li < lines.size(); ++li) {
            if (lineUsed[li]) {
                continue;
            }
            const LineSegment& line = lines[li];
            cv::Rect lineRect = line.BoundingRect(4, gray.size());
            if ((circleRect & lineRect).area() <= 0) {
                continue;
            }

            double radius = static_cast<double>(circle.radius);
            double tol = std::max(6.0, radius * 0.25);
            double dist1 = cv::norm(line.p1 - circle.center);
            double dist2 = cv::norm(line.p2 - circle.center);
            bool touches = (std::abs(dist1 - radius) <= tol) || (std::abs(dist2 - radius) <= tol);

            if (!touches) {
                continue;
            }

            lineUsed[li] = true;
            circleUsed[ci] = true;

            Shape combined;
            combined.type = ShapeType::LineAndCircle;
            combined.line = line;
            combined.circle = circle;
            combined.boundingBox = circleRect | lineRect;
            combined.contour = BuildCircleContour(circle);
            combined.contour.push_back(line.p1);
            combined.contour.push_back(line.p2);
            results.push_back(std::move(combined));
            break;
        }
    }

    for (size_t i = 0; i < lines.size(); ++i) {
        if (lineUsed[i]) {
            continue;
        }
        Shape shape;
        shape.type = ShapeType::Line;
        shape.line = lines[i];
        shape.boundingBox = lines[i].BoundingRect(4, gray.size());
        shape.contour = { lines[i].p1, lines[i].p2 };
        results.push_back(std::move(shape));
    }

    for (size_t i = 0; i < circles.size(); ++i) {
        if (circleUsed[i]) {
            continue;
        }
        Shape shape;
        shape.type = ShapeType::Circle;
        shape.circle = circles[i];
        shape.boundingBox = circles[i].BoundingRect(2, gray.size());
        shape.contour = BuildCircleContour(circles[i]);
        results.push_back(std::move(shape));
    }

    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(edges, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

    for (const auto& contour : contours) {
        if (contour.size() < 5) {
            continue;
        }
        double area = std::abs(cv::contourArea(contour));
        if (area <= 0) {
            continue;
        }
        double circularity = ContourCircularity(contour);
        cv::Rect bbox = cv::boundingRect(contour);
        if (bbox.width <= 0 || bbox.height <= 0) {
            continue;
        }

        bool overlapsExisting = false;
        for (const auto& existing : results) {
            if ((existing.boundingBox & bbox).area() > 0) {
                overlapsExisting = true;
                break;
            }
        }
        if (overlapsExisting) {
            continue;
        }

        if (circularity > 0.7) {
            Circle circle;
            circle.center = cv::Point(bbox.x + bbox.width / 2, bbox.y + bbox.height / 2);
            circle.radius = static_cast<int>(std::round((bbox.width + bbox.height) / 4.0));
            Shape shape;
            shape.type = ShapeType::Circle;
            shape.circle = circle;
            shape.boundingBox = circle.BoundingRect(0, gray.size());
            shape.contour = contour;
            results.push_back(std::move(shape));
        }
    }

    return results;
}

cv::Mat ShapeExtractor::RenderResult(const cv::Mat& input, const std::vector<Shape>& shapes) const
{
    if (input.empty()) {
        return {};
    }

    cv::Mat canvas;
    switch (input.channels()) {
    case 1:
        cv::cvtColor(input, canvas, cv::COLOR_GRAY2BGR);
        break;
    case 3:
        canvas = input.clone();
        break;
    case 4:
        cv::cvtColor(input, canvas, cv::COLOR_BGRA2BGR);
        break;
    default:
        input.convertTo(canvas, CV_8UC3);
        break;
    }

    for (const auto& shape : shapes) {
        cv::Scalar color;
        std::string label;
        switch (shape.type) {
        case ShapeType::Line:
            color = cv::Scalar(60, 220, 60);
            label = "Line";
            break;
        case ShapeType::Circle:
            color = cv::Scalar(60, 60, 220);
            label = "Circle";
            break;
        case ShapeType::LineAndCircle:
            color = cv::Scalar(220, 70, 200);
            label = "Line+Circle";
            break;
        }

        if (shape.circle.has_value()) {
            cv::circle(canvas, shape.circle->center, shape.circle->radius, color, 2, cv::LINE_AA);
        }
        if (shape.line.has_value()) {
            cv::line(canvas, shape.line->p1, shape.line->p2, color, 2, cv::LINE_AA);
        }
        if (shape.boundingBox.width > 0 && shape.boundingBox.height > 0) {
            cv::rectangle(canvas, shape.boundingBox, color, 1, cv::LINE_AA);
        }

        cv::Point textPt(shape.boundingBox.x, shape.boundingBox.y - 5);
        if (textPt.y < 10) {
            textPt.y = shape.boundingBox.y + shape.boundingBox.height + 15;
        }
        textPt = ClampPoint(textPt, canvas.size());
        cv::putText(canvas, label, textPt, cv::FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv::LINE_AA);
    }

    return canvas;
}

std::vector<cv::Point> ShapeExtractor::BuildCircleContour(const Circle& circle)
{
    std::vector<cv::Point> contour;
    const int segments = 36;
    contour.reserve(static_cast<size_t>(segments));
    for (int i = 0; i < segments; ++i) {
        double angle = 2.0 * CV_PI * static_cast<double>(i) / static_cast<double>(segments);
        int x = static_cast<int>(std::round(circle.center.x + circle.radius * std::cos(angle)));
        int y = static_cast<int>(std::round(circle.center.y + circle.radius * std::sin(angle)));
        contour.emplace_back(x, y);
    }
    return contour;
}

double ShapeExtractor::ContourCircularity(const std::vector<cv::Point>& contour)
{
    double perimeter = cv::arcLength(contour, true);
    if (perimeter <= 0.0) {
        return 0.0;
    }
    double area = std::abs(cv::contourArea(contour));
    return (4.0 * CV_PI * area) / (perimeter * perimeter);
}