#include "pch.h"
#include "DebugAnalyzer.h"
#include <wx/log.h>

DebugAnalyzer::DebugAnalyzer() {
    text.Initialize("eng");
}

void DebugAnalyzer::AnalyzeAndShow(const cv::Mat& roi)
{
    if (roi.empty()) {
        wxLogWarning("ROI is empty!");
        return;
    }

    // 1) 도형 추출
    shape.Analyze(roi);
    shape.DebugPrint();

    // 2) 문자 추출
    std::string t = text.Extract(roi);
    text.DebugPrint(t);

    // 3) 시각화
    cv::Mat vis = roi.clone();
    for (auto& L : shape.GetLines())
        cv::line(vis, L.p1, L.p2, cv::Scalar(0, 255, 0), 2);

    for (auto& C : shape.GetCircles())
        cv::circle(vis, C.center, (int)C.radius, cv::Scalar(255, 0, 0), 2);

    for (auto& B : shape.GetContours())
        cv::rectangle(vis, B.bbox, cv::Scalar(255, 255, 0), 1);

    if (!t.empty())
        cv::putText(vis, t.substr(0, 50), { 10,25 }, cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 255), 1);

    cv::imshow("Debug ROI View", vis);
}