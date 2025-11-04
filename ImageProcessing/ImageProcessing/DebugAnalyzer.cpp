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

    // 2) 문자 추출
    //std::string t = text.Extract(roi);
    //text.DebugPrint(t);

    // 3) 시각화
    cv::Mat vis = shape.DrawLines(roi);


    //if (!t.empty())
    //    cv::putText(vis, t.substr(0, 50), { 10,25 }, cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 255), 1);

    cv::imshow("Debug ROI View", vis);
}