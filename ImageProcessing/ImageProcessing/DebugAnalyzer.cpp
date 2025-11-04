#include "pch.h"
#include "DebugAnalyzer.h"
#include <wx/log.h>

DebugAnalyzer::DebugAnalyzer() {
    TextExtractor::Params tp;
    tp.lang = "kor+eng";
    tp.whitelist.clear(); // ← 한글 포함이면 whitelist 비우는 게 안전
    tp.psmPrimary = tesseract::PSM_SPARSE_TEXT;
    tp.fallbackPsm = { tesseract::PSM_SINGLE_LINE, tesseract::PSM_SINGLE_WORD };
    text.Initialize(tp);
}

void DebugAnalyzer::AnalyzeAndShow(const cv::Mat& roi)
{
    if (roi.empty()) {
        wxLogWarning("ROI is empty!");
        return;
    }

    // 1) 도형 추출
    //shape.Analyze(roi);

    // 2) 문자 추출
    std::string t = text.Extract(roi);
    text.DebugPrint(t);

    TextExtractor::BlueprintParams bp;
    bp.upscale = 2;            // 2~3 추천
    bp.useWhitelist = false;   // kor 포함이면 false 권장
    auto results = text.ExtractBlueprintText(roi, bp);

    for (auto& r : results) {
        if (r.conf >= 60 && !r.text.empty()) {
            //r.text at r.roi
        }
    }

    // 3) 시각화
    cv::Mat vis = shape.DrawLines(roi);


    if (!t.empty())
        cv::putText(vis, t.substr(0, 50), { 10,25 }, cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 255), 1);

    cv::imshow("Debug ROI View", vis);
}