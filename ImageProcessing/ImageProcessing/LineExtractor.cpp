#include "pch.h"
#include "LineExtractor.h"

namespace aind {
    using namespace cv;
    using std::vector;

    void LineExtractor::Analyze(const Mat& roi)
    {
        lines_.clear();
        dbgGray_.release(); dbgBin_.release(); dbgHoriz_.release(); dbgVert_.release();
        if (roi.empty()) return;

        // 전처리 흑,백으로 색상을 맞추고
        makeBinaryWithWhiteStrokes(roi, dbgGray_, dbgBin_);

        // 수평,수직 분리
        extractHV(dbgBin_, dbgHoriz_, dbgVert_);
        
        //선검출
        detectOrthogonalLines(dbgHoriz_, dbgVert_, lines_);

        if (params_.enableSolidFilter)
        {
            //cv::Mat solidMask = BuildSolidMask(dbgBin_,
            //    params_.solidMinArea,
            //    params_.solidErodePx);
            //
            //FilterSolidAreas(lines_,
            //    solidMask,
            //    params_.solidInsideRatioThresh,
            //    params_.solidSampleCount);
        }
    }

    void LineExtractor::makeBinaryWithWhiteStrokes(const Mat& roi, Mat& gray, Mat& bin) const
    {
        // 1) Gray
        if (roi.channels() == 3) cv::cvtColor(roi, gray, cv::COLOR_BGR2GRAY);
        else if (roi.channels() == 4) cv::cvtColor(roi, gray, cv::COLOR_BGRA2GRAY);
        else gray = roi.clone();

        // 1) 배경(저주파) 추정: 큰 커널로 블러
        cv::Mat bg;
        cv::blur(gray, bg, cv::Size(51, 51)); // ← 값 키우면 더 큰 영역 평균

        // 2) 원본 - 배경
        cv::Mat norm;
        cv::absdiff(gray, bg, norm);

        // 3) 대비 보정
        cv::normalize(norm, norm, 0, 255, cv::NORM_MINMAX);

        // 4) 배경 흰색, 선 검정으로 반전
        cv::bitwise_not(norm, norm);

        // 5) 결과 확인
        cv::imshow("Normalized drawing", norm);
    }

    void LineExtractor::extractHV(const Mat& bin, Mat& horiz, Mat& vert) const
    {
        if (bin.empty()) { horiz.release(); vert.release(); return; }

        const int W = bin.cols, H = bin.rows;
        const int kx = std::max(params_.minK, static_cast<int>(W * params_.kxScale));
        const int ky = std::max(params_.minK, static_cast<int>(H * params_.kyScale));

        // 수평(침식,팽창)
        //erode(bin, horiz, getStructuringElement(MORPH_RECT, Size(clampPositive(kx), 1)));
        //dilate(horiz, horiz, getStructuringElement(MORPH_RECT, Size(clampPositive(kx), 1)));

        cv::erode(bin, horiz, cv::getStructuringElement(cv::MORPH_RECT, { kx,1 }));
        cv::dilate(horiz, horiz, cv::getStructuringElement(cv::MORPH_RECT, { kx,1 }));

        cv::morphologyEx(horiz, horiz, cv::MORPH_CLOSE, cv::getStructuringElement(cv::MORPH_RECT, { 9,1 }));

        // 수직
        //erode(bin, vert, getStructuringElement(MORPH_RECT, Size(1, clampPositive(ky))));
        //dilate(vert, vert, getStructuringElement(MORPH_RECT, Size(1, clampPositive(ky))));

        cv::erode(bin, vert, cv::getStructuringElement(cv::MORPH_RECT, { 1,ky }));
        cv::dilate(vert, vert, cv::getStructuringElement(cv::MORPH_RECT, { 1,ky }));
    }

    void LineExtractor::detectOrthogonalLines(const Mat& horiz, const Mat& vert, std::vector<Line>& out) const
    {
        out.clear();
        if (horiz.empty() && vert.empty()) return;

        const int W = horiz.empty() ? vert.cols : horiz.cols;
        const int H = horiz.empty() ? vert.rows : horiz.rows;
        const int shortDim = std::min(W, H);

        const double minLen = std::max(params_.minLenAbs, params_.minLenFrac * shortDim);
        const int    maxGap = std::max(params_.maxGapMin, static_cast<int>(shortDim * params_.maxGapFrac));

        std::vector<Vec4i> hl, vl;
        if (!horiz.empty()) HoughLinesP(horiz, hl, 1, CV_PI / 180, params_.houghThreshold, minLen, maxGap);
        if (!vert.empty())  HoughLinesP(vert, vl, 1, CV_PI / 180, params_.houghThreshold, minLen, maxGap);

        auto pushLine = [&](const Vec4i& L) {
            Point a(L[0], L[1]), b(L[2], L[3]);
            if (norm(a - b) < minLen) return;

            double ang = std::atan2(double(b.y - a.y), double(b.x - a.x));

            // 0/90° 스냅
            if (std::abs(std::sin(ang)) < params_.snapSinThresh) {
                b.y = a.y; ang = 0.0;
            }
            else if (std::abs(std::cos(ang)) < params_.snapCosThresh) {
                b.x = a.x; ang = CV_PI / 2;
            }
            else {
                // 직교가 아니면 스킵(원하면 유지하도록 옵션화 가능)
                return;
            }

            double len = norm(a - b);     // 스냅 후 길이 재계산
            if (len < minLen) return;

            out.push_back(Line{ a, b, len, ang });
            };

        for (const auto& L : hl) pushLine(L);
        for (const auto& L : vl) pushLine(L);

        // (옵션) 여기서 같은 y/x 라인 병합 로직 추가 가능
    }

    cv::Mat LineExtractor::BuildSolidMask(const cv::Mat& bin, double minArea, int erodePx) const
    {
        std::vector<std::vector<cv::Point>> cnts;
        cv::findContours(bin, cnts, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

        cv::Mat mask = cv::Mat::zeros(bin.size(), CV_8UC1);

        for (auto& c : cnts)
        {
            double area = cv::contourArea(c);
            if (area < minArea) continue;
            cv::drawContours(mask, std::vector<std::vector<cv::Point>>{c}, -1, 255, cv::FILLED);
        }

        if (erodePx > 0)
        {
            cv::erode(mask, mask,
                cv::getStructuringElement(cv::MORPH_RECT, { erodePx, erodePx }));
        }

        return mask;
    }
    void LineExtractor::FilterSolidAreas(std::vector<Line>& lines, const cv::Mat& solidMask, double insideRatioThresh, int samples) const
    {
        auto inside = [&](const cv::Point& p) {
            if ((unsigned)p.x >= (unsigned)solidMask.cols ||
                (unsigned)p.y >= (unsigned)solidMask.rows)
                return false;
            return solidMask.at<uint8_t>(p) > 0;
            };

        std::vector<Line> kept;
        kept.reserve(lines.size());

        for (auto& L : lines)
        {
            int hit = 0;
            for (int i = 0; i < samples; i++)
            {
                double t = double(i) / (samples - 1);
                int x = cvRound(L.a.x * (1 - t) + L.b.x * t);
                int y = cvRound(L.a.y * (1 - t) + L.b.y * t);
                if (inside({ x, y })) hit++;
            }

            double ratio = double(hit) / samples;
            if (ratio < insideRatioThresh)
                kept.push_back(L);
        }

        lines.swap(kept);
    }
}