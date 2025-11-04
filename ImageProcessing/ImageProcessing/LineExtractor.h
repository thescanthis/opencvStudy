#pragma once

//CLAHE(Contrast Limited Adaptive Histogram Equalization)
// 1. 불필요한 노이즈 제거
// 2. 대비 향상
// 3. 조명 불균형 보정
// 4. 관심영역 강조

// Contrast : 이미지의 밝은 부분과 어두운 부분의 차이 (고콘트라스트 : 경계뚜렷, 세부 잘보임, 저콘트라스트 : 모든영역이 비슷한 밝기 흐릿한 이미지로 만듬)
// Histogram Equalization : 전체 픽셀 밝기 분포를 넓혀 대비를 강화(조명 불균형에 취약)
// Adaptive Histogram Equalization : 이미지를 작은 블록으로 나누고 블록별 대비 강화(국소 영역의 세부가 잘보이지만, 잡음이 심해 질 수있다)
// CLAHE : AHE에 대비 제한을 추가, 과도한 대비증가를 방지 결과적으로 세부를 살리고 노이즈를 줄인다.

namespace aind {

struct Line {
	cv::Point a;
	cv::Point b;
    double    len{};
    double    ang{}; // 0 or PI/2 (snapped)
};

enum class StrokePolarity { Auto, StrokeWhite, StrokeBlack };


class LineExtractor
{
public:
    struct Params {
        // CLAHE
        double  claheClipLimit = 3;       //대비
        cv::Size claheTileGrid{ 8, 8 };

        // 이진화/반전
        bool    useOtsu = true;
        double  manualThresh = 128.0;
        bool    autoInvertIfWhiteMajor = true;
        StrokePolarity polarity = StrokePolarity::StrokeWhite; // 기본: 선=흰

        // 모폴로지 커널
        int     minK = 8;                  // 커널 길이
        double  kxScale = 1.0 / 120.0;       // W * kxScale  가로선
        double  kyScale = 1.0 / 80.0;       // H * kyScale  세로선

        // 허프/길이/틈
        double  minLenFrac = 0.10;           // 짧은선 검출영역
        double  minLenAbs = 18;           // 픽셀 범위조정
        int     houghThreshold = 50;        // 민감도
        int     maxGapMin = 8;              // 단절 허용범위
        double  maxGapFrac = 1.0 / 100.0;   // 세밀한 연결 유지

        // 스냅 기준 (~10도)
        double  snapSinThresh = 0.173;      // = sin(10°)
        double  snapCosThresh = 0.173;      // = cos(10°)

        // ===== 도형(폐영역) 필터 옵션 =====
        bool   enableSolidFilter = true;
        double solidMinArea = 150.0;       // 도형으로 간주할 최소 면적
        int    solidErodePx = 2;           // 테두리 보존용 erode 픽셀
        double solidInsideRatioThresh = 0.7; // 내부 통과 비율 임계
        int    solidSampleCount = 200;        // 선 샘플링 개수
    };

public:
    LineExtractor() = default;
    explicit LineExtractor(const Params& p) : params_(p) {}

    // 복사 금지(필요 시 허용 가능)
    LineExtractor(const LineExtractor&) = delete;
    LineExtractor& operator=(const LineExtractor&) = delete;
    LineExtractor(LineExtractor&&) = default;
    LineExtractor& operator=(LineExtractor&&) = default;

    // 파라미터 접근
    const Params& params() const { return params_; }
    void setParams(const Params& p) { params_ = p; }

    // 메인 API: ROI에서 선 검출
    void Analyze(const cv::Mat& roi);

    // 결과 조회
    const std::vector<Line>& lines() const { return lines_; }

    // 디버그 이미지(선택적으로 호출)
    // 마지막 Analyze()에서 계산된 중간 산출물 반환
    const cv::Mat& debugGray()  const { return dbgGray_; }
    const cv::Mat& debugBin()   const { return dbgBin_; }
    const cv::Mat& debugHoriz() const { return dbgHoriz_; }
    const cv::Mat& debugVert()  const { return dbgVert_; }

private:
    // 내부 단계
    void makeBinaryWithWhiteStrokes(const cv::Mat& roi, cv::Mat& gray, cv::Mat& bin) const;
    void extractHV(const cv::Mat& bin, cv::Mat& horiz, cv::Mat& vert) const;
    void detectOrthogonalLines(const cv::Mat& horiz, const cv::Mat& vert, std::vector<Line>& out) const;

    //Filterring
    cv::Mat BuildSolidMask(const cv::Mat& bin, double minArea, int erodePx) const;
    void FilterSolidAreas(std::vector<Line>& lines, const cv::Mat& solidMask, double insideRatioThresh, int samples) const;

    static int clampPositive(int v, int lo = 1) { return std::max(v, lo); }

private:
    Params params_;
    std::vector<Line> lines_;

    // 디버그 캐시
    cv::Mat dbgGray_, dbgBin_, dbgHoriz_, dbgVert_;
};

}//namespace