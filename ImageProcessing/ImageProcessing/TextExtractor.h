#pragma once
#include <opencv2/opencv.hpp>
#include <tesseract/baseapi.h>
#include <leptonica/allheaders.h>
#include <memory>
#include <string>
#include <vector>

struct TextResult {
    std::string text;
    double conf = 0.0;      // 평균 confidence (0~100)
    cv::Rect roi;           // 입력 ROI
    int err = 0;            // 0=OK, 음수=에러코드
    std::string errMsg;     // 에러 메시지
};

class TextExtractor {
public:
    struct Params {
        // 전처리
        bool   normalizeBackground = true;     // 배경 평탄화(divide) 사용
        int    normKernelFrac = 12;            // 가우시안 블러 커널 = min(W,H)/normKernelFrac (홀수로 스냅)
        double claheClip = 3.0;
        cv::Size claheGrid{ 8,8 };
        bool   useAdaptive = true;             // 적응 이진화 사용(조도 들쭉 개선)
        int    adaptiveBlock = 31;             // 홀수
        int    adaptiveC = 5;

        // 라인 제거
        bool   removeLines = true;             // 수평/수직 라인 제거
        int    kxDiv = 110;                    // 수평 커널 = W/kxDiv
        int    kyDiv = 110;                    // 수직 커널 = H/kyDiv

        // Tesseract
        std::string lang = "eng";
        std::string whitelist = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-()/:";
        tesseract::PageSegMode psmPrimary = tesseract::PSM_SPARSE_TEXT;  // 작은 조각 텍스트들
        // 실패시 재시도 모드
        std::vector<tesseract::PageSegMode> fallbackPsm = {
            tesseract::PSM_SINGLE_LINE, tesseract::PSM_SINGLE_WORD
        };

        // ROI 필터
        int minW = 6, minH = 8, minArea = 60;
        double minAspect = 0.3, maxAspect = 10.0;
        int pad = 2; // ROI 주변 패딩
    };

    struct BlueprintParams {
        // 전처리
        int  upscale = 2;                 // 1=off, 2~3 권장
        bool normalizeBackground = true;  // divide normalize
        int  normKernelFrac = 12;         // min(W,H)/N  (홀수로 스냅)

        // 라인 제거 커널
        int kxDiv = 110; // horiz kernel = W/kxDiv
        int kyDiv = 110; // vert  kernel = H/kyDiv

        // 이진화
        int  adaptBlock = 25;     // 홀수
        int  adaptC = 15;

        // ROI 필터
        int   minW = 8, minH = 10, minArea = 80;
        int   maxW = 400, maxH = 120;
        double minAspect = 0.25, maxAspect = 8.0;
        int   pad = 2;

        // Tesseract
        tesseract::PageSegMode psm = tesseract::PSM_AUTO;
        std::string whitelist_ascii =
            "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
            "abcdefghijklmnopqrstuvwxyz"
            "0123456789-()/:";
        bool useWhitelist = true; // kor+eng 같이 쓸 땐 false가 안전
    };

public:
    TextExtractor();
    ~TextExtractor();

    // 초기화(언어, 화이트리스트 포함)
    bool Initialize(const Params& p = Params{});

    // 단일 ROI를 넣으면 텍스트 한 덩어리로 추출
    std::string Extract(const cv::Mat& roi);

    // 컨투어로 문자 후보를 찾아 여러 박스 OCR (도면용 권장)
    std::vector<TextResult> ExtractBoxes(const cv::Mat& roi);

    void DebugPrint(const std::string& text) const;

    // 파라미터 런타임 변경
    void SetParams(const Params& p);
    const Params& GetParams() const;

    std::vector<TextResult> ExtractBlueprintText(const cv::Mat& roi,
        const BlueprintParams& bp = {});

private:
    // 내부 헬퍼
    cv::Mat bpPreprocessGray_(const cv::Mat& src, const BlueprintParams& bp) const;
    cv::Mat bpBuildTextMask_(const cv::Mat& eq, const BlueprintParams& bp) const;

private:
    class Impl;
    Impl* m_impl;
};
