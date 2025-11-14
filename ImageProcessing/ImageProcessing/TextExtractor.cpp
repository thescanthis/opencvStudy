#include "pch.h"
#include "TextExtractor.h"
#include <wx/log.h>

// ---------------- Impl ----------------
class TextExtractor::Impl {
public:
    tesseract::TessBaseAPI tess;
    bool isInit = false;
    Params params;

    void applyTessVariables() {
        if (!isInit) return;
        if (!tess.SetVariable("tessedit_char_whitelist", params.whitelist.c_str())) {
            wxLogWarning("Failed to set tessedit_char_whitelist");
        }
        if (!tess.SetVariable("tessedit_do_invert", params.autoInvertPolarity ? "1" : "0")) {
            wxLogWarning("Failed to set tessedit_do_invert");
        }
        tess.ClearAdaptiveClassifier();
    }

    static int snapOdd(int v) { return (v % 2) ? v : v + 1; }

    // 1) 배경 평탄화 + CLAHE
    cv::Mat preprocessGray(const cv::Mat& srcGray) const {
        cv::Mat gray = srcGray;
        // 배경 평탄화(divide)로 조명/밴딩 억제
        if (params.normalizeBackground) {
            int k = std::max(snapOdd(std::min(gray.cols, gray.rows) / params.normKernelFrac), 31);
            cv::Mat bg, f1, f2, norm;
            cv::GaussianBlur(gray, bg, cv::Size(k, k), 0);
            gray.convertTo(f1, CV_32F);
            bg.convertTo(f2, CV_32F);
            norm = f1 / (f2 + 1e-6f);
            cv::normalize(norm, norm, 0, 255, cv::NORM_MINMAX);
            norm.convertTo(gray, CV_8U);
        }
        // CLAHE (약하게)
        cv::Ptr<cv::CLAHE> clahe = cv::createCLAHE(params.claheClip, params.claheGrid);
        cv::Mat eq; clahe->apply(gray, eq);
        return eq;
    }

    // 2) 이진화 (문자=흰이 되도록)
    cv::Mat binarize(const cv::Mat& eq) const {
        cv::Mat bin;
        if (params.useAdaptive) {
            int blk = snapOdd(params.adaptiveBlock);
            cv::adaptiveThreshold(eq, bin, 255,
                cv::ADAPTIVE_THRESH_GAUSSIAN_C, cv::THRESH_BINARY,
                blk, params.adaptiveC);
        }
        else {
            cv::threshold(eq, bin, 0, 255, cv::THRESH_BINARY | cv::THRESH_OTSU);
        }

        if (params.autoInvertPolarity) {
            double wr = double(cv::countNonZero(bin)) / double(bin.total());
            if (wr < 0.5) cv::bitwise_not(bin, bin);
        }

        return bin;
    }

    // 3) 라인 제거(수평/수직)
    cv::Mat removeLineMask(const cv::Mat& bin) const {
        if (!params.removeLines) return cv::Mat::zeros(bin.size(), CV_8U);

        int kx = std::max(8, bin.cols / std::max(1, params.kxDiv));
        int ky = std::max(8, bin.rows / std::max(1, params.kyDiv));

        cv::Mat horiz, vert;
        cv::erode(bin, horiz, cv::getStructuringElement(cv::MORPH_RECT, { kx,1 }));
        cv::dilate(horiz, horiz, cv::getStructuringElement(cv::MORPH_RECT, { kx,1 }));

        cv::erode(bin, vert, cv::getStructuringElement(cv::MORPH_RECT, { 1,ky }));
        cv::dilate(vert, vert, cv::getStructuringElement(cv::MORPH_RECT, { 1,ky }));

        cv::Mat lineMask; cv::bitwise_or(horiz, vert, lineMask);
        return lineMask;
    }

    // 4) 컨투어로 문자 후보 박스 뽑기
    std::vector<cv::Rect> findTextBoxes(const cv::Mat& textOnly, const cv::Size& sz) const {
        std::vector<std::vector<cv::Point>> cnts;
        cv::findContours(textOnly, cnts, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
        std::vector<cv::Rect> rois; rois.reserve(cnts.size());
        for (auto& c : cnts) {
            cv::Rect r = cv::boundingRect(c);
            if (r.width < params.minW || r.height < params.minH) continue;
            if (r.area() < params.minArea) continue;
            double ar = (double)r.width / std::max(1, r.height);
            if (ar < params.minAspect || ar > params.maxAspect) continue;

            // 패딩
            r.x = std::max(0, r.x - params.pad);
            r.y = std::max(0, r.y - params.pad);
            r.width = std::min(sz.width - r.x, r.width + 2 * params.pad);
            r.height = std::min(sz.height - r.y, r.height + 2 * params.pad);
            rois.push_back(r);
        }
        return rois;
    }

    // 5) Tesseract 1회 호출
    TextResult runTess(const cv::Mat& src, tesseract::PageSegMode psm) {
        TextResult tr;
        tr.roi = { 0,0, src.cols, src.rows };

        // 0) 초기화 가드
        if (!isInit) { tr.err = -10; tr.errMsg = "Tesseract not initialized"; return tr; }

        // 1) 입력 가드: 크기/타입
        if (src.empty() || src.cols < 1 || src.rows < 1) {
            tr.err = -11; tr.errMsg = "Empty image"; return tr;
        }

        cv::Mat im8u;
        if (src.type() == CV_8U) im8u = src;
        else cv::cvtColor(src, im8u, cv::COLOR_BGR2GRAY); // 방어적 변환

        // 2) Tesseract에 전달
        try {
            tess.SetPageSegMode(psm);
            tess.SetImage(im8u.data, im8u.cols, im8u.rows, /*bytespp=*/1, /*bytespl=*/(int)im8u.step);

            // (선택) DPI 설정: 일부 버전에서 레이아웃 힌트에 도움
            tess.SetSourceResolution(300);

            // 인식
            if (tess.Recognize(nullptr) != 0) {
                tr.err = -12; tr.errMsg = "Recognize() failed"; return tr;
            }

            // 결과
            std::unique_ptr<char, void(*)(void*)> out(tess.GetUTF8Text(), [](void* p) { if (p) delete[](char*)p; });
            if (!out) { tr.err = -13; tr.errMsg = "GetUTF8Text() null"; return tr; }

            tr.text = out.get();
            tr.conf = tess.MeanTextConf(); // 0~100
            tr.err = 0;
        }
        catch (const std::exception& e) {
            tr.err = -14; tr.errMsg = std::string("Exception: ") + e.what();
        }
        catch (...) {
            tr.err = -15; tr.errMsg = "Unknown exception in runTess";
        }
        return tr;
    }
};

static int snapOdd_(int v) { return (v % 2) ? v : v + 1; }

// ---------------- API ----------------
TextExtractor::TextExtractor() : m_impl(new Impl) {}
TextExtractor::~TextExtractor() { delete m_impl; }

bool TextExtractor::Initialize(const Params& p)
{
    if (!m_impl) return false;
    m_impl->params = p;

    // tessdata 경로가 기본 위치에 없다면 명시 (예시 경로)
    const char* dp = std::getenv("TESSDATA_PREFIX");

    if (!dp) {
        wxLogError("TESSDATA_PREFIX not set. Please set environment variable.");
        return false;
    }

    // Tesseract 엔진 초기화
    if (m_impl->tess.Init(dp, p.lang.c_str(), tesseract::OEM_LSTM_ONLY) != 0) {
        wxLogError("Tesseract Init failed. datapath=%s lang=%s", dp, p.lang.c_str());
        m_impl->isInit = false;
        return false;
    }

    // 러닝/사전 기능 비활성화 (도면용)
    m_impl->tess.SetVariable("user_defined_dpi", "300");
    m_impl->tess.SetVariable("classify_enable_learning", "0");
    m_impl->tess.SetVariable("load_system_dawg", "F");
    m_impl->tess.SetVariable("load_freq_dawg", "F");

    m_impl->isInit = true;
    return true;
}

void TextExtractor::SetParams(const Params& p) {
    if (!m_impl) return;
    m_impl->params = p;
    if (m_impl->isInit) {
        m_impl->applyTessVariables();
    }
}

const TextExtractor::Params& TextExtractor::GetParams() const { return m_impl->params; }

std::vector<TextResult> TextExtractor::ExtractBlueprintText(const cv::Mat& roi, const BlueprintParams& bp)
{
    std::vector<TextResult> out;
    if (!m_impl || roi.empty()) return out;

    // 전처리
    cv::Mat gray = bpPreprocessGray_(roi, bp);
    cv::Mat textMask = bpBuildTextMask_(gray, bp); // 흰=text

    // 컨투어 탐색
    std::vector<std::vector<cv::Point>> cnts;
    cv::findContours(textMask, cnts, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

    // 후보 ROI 필터 & 패딩
    std::vector<cv::Rect> rois; rois.reserve(cnts.size());
    for (auto& c : cnts) {
        cv::Rect r = cv::boundingRect(c);
        if (r.width < bp.minW || r.height < bp.minH) continue;
        if (r.area() < bp.minArea) continue;
        if (r.width > bp.maxW || r.height > bp.maxH) continue;
        double ar = (double)r.width / std::max(1, r.height);
        if (ar < bp.minAspect || ar > bp.maxAspect) continue;

        r.x = std::max(0, r.x - bp.pad);
        r.y = std::max(0, r.y - bp.pad);
        r.width = std::min(gray.cols - r.x, r.width + 2 * bp.pad);
        r.height = std::min(gray.rows - r.y, r.height + 2 * bp.pad);
        rois.push_back(r);
    }

    // 좌표 정렬 (위→아래, 같은 줄은 좌→우)
    std::sort(rois.begin(), rois.end(), [](const cv::Rect& a, const cv::Rect& b) {
        const int yTol = 10;
        if (std::abs(a.y - b.y) < yTol) return a.x < b.x;
        return a.y < b.y;
        });

    // OCR 공통 설정
    m_impl->tess.SetPageSegMode(bp.psm);
    m_impl->tess.SetVariable("user_defined_dpi", "300");
    m_impl->tess.SetVariable("load_system_dawg", "F");
    m_impl->tess.SetVariable("load_freq_dawg", "F");
    m_impl->tess.SetVariable("classify_enable_learning", "0");
    m_impl->tess.ClearAdaptiveClassifier();
    if (bp.useWhitelist)
        m_impl->tess.SetVariable("tessedit_char_whitelist", bp.whitelist_ascii.c_str());
    else
        m_impl->tess.SetVariable("tessedit_char_whitelist", ""); // off

    // 각 ROI OCR
    out.reserve(rois.size());
    for (auto& r : rois) {
        cv::Mat sub = gray(r); // 회색 또는 textMask(r)도 시도 가능
        // Tesseract는 검정글자/흰배경 선호 → 보장
        cv::Mat bin; cv::threshold(sub, bin, 0, 255, cv::THRESH_BINARY | cv::THRESH_OTSU);

        TextResult tr; tr.roi = (bp.upscale > 1) ?
            cv::Rect(r.x / bp.upscale, r.y / bp.upscale, r.width / bp.upscale, r.height / bp.upscale) : r;

        m_impl->tess.SetImage(bin.data, bin.cols, bin.rows, 1, (int)bin.step);
        if (m_impl->tess.Recognize(nullptr) == 0) {
            std::unique_ptr<char, void(*)(void*)> outTxt(m_impl->tess.GetUTF8Text(),
                [](void* p) { if (p) delete[](char*)p; });
            tr.text = outTxt ? outTxt.get() : "";
            tr.conf = m_impl->tess.MeanTextConf();
            // 깔끔하게 정리
            tr.text.erase(std::remove(tr.text.begin(), tr.text.end(), '\n'), tr.text.end());
        }
        else {
            tr.text = ""; tr.conf = 0.0; tr.err = -12; tr.errMsg = "Recognize failed";
        }
        out.push_back(tr);
    }
    return out;
}

cv::Mat TextExtractor::bpPreprocessGray_(const cv::Mat& src, const BlueprintParams& bp) const
{
    cv::Mat img = src;
    if (bp.upscale > 1)
        cv::resize(src, img, cv::Size(), bp.upscale, bp.upscale, cv::INTER_CUBIC);

    cv::Mat gray;
    if (img.channels() == 3) cv::cvtColor(img, gray, cv::COLOR_BGR2GRAY);
    else if (img.channels() == 4) cv::cvtColor(img, gray, cv::COLOR_BGRA2GRAY);
    else gray = img.clone();

    if (bp.normalizeBackground) {
        int k = std::max(snapOdd_(std::min(gray.cols, gray.rows) / bp.normKernelFrac), 31);
        cv::Mat bg, f1, f2, norm;
        cv::GaussianBlur(gray, bg, cv::Size(k, k), 0);
        gray.convertTo(f1, CV_32F); bg.convertTo(f2, CV_32F);
        norm = f1 / (f2 + 1e-6f);
        cv::normalize(norm, norm, 0, 255, cv::NORM_MINMAX);
        norm.convertTo(gray, CV_8U);
    }
    return gray;
}

cv::Mat TextExtractor::bpBuildTextMask_(const cv::Mat& eq, const BlueprintParams& bp) const
{
    cv::Mat bin;
    cv::adaptiveThreshold(eq, bin, 255,
        cv::ADAPTIVE_THRESH_MEAN_C, cv::THRESH_BINARY_INV,
        snapOdd_(bp.adaptBlock), bp.adaptC);

    // 라인 제거
    int kx = std::max(8, bin.cols / std::max(1, bp.kxDiv));
    int ky = std::max(8, bin.rows / std::max(1, bp.kyDiv));
    cv::Mat horiz, vert;
    cv::erode(bin, horiz, cv::getStructuringElement(cv::MORPH_RECT, { kx,1 }));
    cv::dilate(horiz, horiz, cv::getStructuringElement(cv::MORPH_RECT, { kx,1 }));
    cv::erode(bin, vert, cv::getStructuringElement(cv::MORPH_RECT, { 1,ky }));
    cv::dilate(vert, vert, cv::getStructuringElement(cv::MORPH_RECT, { 1,ky }));
    cv::Mat lineMask; cv::bitwise_or(horiz, vert, lineMask);

    cv::Mat text = bin.clone();
    text.setTo(0, lineMask);

    // 세밀한 잡음 제거 + 약간 굵게
    cv::morphologyEx(text, text, cv::MORPH_OPEN, cv::getStructuringElement(cv::MORPH_RECT, { 2,2 }));
    cv::dilate(text, text, cv::getStructuringElement(cv::MORPH_RECT, { 2,2 }), cv::Point(-1, -1), 1);
    return text; // 흰=텍스트
}

std::string TextExtractor::Extract(const cv::Mat& roi)
{
    if (!m_impl || roi.empty() || roi.cols < 8 || roi.rows < 8) return {};

    // 8-bit GRAY
    cv::Mat gray;
    if (roi.channels() == 4) cv::cvtColor(roi, gray, cv::COLOR_BGRA2GRAY);
    else if (roi.channels() == 3) cv::cvtColor(roi, gray, cv::COLOR_BGR2GRAY);
    else gray = roi.clone();

    // 전처리 → 이진화
    cv::Mat eq = m_impl->preprocessGray(gray);
    cv::Mat bin = m_impl->binarize(eq);

    // 라인 제거
    if (m_impl->params.removeLines) {
        cv::Mat lineMask = m_impl->removeLineMask(bin);
        bin.setTo(0, lineMask);
        cv::morphologyEx(bin, bin, cv::MORPH_CLOSE, cv::getStructuringElement(cv::MORPH_RECT, { 3,3 }));
    }

    // 1차 PSM
    TextResult best = m_impl->runTess(bin, m_impl->params.psmPrimary);

    // 재시도(실패/저신뢰)
    if (best.text.empty() || best.conf < 50.0) {
        for (auto psm : m_impl->params.fallbackPsm) {
            TextResult t = m_impl->runTess(bin, psm);
            if (t.conf > best.conf) best = t;
            if (best.conf >= 70.0 && !best.text.empty()) break;
        }
    }
    return best.text;
}

std::vector<TextResult> TextExtractor::ExtractBoxes(const cv::Mat& roi)
{
    std::vector<TextResult> out;
    if (!m_impl || roi.empty() || roi.cols < 8 || roi.rows < 8) return out;

    // GRAY
    cv::Mat gray;
    if (roi.channels() == 4) cv::cvtColor(roi, gray, cv::COLOR_BGRA2GRAY);
    else if (roi.channels() == 3) cv::cvtColor(roi, gray, cv::COLOR_BGR2GRAY);
    else gray = roi.clone();

    // 전처리 → 이진화
    cv::Mat eq = m_impl->preprocessGray(gray);
    cv::Mat bin = m_impl->binarize(eq);

    // 라인 제거 → 텍스트만
    cv::Mat textOnly = bin.clone();
    if (m_impl->params.removeLines) {
        cv::Mat lineMask = m_impl->removeLineMask(bin);
        textOnly.setTo(0, lineMask);
        cv::morphologyEx(textOnly, textOnly, cv::MORPH_CLOSE, cv::getStructuringElement(cv::MORPH_RECT, { 3,3 }));
    }

    // 후보 박스 탐색
    std::vector<cv::Rect> rois = m_impl->findTextBoxes(textOnly, roi.size());
    out.reserve(rois.size());

    for (auto& r : rois) {
        cv::Mat patch = textOnly(r);
        TextResult best = m_impl->runTess(patch, m_impl->params.psmPrimary);
        if (best.text.empty() || best.conf < 50.0) {
            for (auto psm : m_impl->params.fallbackPsm) {
                TextResult t = m_impl->runTess(patch, psm);
                if (t.conf > best.conf) best = t;
                if (best.conf >= 70.0 && !best.text.empty()) break;
            }
        }
        best.roi = r;
        out.push_back(best);
    }
    return out;
}

void TextExtractor::DebugPrint(const std::string& text) const
{
    wxLogMessage("TextExtractor Debug:");
    wxLogMessage("----------------------");
    wxLogMessage("%s", text.c_str());
    wxLogMessage("----------------------");
}
