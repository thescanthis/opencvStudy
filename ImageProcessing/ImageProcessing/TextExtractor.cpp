#include "pch.h"
#include "TextExtractor.h"
#include <tesseract/baseapi.h>
#include <wx/log.h>

class TextExtractor::Impl {
public:
    tesseract::TessBaseAPI tess;
};

TextExtractor::TextExtractor() : m_impl(new Impl) {}
TextExtractor::~TextExtractor() { delete m_impl; }

bool TextExtractor::Initialize(const std::string& lang)
{
    return m_impl->tess.Init(nullptr, lang.c_str(), tesseract::OEM_LSTM_ONLY) == 0;
}

std::string TextExtractor::Extract(const cv::Mat& roi)
{
   // 0) 방어
    if (!m_impl) return {};
    auto& api = m_impl->tess;
    if (roi.empty() || roi.cols < 8 || roi.rows < 8) return {};

    // 1) 8-bit GRAY 보장
    cv::Mat gray;
    if (roi.channels() == 4)      cv::cvtColor(roi, gray, cv::COLOR_BGRA2GRAY);
    else if (roi.channels() == 3) cv::cvtColor(roi, gray, cv::COLOR_BGR2GRAY);
    else                          gray = roi.clone();

    // 2) 가벼운 전처리(인식 안정화)
    cv::Mat bin;
    cv::threshold(gray, bin, 0, 255, cv::THRESH_OTSU | cv::THRESH_BINARY);

    // 3) 텟서랙트 안정화 설정
    api.SetPageSegMode(tesseract::PSM_SINGLE_BLOCK);     // 필요시 SINGLE_LINE/WORD로 바꿔 테스트
    api.SetVariable("classify_enable_learning", "0");    // 적응학습 OFF → AdaptedTemplates NPE 우회
    api.SetVariable("load_system_dawg", "F");            // 사전 로딩 OFF(선택)
    api.SetVariable("load_freq_dawg",   "F");
    api.ClearAdaptiveClassifier();                       // 내부 상태 초기화

    // 4) 인식
    api.SetImage(bin.data, bin.cols, bin.rows, 1, (int)bin.step);
    if (api.Recognize(nullptr) != 0) {
        return {}; // segmentation 실패 시 빈 문자열 반환
    }

    // 5) UTF-8 결과 수집
    char* out = api.GetUTF8Text();       // UTF-8 인코딩
    if (!out) return {};
    std::string result(out);
    delete[] out;

    return result;
}

void TextExtractor::DebugPrint(const std::string& text) const
{
    wxLogMessage("TextExtractor Debug:");
    wxLogMessage("----------------------");
    wxLogMessage("%s", text.c_str());
    wxLogMessage("----------------------");
}