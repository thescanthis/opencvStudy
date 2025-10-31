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
   // 0) ���
    if (!m_impl) return {};
    auto& api = m_impl->tess;
    if (roi.empty() || roi.cols < 8 || roi.rows < 8) return {};

    // 1) 8-bit GRAY ����
    cv::Mat gray;
    if (roi.channels() == 4)      cv::cvtColor(roi, gray, cv::COLOR_BGRA2GRAY);
    else if (roi.channels() == 3) cv::cvtColor(roi, gray, cv::COLOR_BGR2GRAY);
    else                          gray = roi.clone();

    // 2) ������ ��ó��(�ν� ����ȭ)
    cv::Mat bin;
    cv::threshold(gray, bin, 0, 255, cv::THRESH_OTSU | cv::THRESH_BINARY);

    // 3) �ݼ���Ʈ ����ȭ ����
    api.SetPageSegMode(tesseract::PSM_SINGLE_BLOCK);     // �ʿ�� SINGLE_LINE/WORD�� �ٲ� �׽�Ʈ
    api.SetVariable("classify_enable_learning", "0");    // �����н� OFF �� AdaptedTemplates NPE ��ȸ
    api.SetVariable("load_system_dawg", "F");            // ���� �ε� OFF(����)
    api.SetVariable("load_freq_dawg",   "F");
    api.ClearAdaptiveClassifier();                       // ���� ���� �ʱ�ȭ

    // 4) �ν�
    api.SetImage(bin.data, bin.cols, bin.rows, 1, (int)bin.step);
    if (api.Recognize(nullptr) != 0) {
        return {}; // segmentation ���� �� �� ���ڿ� ��ȯ
    }

    // 5) UTF-8 ��� ����
    char* out = api.GetUTF8Text();       // UTF-8 ���ڵ�
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