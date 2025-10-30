#include "pch.h"
#include "ImageUtils.h"

//함수를 Bind하는 이벤트 테이블
wxBEGIN_EVENT_TABLE(ImageCasting, wxPanel)
EVT_PAINT(ImageCasting::OnPaint)
EVT_SIZE(ImageCasting::OnSize)
EVT_ERASE_BACKGROUND(ImageCasting::OnEraseBG)
wxEND_EVENT_TABLE()

ImageCasting::ImageCasting(wxWindow* parent)
    : wxPanel(parent, wxID_ANY)
{
    SetBackgroundStyle(wxBG_STYLE_PAINT);  // 더블버퍼
    SetBackgroundColour(*wxWHITE);         // 배경색 명시
}

void ImageCasting::OnEraseBG(wxEraseEvent&) {}

void ImageCasting::SetImage(const cv::Mat& mat)
{
    m_matOriginal = mat.clone();
    m_bitmap = CvMatToWxBitmap(mat);
    RebuildScaled();
    Refresh();
}

bool ImageCasting::ScreenToImagePt(const wxPoint& p, cv::Point& out) const
{
    if (!m_scaled.IsOk() || !m_bitmap.IsOk() || m_scale <= 0.0) return false;
    wxRect dst(m_drawPos, m_scaled.GetSize());
    if (!dst.Contains(p)) return false;
    double ix = (p.x - m_drawPos.x) / m_scale;
    double iy = (p.y - m_drawPos.y) / m_scale;
    ix = std::clamp(ix, 0.0, (double)m_bitmap.GetWidth() - 1);
    iy = std::clamp(iy, 0.0, (double)m_bitmap.GetHeight() - 1);
    out = cv::Point((int)std::floor(ix + 0.5), (int)std::floor(iy + 0.5));
    return true;
}

bool ImageCasting::ClientRectToImageRect(const wxRect& rClient, cv::Rect& out) const
{
    cv::Point a, b;
    if (!ScreenToImagePt(rClient.GetTopLeft(), a)) return false;
    if (!ScreenToImagePt(rClient.GetBottomRight(), b)) return false;
    int x = std::min(a.x, b.x), y = std::min(a.y, b.y);
    int w = std::abs(a.x - b.x), h = std::abs(a.y - b.y);
    if (w < 2 || h < 2) return false;
    int W = m_matOriginal.cols, H = m_matOriginal.rows;
    x = std::clamp(x, 0, W - 1); y = std::clamp(y, 0, H - 1);
    if (x + w > W) w = W - x; if (y + h > H) h = H - y;
    out = cv::Rect(x, y, w, h);
    return true;
}

// ----- cv::Mat -> wxBitmap -----
wxBitmap ImageCasting::CvMatToWxBitmap(const cv::Mat& src)
{
    if (src.empty()) return wxBitmap();

    // --- 4채널(BGRA/ARGB 계열) ---
    if (src.channels() == 4)
    {
        cv::Mat rgb;    // 3채널
        cv::Mat alpha;  // 1채널

        // OpenCV BGRA -> RGB, 그리고 알파 분리
        cv::cvtColor(src, rgb, cv::COLOR_BGRA2RGB);
        cv::extractChannel(src, alpha, 3);

        wxImage img(rgb.cols, rgb.rows);             // RGB 버퍼(3*cols)
        unsigned char* dstRGB = img.GetData();

        const int srcStepRGB = (int)rgb.step[0];
        const int rowRGBBytes = rgb.cols * 3;
        const unsigned char* srcRGB = rgb.data;

        // 행 단위 복사 (stride 안전)
        for (int y = 0; y < rgb.rows; ++y)
            std::memcpy(dstRGB + y * rowRGBBytes, srcRGB + y * srcStepRGB, rowRGBBytes);

        // 알파 버퍼 할당 후 행 단위 복사
        img.SetAlpha(new unsigned char[rgb.cols * rgb.rows]);
        for (int y = 0; y < alpha.rows; ++y)
            std::memcpy(img.GetAlpha() + y * alpha.cols,
                alpha.ptr<unsigned char>(y),
                alpha.cols);

        return wxBitmap(img);
    }

    // --- 3채널(BGR) ---
    if (src.channels() == 3)
    {
        cv::Mat rgb;
        cv::cvtColor(src, rgb, cv::COLOR_BGR2RGB);

        wxImage img(rgb.cols, rgb.rows);
        unsigned char* dst = img.GetData();

        const int srcStep = (int)rgb.step[0];
        const int rowBytes = rgb.cols * 3;
        const unsigned char* p = rgb.data;

        for (int y = 0; y < rgb.rows; ++y)
            std::memcpy(dst + y * rowBytes, p + y * srcStep, rowBytes);

        return wxBitmap(img);
    }

    // --- 1채널(Gray) ---
    if (src.channels() == 1)
    {
        cv::Mat rgb;
        cv::cvtColor(src, rgb, cv::COLOR_GRAY2RGB);

        wxImage img(rgb.cols, rgb.rows);
        unsigned char* dst = img.GetData();

        const int srcStep = (int)rgb.step[0];
        const int rowBytes = rgb.cols * 3;
        const unsigned char* p = rgb.data;

        for (int y = 0; y < rgb.rows; ++y)
            std::memcpy(dst + y * rowBytes, p + y * srcStep, rowBytes);

        return wxBitmap(img);
    }

    // 그 외 포맷은 8UC3로 강제 변환 후 처리
    cv::Mat tmp8u3, rgb;
    src.convertTo(tmp8u3, CV_8UC3);
    cv::cvtColor(tmp8u3, rgb, cv::COLOR_BGR2RGB);

    wxImage img(rgb.cols, rgb.rows);
    unsigned char* dst = img.GetData();
    const int srcStep = (int)rgb.step[0];
    const int rowBytes = rgb.cols * 3;
    const unsigned char* p = rgb.data;

    for (int y = 0; y < rgb.rows; ++y)
        std::memcpy(dst + y * rowBytes, p + y * srcStep, rowBytes);

    return wxBitmap(img);
}

void ImageCasting::OnPaint(wxPaintEvent&)
{
    wxAutoBufferedPaintDC dc(this);
    dc.Clear();

    if (!m_bitmap.IsOk()) return;

    // 고품질 스케일링
    if (wxGraphicsContext* gc = wxGraphicsContext::Create(dc))
    {
        const wxSize cs = GetClientSize();
        const int bw = m_bitmap.GetWidth();
        const int bh = m_bitmap.GetHeight();

        const double sx = static_cast<double>(cs.x) / bw;
        const double sy = static_cast<double>(cs.y) / bh;
        const double s = std::min(sx, sy);

        const double drawW = bw * s;
        const double drawH = bh * s;
        const double x = (cs.x - drawW) * 0.5;
        const double y = (cs.y - drawH) * 0.5;

        gc->DrawBitmap(m_bitmap, x, y, drawW, drawH);
        delete gc;
    }
    else
    {
        // 그래픽스 컨텍스트 생성 실패 시 기본 드로우
        dc.DrawBitmap(m_bitmap, 0, 0, true);
    }
}

void ImageCasting::OnSize(wxSizeEvent& e)
{
    RebuildScaled();      // 사이즈 바뀔 때만 새로 스케일
    Refresh(false);
    e.Skip();
}


void ImageCasting::RebuildScaled()
{
    m_scaled = wxBitmap(); // 비우기
    m_drawPos = { 0,0 };
    if (!m_bitmap.IsOk()) return;

    const wxSize cs = GetClientSize();
    if (cs.x <= 0 || cs.y <= 0) return;

    const int bw = m_bitmap.GetWidth();
    const int bh = m_bitmap.GetHeight();

    // 비율 유지 스케일 (정수 픽셀로 반올림)
    const double sx = (double)cs.x / bw;
    const double sy = (double)cs.y / bh;
    const double s = std::min(sx, sy);

    const int dw = std::max(1, (int)std::floor(bw * s + 0.5));
    const int dh = std::max(1, (int)std::floor(bh * s + 0.5));

    wxImage img = m_bitmap.ConvertToImage();
    img = img.Scale(dw, dh, wxIMAGE_QUALITY_HIGH); // 여기서만 고품질 스케일
    m_scaled = wxBitmap(img);

    // 중앙 배치 (정수 좌표)
    m_drawPos.x = (cs.x - dw) / 2;
    m_drawPos.y = (cs.y - dh) / 2;
}

