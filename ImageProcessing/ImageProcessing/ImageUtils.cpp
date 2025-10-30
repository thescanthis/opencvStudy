#include "pch.h"
#include "ImageUtils.h"

//�Լ��� Bind�ϴ� �̺�Ʈ ���̺�
wxBEGIN_EVENT_TABLE(ImageCasting, wxPanel)
EVT_PAINT(ImageCasting::OnPaint)
EVT_SIZE(ImageCasting::OnSize)
EVT_ERASE_BACKGROUND(ImageCasting::OnEraseBG)
wxEND_EVENT_TABLE()

ImageCasting::ImageCasting(wxWindow* parent)
    : wxPanel(parent, wxID_ANY)
{
    SetBackgroundStyle(wxBG_STYLE_PAINT);  // �������
    SetBackgroundColour(*wxWHITE);         // ���� ���
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

    // --- 4ä��(BGRA/ARGB �迭) ---
    if (src.channels() == 4)
    {
        cv::Mat rgb;    // 3ä��
        cv::Mat alpha;  // 1ä��

        // OpenCV BGRA -> RGB, �׸��� ���� �и�
        cv::cvtColor(src, rgb, cv::COLOR_BGRA2RGB);
        cv::extractChannel(src, alpha, 3);

        wxImage img(rgb.cols, rgb.rows);             // RGB ����(3*cols)
        unsigned char* dstRGB = img.GetData();

        const int srcStepRGB = (int)rgb.step[0];
        const int rowRGBBytes = rgb.cols * 3;
        const unsigned char* srcRGB = rgb.data;

        // �� ���� ���� (stride ����)
        for (int y = 0; y < rgb.rows; ++y)
            std::memcpy(dstRGB + y * rowRGBBytes, srcRGB + y * srcStepRGB, rowRGBBytes);

        // ���� ���� �Ҵ� �� �� ���� ����
        img.SetAlpha(new unsigned char[rgb.cols * rgb.rows]);
        for (int y = 0; y < alpha.rows; ++y)
            std::memcpy(img.GetAlpha() + y * alpha.cols,
                alpha.ptr<unsigned char>(y),
                alpha.cols);

        return wxBitmap(img);
    }

    // --- 3ä��(BGR) ---
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

    // --- 1ä��(Gray) ---
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

    // �� �� ������ 8UC3�� ���� ��ȯ �� ó��
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

    // ��ǰ�� �����ϸ�
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
        // �׷��Ƚ� ���ؽ�Ʈ ���� ���� �� �⺻ ��ο�
        dc.DrawBitmap(m_bitmap, 0, 0, true);
    }
}

void ImageCasting::OnSize(wxSizeEvent& e)
{
    RebuildScaled();      // ������ �ٲ� ���� ���� ������
    Refresh(false);
    e.Skip();
}


void ImageCasting::RebuildScaled()
{
    m_scaled = wxBitmap(); // ����
    m_drawPos = { 0,0 };
    if (!m_bitmap.IsOk()) return;

    const wxSize cs = GetClientSize();
    if (cs.x <= 0 || cs.y <= 0) return;

    const int bw = m_bitmap.GetWidth();
    const int bh = m_bitmap.GetHeight();

    // ���� ���� ������ (���� �ȼ��� �ݿø�)
    const double sx = (double)cs.x / bw;
    const double sy = (double)cs.y / bh;
    const double s = std::min(sx, sy);

    const int dw = std::max(1, (int)std::floor(bw * s + 0.5));
    const int dh = std::max(1, (int)std::floor(bh * s + 0.5));

    wxImage img = m_bitmap.ConvertToImage();
    img = img.Scale(dw, dh, wxIMAGE_QUALITY_HIGH); // ���⼭�� ��ǰ�� ������
    m_scaled = wxBitmap(img);

    // �߾� ��ġ (���� ��ǥ)
    m_drawPos.x = (cs.x - dw) / 2;
    m_drawPos.y = (cs.y - dh) / 2;
}

