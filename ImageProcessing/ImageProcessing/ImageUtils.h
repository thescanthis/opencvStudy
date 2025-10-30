#include <wx/wx.h>
#include <wx/dcbuffer.h>
#include <wx/filedlg.h>
#include <wx/graphics.h>
#include <opencv2/opencv.hpp>
#include <iostream>

// ------- �̹��� ǥ�� �г� -------
class ImageCasting : public wxPanel
{
public:
    explicit ImageCasting(wxWindow* parent);

    void SetImage(const cv::Mat& mat);
    bool HasImage() const { return m_bitmap.IsOk(); }

public:
    // ȭ�鿡 ���� �׷��� ��Ʈ���� �簢�� (client ��ǥ)
    wxRect GetDrawnRectOnScreen() const { return wxRect(m_drawPos, m_scaled.IsOk() ? m_scaled.GetSize() : wxSize()); }

    // ȭ����ǥ �� ���� �̹�����ǥ (���� �� true)
    bool ScreenToImagePt(const wxPoint& p, cv::Point& out) const;

    // client �簢��(ȭ����ǥ) �� ���� �̹��� �簢��
    bool ClientRectToImageRect(const wxRect& rClient, cv::Rect& out) const;

private:
    // cv::Mat -> wxBitmap ��ȯ ��ƿ (������ .cpp��)
    wxBitmap CvMatToWxBitmap(const cv::Mat& matSrc);
    void OnPaint(wxPaintEvent&);

    void OnSize(wxSizeEvent&);
    void OnEraseBG(wxEraseEvent&); // �������� ����
    void RebuildScaled();          // ������ ĳ�� �����

private:
    wxBitmap m_bitmap;
    wxBitmap m_scaled;        // ���� �г� ũ�⿡ ���� ĳ��
    wxPoint  m_drawPos{ 0,0 };  // �߾� ��ġ ��ǥ
    double   m_scale = 1.0;   // 

    cv::Mat  m_matOriginal;

    wxDECLARE_EVENT_TABLE();
};