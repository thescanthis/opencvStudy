#include <wx/wx.h>
#include <wx/dcbuffer.h>
#include <wx/filedlg.h>
#include <wx/graphics.h>
#include <opencv2/opencv.hpp>
#include <iostream>

// ------- 이미지 표시 패널 -------
class ImageCasting : public wxPanel
{
public:
    explicit ImageCasting(wxWindow* parent);

    void SetImage(const cv::Mat& mat);
    bool HasImage() const { return m_bitmap.IsOk(); }

public:
    // 화면에 실제 그려진 비트맵의 사각형 (client 좌표)
    wxRect GetDrawnRectOnScreen() const { return wxRect(m_drawPos, m_scaled.IsOk() ? m_scaled.GetSize() : wxSize()); }

    // 화면좌표 → 원본 이미지좌표 (성공 시 true)
    bool ScreenToImagePt(const wxPoint& p, cv::Point& out) const;

    // client 사각형(화면좌표) → 원본 이미지 사각형
    bool ClientRectToImageRect(const wxRect& rClient, cv::Rect& out) const;

private:
    // cv::Mat -> wxBitmap 변환 유틸 (구현은 .cpp에)
    wxBitmap CvMatToWxBitmap(const cv::Mat& matSrc);
    void OnPaint(wxPaintEvent&);

    void OnSize(wxSizeEvent&);
    void OnEraseBG(wxEraseEvent&); // 배경지우기 무시
    void RebuildScaled();          // 스케일 캐시 재생성

private:
    wxBitmap m_bitmap;
    wxBitmap m_scaled;        // 현재 패널 크기에 맞춘 캐시
    wxPoint  m_drawPos{ 0,0 };  // 중앙 배치 좌표
    double   m_scale = 1.0;   // 

    cv::Mat  m_matOriginal;

    wxDECLARE_EVENT_TABLE();
};