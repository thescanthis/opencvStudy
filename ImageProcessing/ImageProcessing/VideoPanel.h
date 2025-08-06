#pragma once
#include <wx/wx.h>
#include <wx/dcbuffer.h>

class VideoPanel : public wxPanel,public std::enable_shared_from_this<VideoPanel>
{
public:
    VideoPanel(wxWindow* parent)
        : wxPanel(parent), m_bitmap()
    {
        SetBackgroundStyle(wxBG_STYLE_PAINT);
    }
    ~VideoPanel()
    {

    }
    void SetFrame(const wxBitmap& bmp) {
        m_bitmap = bmp;
        Refresh();  // OnPaint 호출 유도
    }
    void ClearFrame()
    {
        m_bitmap = wxNullBitmap;
        Refresh();
        Update();
    }

private:
    wxBitmap m_bitmap;

    void OnPaint(wxPaintEvent& event) {
        wxAutoBufferedPaintDC dc(this);
        dc.Clear();

        if (m_bitmap.IsOk()) {
            wxSize panelSize = GetClientSize();
            wxSize bmpSize = m_bitmap.GetSize();

            // 중앙 정렬
            int x = (panelSize.GetWidth() - bmpSize.GetWidth()) / 2;
            int y = (panelSize.GetHeight() - bmpSize.GetHeight()) / 2;

            dc.DrawBitmap(m_bitmap, x, y, false);
        }
        else {
            dc.SetBrush(*wxWHITE_BRUSH);
            dc.Clear();
        }
    }

    wxDECLARE_EVENT_TABLE();
};