#pragma once
#include <functional>

class DragSelectionOverlay : public wxPanel {
public:
    // onDone: 드래그 완료 시 client 좌표 rect 전달
    explicit DragSelectionOverlay(wxWindow* parent,
        std::function<void(const wxRect&)> onDone);

    // 선택 중 박스 시각화 on/off
    void EnableVisual(bool on) { m_visual = on; Refresh(false); }

    // 외부에서 전체영역 덮도록 리사이즈 호출
    void FitToParent();

private:
    void OnEraseBG(wxEraseEvent&);
    void OnPaint(wxPaintEvent&);
    void OnLeftDown(wxMouseEvent&);
    void OnLeftUp(wxMouseEvent&);
    void OnMotion(wxMouseEvent&);
    void OnSize(wxSizeEvent&);

    wxRect CurrentRect() const;

    std::function<void(const wxRect&)> m_onDone;
    bool     m_dragging{ false };
    wxPoint  m_start{}, m_end{};
    bool     m_visual{ true };

    wxDECLARE_EVENT_TABLE();
};