#pragma once
#include <functional>

class DragSelectionOverlay : public wxPanel {
public:
    // onDone: �巡�� �Ϸ� �� client ��ǥ rect ����
    explicit DragSelectionOverlay(wxWindow* parent,
        std::function<void(const wxRect&)> onDone);

    // ���� �� �ڽ� �ð�ȭ on/off
    void EnableVisual(bool on) { m_visual = on; Refresh(false); }

    // �ܺο��� ��ü���� ������ �������� ȣ��
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