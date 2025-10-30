#pragma once
#include <wx/wx.h>
#include <wx/overlay.h>
#include <functional>

class DragSelectionOverlay : public wxEvtHandler {
public:
    // target: �巡�׸� ������ �г�(�̹��� �׸��� �г�)
    // onDone: �巡�� ���� �� target�� client ��ǥ�� ���� �簢�� ����
    DragSelectionOverlay(wxWindow* target,
        std::function<void(const wxRect&)> onDone);
    ~DragSelectionOverlay();

    void EnableVisual(bool on) { m_visual = on; }
    bool IsEnabled() const { return m_visual; }

private:
    // �̺�Ʈ �ڵ鷯
    void OnLeftDown(wxMouseEvent&);
    void OnLeftUp(wxMouseEvent&);
    void OnMotion(wxMouseEvent&);
    void OnTargetSize(wxSizeEvent&);
    void OnTargetPaint(wxPaintEvent&); // Ÿ�� repaint �� �������̰� ������ �� �־� ��׸� ����(�ɼ�)

    // ��ƿ
    wxRect CurrentRect() const;
    void   Draw();     // ���� ���·� �������� �׸���
    void   Clear();    // ���� �������� �����

    wxWindow* m_target = nullptr;
    std::function<void(const wxRect&)> m_onDone;

    bool     m_visual{ true };
    bool     m_dragging{ false };
    wxPoint  m_start{}, m_end{};

    wxOverlay m_overlay;
};