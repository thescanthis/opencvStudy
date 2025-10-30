#pragma once
#include <wx/wx.h>
#include <wx/overlay.h>
#include <functional>

class DragSelectionOverlay : public wxEvtHandler {
public:
    // target: 드래그를 적용할 패널(이미지 그리는 패널)
    // onDone: 드래그 종료 시 target의 client 좌표로 선택 사각형 전달
    DragSelectionOverlay(wxWindow* target,
        std::function<void(const wxRect&)> onDone);
    ~DragSelectionOverlay();

    void EnableVisual(bool on) { m_visual = on; }
    bool IsEnabled() const { return m_visual; }

private:
    // 이벤트 핸들러
    void OnLeftDown(wxMouseEvent&);
    void OnLeftUp(wxMouseEvent&);
    void OnMotion(wxMouseEvent&);
    void OnTargetSize(wxSizeEvent&);
    void OnTargetPaint(wxPaintEvent&); // 타겟 repaint 시 오버레이가 지워질 수 있어 재그림 보완(옵션)

    // 유틸
    wxRect CurrentRect() const;
    void   Draw();     // 현재 상태로 오버레이 그리기
    void   Clear();    // 기존 오버레이 지우기

    wxWindow* m_target = nullptr;
    std::function<void(const wxRect&)> m_onDone;

    bool     m_visual{ true };
    bool     m_dragging{ false };
    wxPoint  m_start{}, m_end{};

    wxOverlay m_overlay;
};