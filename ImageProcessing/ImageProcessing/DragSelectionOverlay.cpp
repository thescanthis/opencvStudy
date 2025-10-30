#include "pch.h"
#include "DragSelectionOverlay.h"

wxBEGIN_EVENT_TABLE(DragSelectionOverlay, wxPanel)
EVT_ERASE_BACKGROUND(DragSelectionOverlay::OnEraseBG)
EVT_PAINT(DragSelectionOverlay::OnPaint)
EVT_LEFT_DOWN(DragSelectionOverlay::OnLeftDown)
EVT_LEFT_UP(DragSelectionOverlay::OnLeftUp)
EVT_MOTION(DragSelectionOverlay::OnMotion)
EVT_SIZE(DragSelectionOverlay::OnSize)
wxEND_EVENT_TABLE()

DragSelectionOverlay::DragSelectionOverlay(wxWindow* parent,
    std::function<void(const wxRect&)> onDone)
    : wxPanel(parent, wxID_ANY, wxDefaultPosition, wxDefaultSize,
        wxBORDER_NONE | wxWANTS_CHARS)
    , m_onDone(std::move(onDone))
{
    // 반투명 그리기 위해 배경 직접 지우지 않음
    SetBackgroundStyle(wxBG_STYLE_PAINT);
    SetBackgroundColour(wxTRANSPARENT_BRUSH->GetColour());
    SetCursor(wxCursor(wxCURSOR_CROSS));
    FitToParent();
    Raise(); // 항상 맨 위
}

void DragSelectionOverlay::FitToParent() {
    if (auto* p = GetParent()) SetSize(p->GetClientSize());
}

void DragSelectionOverlay::OnEraseBG(wxEraseEvent&) { /* no-op */ }

void DragSelectionOverlay::OnSize(wxSizeEvent& e) {
    // 부모 크기에 자동 맞춤
    e.Skip();
}

wxRect DragSelectionOverlay::CurrentRect() const {
    int x = std::min(m_start.x, m_end.x);
    int y = std::min(m_start.y, m_end.y);
    int w = std::abs(m_start.x - m_end.x);
    int h = std::abs(m_start.y - m_end.y);
    return wxRect(x, y, w, h);
}

void DragSelectionOverlay::OnPaint(wxPaintEvent&) {
    wxAutoBufferedPaintDC dc(this);
    dc.SetBackground(*wxTRANSPARENT_BRUSH);
    dc.Clear();

    if (!m_visual || !m_dragging) return;
    wxRect r = CurrentRect();
    if (r.width <= 0 || r.height <= 0) return;

    // 반투명 채우기 + 테두리
    wxColour edge(255, 0, 0);
    wxBrush fill(wxColour(255, 0, 0, 40)); // 알파
    dc.SetPen(wxPen(edge, 2));
    dc.SetBrush(fill);
    dc.DrawRectangle(r);
}

void DragSelectionOverlay::OnLeftDown(wxMouseEvent& e) {
    CaptureMouse();
    m_dragging = true;
    m_start = m_end = e.GetPosition();
    Refresh(false);
}

void DragSelectionOverlay::OnMotion(wxMouseEvent& e) {
    if (!m_dragging) return;
    m_end = e.GetPosition();
    Refresh(false);
}

void DragSelectionOverlay::OnLeftUp(wxMouseEvent& e) {
    if (HasCapture()) ReleaseMouse();
    if (!m_dragging) return;
    m_dragging = false;
    m_end = e.GetPosition();
    Refresh(false);

    wxRect r = CurrentRect();
    if (r.width >= 2 && r.height >= 2 && m_onDone) m_onDone(r);
}