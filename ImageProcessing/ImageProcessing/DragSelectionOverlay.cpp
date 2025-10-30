#include "pch.h"
#include "DragSelectionOverlay.h"

DragSelectionOverlay::DragSelectionOverlay(wxWindow* target,
    std::function<void(const wxRect&)> onDone)
    : m_target(target), m_onDone(std::move(onDone))
{
    // Ÿ�ٿ� �̺�Ʈ ���ε�
    m_target->Bind(wxEVT_LEFT_DOWN, &DragSelectionOverlay::OnLeftDown, this);
    m_target->Bind(wxEVT_LEFT_UP, &DragSelectionOverlay::OnLeftUp, this);
    m_target->Bind(wxEVT_MOTION, &DragSelectionOverlay::OnMotion, this);
    m_target->Bind(wxEVT_SIZE, &DragSelectionOverlay::OnTargetSize, this);

    // (�ɼ�) Ÿ���� ������Ʈ�Ǹ� �������̰� ������ �� �����Ƿ� ��׸� Ʈ����
    m_target->Bind(wxEVT_PAINT, &DragSelectionOverlay::OnTargetPaint, this, wxID_ANY);
}

DragSelectionOverlay::~DragSelectionOverlay()
{
    if (m_target) {
        m_target->Unbind(wxEVT_LEFT_DOWN, &DragSelectionOverlay::OnLeftDown, this);
        m_target->Unbind(wxEVT_LEFT_UP, &DragSelectionOverlay::OnLeftUp, this);
        m_target->Unbind(wxEVT_MOTION, &DragSelectionOverlay::OnMotion, this);
        m_target->Unbind(wxEVT_SIZE, &DragSelectionOverlay::OnTargetSize, this);
        m_target->Unbind(wxEVT_PAINT, &DragSelectionOverlay::OnTargetPaint, this, wxID_ANY);
    }
}

void DragSelectionOverlay::OnLeftDown(wxMouseEvent& e)
{
    if (!m_visual) { e.Skip(); return; }
    m_dragging = true;
    if (!m_target->HasCapture()) m_target->CaptureMouse();
    m_start = m_end = e.GetPosition();
    // ù �������� �׸��� �ʾƵ� ��
}

void DragSelectionOverlay::OnMotion(wxMouseEvent& e)
{
    if (!m_dragging || !m_visual) { e.Skip(); return; }

    // ���� �������� �����
    Clear();

    // ��ǥ ������Ʈ(Ŭ����)
    wxPoint p = e.GetPosition();
    const wxSize cs = m_target->GetClientSize();
    p.x = std::clamp(p.x, 0, cs.x - 1);
    p.y = std::clamp(p.y, 0, cs.y - 1);
    m_end = p;

    // �� �������� �׸���
    Draw();
}

void DragSelectionOverlay::OnLeftUp(wxMouseEvent& e)
{
    if (!m_dragging) { e.Skip(); return; }

    // ������ �������� ����
    Clear();
    m_dragging = false;

    wxPoint p = e.GetPosition();
    const wxSize cs = m_target->GetClientSize();
    p.x = std::clamp(p.x, 0, cs.x - 1);
    p.y = std::clamp(p.y, 0, cs.y - 1);
    m_end = p;

    if (m_target->HasCapture()) m_target->ReleaseMouse();

    const wxRect r = CurrentRect();
    if (m_onDone && r.width >= 2 && r.height >= 2)
        m_onDone(r);
}

void DragSelectionOverlay::OnTargetSize(wxSizeEvent& e)
{
    // ������ �ٲ�� �������� ��ȿ
    Clear();
    m_overlay.Reset();
    e.Skip();
}

void DragSelectionOverlay::OnTargetPaint(wxPaintEvent& e)
{
    // Ÿ���� ������Ʈ�Ǹ� �������� ������ ����� �� ������,
    // �巡�� ���̸� �ٽ� �׷���
    e.Skip();
    if (m_dragging) {
        // Paint ���Ŀ� �׸����� PostCall
        m_target->CallAfter([this] { Draw(); });
    }
}

wxRect DragSelectionOverlay::CurrentRect() const
{
    int x = std::min(m_start.x, m_end.x);
    int y = std::min(m_start.y, m_end.y);
    int w = std::abs(m_start.x - m_end.x);
    int h = std::abs(m_start.y - m_end.y);
    return wxRect(x, y, w, h);
}

void DragSelectionOverlay::Draw()
{
    wxClientDC dc(m_target);
    wxDCOverlay odc(m_overlay, &dc);
    // ���⼭�� Clear() ���� ����(�̹� OnMotion���� ���� �������� clear)
    dc.SetPen(wxPen(wxColour(255, 0, 0), 2));
    dc.SetBrush(*wxTRANSPARENT_BRUSH);
    const wxRect r = CurrentRect();
    if (r.width > 0 && r.height > 0)
        dc.DrawRectangle(r);
}

void DragSelectionOverlay::Clear()
{
    wxClientDC dc(m_target);
    wxDCOverlay odc(m_overlay, &dc);
    odc.Clear();        // ���� �������� ����
}
