#include "pch.h"
#include "MainFrame.h"
#include "OpenCVHandler.h"

wxBEGIN_EVENT_TABLE(MainFrame, wxFrame)
EVT_CLOSE(MainFrame::OnClose)
wxEND_EVENT_TABLE()

MainFrame::MainFrame()
    : wxFrame(nullptr, wxID_ANY, "OpenCV ����", wxDefaultPosition, wxSize(800, 600)) {

    wxPanel* panel = new wxPanel(this);

    text = new wxStaticText(panel, wxID_ANY, "�̹��� ��ư�� ��������", wxPoint(20, 20));
    wxButton* button = new wxButton(panel, wxID_ANY, "�ҷ�����", wxPoint(20, 60));
    imageDisplay = new wxStaticBitmap(panel, wxID_ANY, wxBitmap(1, 1), wxPoint(150, 20), wxSize(600, 500));

    button->Bind(wxEVT_BUTTON, &MainFrame::OnButtonClicked, this);
    Bind(wxEVT_CLOSE_WINDOW, &MainFrame::OnClose, this);
}

MainFrame::~MainFrame()
{
    std::cout << "~MainFrame() ȣ���\n";
}

void MainFrame::OnButtonClicked(wxCommandEvent&) {
    text->SetLabel("OpenCV�� �̹��� �ҷ����� ��...");

    wxBitmap bmp = OpenCVHandler::LoadImageAsBitmap("../x64/Debug/sample.png");
    if (bmp.IsOk()) {
        imageDisplay->SetBitmap(bmp);
        text->SetLabel("�Ϸ�!");
    }
    else {
        text->SetLabel("�̹��� �ε� ����");
    }
}

void MainFrame::OnClose(wxCloseEvent& event)
{
    std::cout << "MainFrame ����\n";
    event.Skip();
}
