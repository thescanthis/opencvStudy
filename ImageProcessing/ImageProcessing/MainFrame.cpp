#include "pch.h"
#include "MainFrame.h"
#include "OpencvHandler.h"

wxBEGIN_EVENT_TABLE(MainFrame, wxFrame)
EVT_BUTTON(1000, MainFrame::OnImagePlay)
EVT_BUTTON(1001, MainFrame::OnVideoPlay)
EVT_BUTTON(1002, MainFrame::OnStopVideo)
EVT_CLOSE(MainFrame::OnClose)
wxEND_EVENT_TABLE()

MainFrame::MainFrame()
    : wxFrame(nullptr, wxID_ANY, "OpenCV ����", wxDefaultPosition, wxSize(800, 600)) {

    ImgHandler = std::make_shared<OpencvHandler>();

    wxBoxSizer* Mainsizer = new wxBoxSizer(wxVERTICAL);
    wxBoxSizer* btnSizer = new wxBoxSizer(wxHORIZONTAL);  // ��ư�� ���� ��ġ
    btnImage = new wxButton(this, 1000, "�̹��� ���");
    btnPlay = new wxButton(this, 1001, "���� ���");
    btnStop = new wxButton(this, 1002, "����");
    videoPanel = new VideoPanel(this);
    statusText = new wxStaticText(this, wxID_ANY, "����: �����");

    btnSizer->Add(btnImage, 0, wxALL, 5);
    btnSizer->Add(btnPlay, 0, wxALL, 5);
    btnSizer->Add(btnStop, 0, wxALL, 5);

    Mainsizer->Add(btnSizer, 0, wxALIGN_CENTER);
    Mainsizer->Add(videoPanel, 1, wxALL | wxEXPAND, 5);
    Mainsizer->Add(statusText, 0, wxALL, 5);
    SetSizer(Mainsizer);
}

MainFrame::~MainFrame()
{
    std::cout << "~MainFrame() ȣ���\n";

    if (VideoThread.joinable())
        VideoThread.join();
    
    if (videoPanel)
        delete videoPanel;
    videoPanel = nullptr;
}

void MainFrame::OnImagePlay(wxCommandEvent&) {
    statusText->SetLabel("OpenCV�� �̹��� �ҷ����� ��...");
    
    wxBitmap bmp = ImgHandler->LoadImageAsBitmap(ImgHandler->IMAGE_GRAYSCALE,"../x64/Debug/sample.png", videoPanel->GetSize());

    videoPanel->SetFrame(bmp);
}   

void MainFrame::OnVideoPlay(wxCommandEvent& event)
{
    if (!ImgHandler->ExitChk)
        return;

    statusText->SetLabel("���� �ҷ����� ��...");
    if (VideoThread.joinable())
        VideoThread.join();

    VideoThread = std::thread([handler = ImgHandler, VideoPanel = videoPanel]()
        {
            handler->LoadPlayVideo("../x64/Debug/SampleV1.mp4", VideoPanel);
        });
        
}

void MainFrame::OnStopVideo(wxCommandEvent& event)
{
    statusText->SetLabel("���� ��û��");
    ImgHandler->StopFlag.store(true);
}

void MainFrame::OnClose(wxCloseEvent& event)
{
    std::cout << "MainFrame ����\n";
    event.Skip();
}