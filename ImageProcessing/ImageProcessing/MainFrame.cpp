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
    : wxFrame(nullptr, wxID_ANY, "OpenCV 연동", wxDefaultPosition, wxSize(800, 600)) {

    ImgHandler = std::make_shared<OpencvHandler>();

    wxBoxSizer* Mainsizer = new wxBoxSizer(wxVERTICAL);
    wxBoxSizer* btnSizer = new wxBoxSizer(wxHORIZONTAL);  // 버튼만 가로 배치
    btnImage = new wxButton(this, 1000, "이미지 재생");
    btnPlay = new wxButton(this, 1001, "영상 재생");
    btnStop = new wxButton(this, 1002, "정지");
    videoPanel = new VideoPanel(this);
    statusText = new wxStaticText(this, wxID_ANY, "상태: 대기중");

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
    std::cout << "~MainFrame() 호출됨\n";

    if (VideoThread.joinable())
        VideoThread.join();
    
    if (videoPanel)
        delete videoPanel;
    videoPanel = nullptr;
}

void MainFrame::OnImagePlay(wxCommandEvent&) {
    statusText->SetLabel("OpenCV로 이미지 불러오는 중...");
    
    wxBitmap bmp = ImgHandler->LoadImageAsBitmap(ImgHandler->IMAGE_GRAYSCALE,"../x64/Debug/sample.png", videoPanel->GetSize());

    videoPanel->SetFrame(bmp);
}   

void MainFrame::OnVideoPlay(wxCommandEvent& event)
{
    if (!ImgHandler->ExitChk)
        return;

    statusText->SetLabel("영상 불러오는 중...");
    if (VideoThread.joinable())
        VideoThread.join();

    VideoThread = std::thread([handler = ImgHandler, VideoPanel = videoPanel]()
        {
            handler->LoadPlayVideo("../x64/Debug/SampleV1.mp4", VideoPanel);
        });
        
}

void MainFrame::OnStopVideo(wxCommandEvent& event)
{
    statusText->SetLabel("정지 요청됨");
    ImgHandler->StopFlag.store(true);
}

void MainFrame::OnClose(wxCloseEvent& event)
{
    std::cout << "MainFrame 종료\n";
    event.Skip();
}