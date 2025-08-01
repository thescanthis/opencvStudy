#include "pch.h"
#include "MainFrame.h"
#include "OpencvHandler.h"

wxBEGIN_EVENT_TABLE(MainFrame, wxFrame)
EVT_CLOSE(MainFrame::OnClose)
wxEND_EVENT_TABLE()

MainFrame::MainFrame()
    : wxFrame(nullptr, wxID_ANY, "OpenCV ����", wxDefaultPosition, wxSize(800, 600)) {

    wxPanel* panel = new wxPanel(this);
 
    text = new wxStaticText(panel, wxID_ANY, "�̹��� ��ư�� ��������", wxPoint(20, 20));
    wxButton* button = new wxButton(panel, wxID_ANY, "�ҷ�����", wxPoint(20, 60));
    new wxButton(panel, 1001, "�������", wxPoint(20, 100));
    new wxButton(panel, 1002, "��������", wxPoint(20, 140));
    imageDisplay = new wxStaticBitmap(panel, wxID_ANY, wxBitmap(1, 1), wxPoint(150, 20), wxSize(600, 500));
    videoPanel = new wxPanel(panel, wxID_ANY, wxPoint(150, 20), wxSize(600, 500));
    videoPanel->SetBackgroundStyle(wxBG_STYLE_PAINT);


    timer = new wxTimer(this);

    ImgHandler = std::make_shared<OpencvHandler>();

    button->Bind(wxEVT_BUTTON, &MainFrame::OnButtonClicked, this);
    Bind(wxEVT_BUTTON, &MainFrame::OnVideoPlay, this, 1001);
    Bind(wxEVT_BUTTON, &MainFrame::OnStopVideo, this, 1002);
    Bind(wxEVT_BUTTON, &MainFrame::OnReStart, this, 1003);

}

MainFrame::~MainFrame()
{
    std::cout << "~MainFrame() ȣ���\n";

    if (timer->IsRunning())
        timer->Stop();

    delete timer;
    timer = nullptr;

    cap.release();

    if (VideoUI.joinable())
        VideoUI.join();
    
    if (videoPanel)
        delete videoPanel;
    videoPanel = nullptr;
}

void MainFrame::OnButtonClicked(wxCommandEvent&) {
    text->SetLabel("OpenCV�� �̹��� �ҷ����� ��...");

    wxBitmap bmp = ImgHandler->LoadImageAsBitmap(ImgHandler->IMAGE_GRAYSCALE,"../x64/Debug/sample.png", imageDisplay->GetSize());

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

void MainFrame::OnVideoPlay(wxCommandEvent& event)
{
    text->SetLabel("���� �ҷ����� ��...");
    if (VideoUI.joinable())
        VideoUI.join();

    VideoUI = std::thread([handler = ImgHandler,panel = videoPanel]()
        {
            handler->LoadPlayVideo("../x64/Debug/SampleV1.mp4", panel);
        });
        
}

void MainFrame::OnStopVideo(wxCommandEvent& event)
{
    ImgHandler->VideoFlag = true;
    ImgHandler->PauseFlag.store(true);
    text->SetLabel("�������� �õ���");
}

void MainFrame::OnReStart(wxCommandEvent& event)
{
    text->SetLabel("��������� �õ���");
    ImgHandler->PauseFlag.store(false);
}

void MainFrame::OnTimer(wxTimerEvent& event)
{
    cv::Mat frame;
    if (!cap.read(frame)) {
        timer->Stop();
        return;
    }
    
    cv::cvtColor(frame, frame, cv::COLOR_BGR2RGB);
    wxImage wxImg(frame.cols, frame.rows);
    std::memcpy(wxImg.GetData(), frame.data, frame.cols * frame.rows * frame.channels());
}