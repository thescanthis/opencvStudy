#include "pch.h"
#include "MainFrame.h"
#include "OpencvHandler.h"

wxBEGIN_EVENT_TABLE(MainFrame, wxFrame)
EVT_BUTTON(1000, MainFrame::OnImagePlay)
EVT_BUTTON(1001, MainFrame::OnVideoPlay)
EVT_BUTTON(1002, MainFrame::OnStopVideo)
EVT_BUTTON(1003, MainFrame::PauseVideo)
EVT_CLOSE(MainFrame::OnClose)
wxEND_EVENT_TABLE()

MainFrame::MainFrame()
    : wxFrame(nullptr, wxID_ANY, "OpenCV ����", wxDefaultPosition, wxSize(800, 600)) {

    ImgHandler = std::make_shared<OpencvHandler>();

    wxBoxSizer* Mainsizer = new wxBoxSizer(wxVERTICAL);
    wxBoxSizer* btnSizer = new wxBoxSizer(wxHORIZONTAL);  // ��ư�� ���� ��ġ
    btnImage = new wxButton(this, 1000, "�̹��� ���");
    btnPlay = new wxButton(this, 1001, "���� ���");
    btnStop = new wxButton(this, 1002, "�ߴ�");
    Pause = new wxButton(this, 1003, "����");
    videoPanel = std::make_shared<VideoPanel>(this);
    statusText = new wxStaticText(this, wxID_ANY, "����: �����");
    videoSlider = new wxSlider(this, wxID_ANY, 0, 0,1000);

    Layout();
    btnSizer->Add(btnImage, 0, wxALL, 5);
    btnSizer->Add(btnPlay, 0, wxALL, 5);
    btnSizer->Add(btnStop, 0, wxALL, 5);
    btnSizer->Add(Pause, 0, wxALL, 5);

    Mainsizer->Add(btnSizer, 0, wxALIGN_CENTER);
    Mainsizer->Add(videoPanel.get(), 1, wxALL | wxEXPAND, 5);
    Mainsizer->Add(videoSlider, 0, wxEXPAND | wxLEFT | wxRIGHT | wxBOTTOM, 5);
    Mainsizer->Add(statusText, 0, wxALL, 5);

    videoSlider->Bind(wxEVT_SCROLL_THUMBTRACK, &MainFrame::OnSliderChaged, this);     // �巡�� ��
    videoSlider->Bind(wxEVT_SCROLL_CHANGED, &MainFrame::OnSliderReleased, this);
    SetSizer(Mainsizer);
}

MainFrame::~MainFrame()
{
    std::cout << "~MainFrame() ȣ���\n";

    ThreadExit();
    wxTheApp->Yield();       // ���� �̺�Ʈ ����
    wxTheApp->ProcessIdle(); // �̺�Ʈ ť ����
}

void MainFrame::OnImagePlay(wxCommandEvent&) {
    statusText->SetLabel("OpenCV�� �̹��� �ҷ����� ��...");
    
    wxBitmap bmp = ImgHandler->LoadImageAsBitmap(ImgHandler->IMAGE_GRAYSCALE,"../x64/Debug/sample.jpg", videoPanel->GetSize());
    videoPanel->SetFrame(bmp);
}   

void MainFrame::OnVideoPlay(wxCommandEvent& event)
{
    if (!ImgHandler->ExitChk)
        return;

    videoSlider->SetValue(0);
    statusText->SetLabel("���� �ҷ����� ��...");
    ThreadJoinAble();

    ImgHandler->PauseFlag.store(true);
    VideoThread = std::thread([&handler = ImgHandler, &VideoPanel = videoPanel,&Slider = videoSlider](){
            handler->LoadPlayVideo("../x64/Debug/LoadCar1.mp4", VideoPanel.get());
        });
       
    ImgHandler->cv.notify_one();
}

void MainFrame::OnStopVideo(wxCommandEvent& event)
{
    statusText->SetLabel("���� ��û��");
    ThreadExit();

    wxTheApp->CallAfter([this]() {
        videoPanel->ClearFrame();
        });
}

void MainFrame::PauseVideo(wxCommandEvent& event)
{
    bool newState = !ImgHandler->PauseFlag.load();
    ImgHandler->PauseFlag.store(newState);

    wxButton* btn = dynamic_cast<wxButton*>(event.GetEventObject());
    if (btn) {
        btn->SetLabel(newState ? "�Ͻ�����" : "���");
    }

    if (newState)
        ImgHandler->cv.notify_one();
}

void MainFrame::OnClose(wxCloseEvent& event)
{
    std::cout << "MainFrame ����\n";
    event.Skip();
}

void MainFrame::ThreadJoinAble()
{
    if (VideoThread.joinable())
        VideoThread.join();  // �ݵ�� ���� ��ٷ��� ��
}

void MainFrame::ThreadExit()
{
    ImgHandler->StopFlag.store(true);
    ImgHandler->cv.notify_one();
    ThreadJoinAble();
}

void MainFrame::OnSliderChaged(wxScrollEvent& event)
{
    if (!ImgHandler || !ImgHandler->cap.isOpened())
         return;

    if (isSeeking.load()) return;

    ImgHandler->PauseFlag.store(false);
    isSeeking.store(true);

    seekingThread = std::thread([&isSeeking= isSeeking, videoSlider= videoSlider,&ImgHandler= ImgHandler,&videoPanel= videoPanel]() {
        
        while (isSeeking)
        {
            cv::Mat frame;
            int frameIdx = videoSlider->GetValue();
            {
                std::lock_guard<std::mutex> _lock(ImgHandler->capMtx);
                ImgHandler->cap.set(cv::CAP_PROP_POS_FRAMES, frameIdx);
                ImgHandler->cap >> frame;
            }

            if (!frame.empty())
            {
                wxImage wxImg(frame.cols, frame.rows, frame.data, true);
                wxBitmap bmp(wxImg);

                wxTheApp->CallAfter([panel = videoPanel, bmp]() {
                    panel->SetFrame(bmp);
                    });
            }

            std::this_thread::sleep_for(std::chrono::milliseconds(10));  // 20fps ����
        }
        }
    );
}

void MainFrame::OnSliderReleased(wxScrollEvent& event)
{
    if (!ImgHandler || !ImgHandler->cap.isOpened()) return;

    isSeeking.store(false);

    if (seekingThread.joinable())
        seekingThread.join();  // ��ŷ ������ ����

    ImgHandler->PauseFlag.store(true);  // ��� �簳
    ImgHandler->cv.notify_one();         // ��� ������ ����
}
