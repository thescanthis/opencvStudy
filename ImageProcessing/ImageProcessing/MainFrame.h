#pragma once
#include <wx/timer.h>
#include "VideoPanel.h"

class OpencvHandler;

class MainFrame : public wxFrame
{
public:
    MainFrame();
    ~MainFrame();
private:
    void OnImagePlay(wxCommandEvent& event);

    void OnVideoPlay(wxCommandEvent& event);
    void OnStopVideo(wxCommandEvent& event);
    void PauseVideo(wxCommandEvent& event);

    void OnClose(wxCloseEvent& event);

    void ThreadJoinAble();
    void ThreadExit();
    void OnSliderChaged(wxScrollEvent& event);
    void OnSliderReleased(wxScrollEvent& event);
private:
    wxButton* btnPlay,*btnImage;
    wxButton* btnStop,*Pause;
    wxStaticText* statusText;
    wxSlider* videoSlider;

    std::shared_ptr<VideoPanel> videoPanel;
    std::shared_ptr<OpencvHandler> ImgHandler;
    std::thread VideoThread;
    std::atomic<bool> isSeeking = false;
    std::thread seekingThread;

    wxDECLARE_EVENT_TABLE();  // 이벤트 테이블 선언
};

