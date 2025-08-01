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

    void OnClose(wxCloseEvent& event);

private:
    wxButton* btnPlay,*btnImage;
    wxButton* btnStop;
    VideoPanel* videoPanel;
    wxStaticText* statusText;

    std::shared_ptr<OpencvHandler> ImgHandler;
    std::thread VideoThread;
    
    wxDECLARE_EVENT_TABLE();  // 이벤트 테이블 선언
};

