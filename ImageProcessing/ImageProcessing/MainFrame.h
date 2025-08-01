#pragma once
#include <wx/timer.h>
class MainFrame : public wxFrame
{
public:
    MainFrame();
    ~MainFrame();
private:
    void OnButtonClicked(wxCommandEvent& event);
    void OnClose(wxCloseEvent& event);

    void OnVideoPlay(wxCommandEvent& event);
    void OnStopVideo(wxCommandEvent& event);
    void OnReStart(wxCommandEvent& event);
    void OnTimer(wxTimerEvent& event);

private:
    wxStaticText* text;
    wxStaticBitmap* imageDisplay;
    wxTimer* timer;

    cv::VideoCapture cap;
    std::thread VideoUI;

    wxPanel* videoPanel;
    std::shared_ptr<class OpencvHandler> ImgHandler;
    wxDECLARE_EVENT_TABLE();  // 이벤트 테이블 선언
};

