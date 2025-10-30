#include "pch.h"
#include "MainFrame.h"


class MyApp : public wxApp {
public:
    virtual bool OnInit() {
        wxInitAllImageHandlers(); // wxImage용 초기화
        
        frame = new MainFrame();
        frame->Show();
        SetTopWindow(frame);
        SetExitOnFrameDelete(true);  // 이거 설정했으면 프레임이 닫힐 때 delete 됨
        return true;
    }

    virtual int OnExit() override {
        wxImage::CleanUpHandlers(); // 정리 작업
        return wxApp::OnExit();
    }

private:
    MainFrame* frame = nullptr;
};

wxIMPLEMENT_APP(MyApp);