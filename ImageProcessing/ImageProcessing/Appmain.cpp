#include "pch.h"
#include "MainFrame.h"


class MyApp : public wxApp {
public:
    virtual bool OnInit() {

        wxInitAllImageHandlers(); // wxImage용 초기화
        
        frame = new MainFrame();
        frame->Bind(wxEVT_DESTROY, &MyApp::OnMainFrameDestroyed, this);
        frame->Show();
        SetTopWindow(frame);
        SetExitOnFrameDelete(true);  // 이거 설정했으면 프레임이 닫힐 때 delete 됨
        return true;
    }

    virtual int OnExit() override {
        wxImage::CleanUpHandlers(); //  
        if (frame && !frame->IsBeingDeleted()) {
            frame->Destroy();
        }
        frame = nullptr;
        return wxApp::OnExit();
    }

private:
    void OnMainFrameDestroyed(wxWindowDestroyEvent&) {
        frame = nullptr;
    }

    MainFrame* frame = nullptr;
};

wxIMPLEMENT_APP(MyApp);