#include "pch.h"
#include "MainFrame.h"


class MyApp : public wxApp {
public:
    virtual bool OnInit() {

        wxInitAllImageHandlers(); // wxImage�� �ʱ�ȭ
        
        frame = new MainFrame();
        frame->Bind(wxEVT_DESTROY, &MyApp::OnMainFrameDestroyed, this);
        frame->Show();
        SetTopWindow(frame);
        SetExitOnFrameDelete(true);  // �̰� ���������� �������� ���� �� delete ��
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