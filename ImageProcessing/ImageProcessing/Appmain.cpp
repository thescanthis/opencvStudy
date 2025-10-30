#include "pch.h"
#include "MainFrame.h"


class MyApp : public wxApp {
public:
    virtual bool OnInit() {
        wxInitAllImageHandlers(); // wxImage�� �ʱ�ȭ
        
        frame = new MainFrame();
        frame->Show();
        SetTopWindow(frame);
        SetExitOnFrameDelete(true);  // �̰� ���������� �������� ���� �� delete ��
        return true;
    }

    virtual int OnExit() override {
        wxImage::CleanUpHandlers(); // ���� �۾�
        return wxApp::OnExit();
    }

private:
    MainFrame* frame = nullptr;
};

wxIMPLEMENT_APP(MyApp);