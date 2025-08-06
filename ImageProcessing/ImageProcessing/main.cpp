#include "pch.h"
#include "MainFrame.h"
#define _CRTDBG_MAP_ALLOC
#include <crtdbg.h>

class MyApp : public wxApp {
public:
    virtual bool OnInit() {
        _CrtSetDbgFlag(_CRTDBG_ALLOC_MEM_DF | _CRTDBG_LEAK_CHECK_DF);  // �ڵ� �� üũ
        wxInitAllImageHandlers(); // wxImage�� �ʱ�ȭ

        //_CrtSetBreakAlloc(55556);  // �� ���⼭ �ش� �� ID ����
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