#include "pch.h"
#include "MainFrame.h"

#ifdef _DEBUG
#define _CRTDBG_MAP_ALLOC
#include <crtdbg.h>

// ����/���� ������ new �����ε� (�� .cpp������!)
#define DBG_NEW new(_NORMAL_BLOCK, __FILE__, __LINE__)
#define new DBG_NEW
#endif

class MyApp : public wxApp {
public:
    virtual bool OnInit() {

        frame = new MainFrame();
        frame->Show();
        SetTopWindow(frame);
        // OnExit() ���� ������ ������ ��� ������

        return true;
    }

    virtual int OnExit() override {
        return wxApp::OnExit();
    }


private:
    MainFrame* frame = nullptr;
};

wxIMPLEMENT_APP(MyApp);