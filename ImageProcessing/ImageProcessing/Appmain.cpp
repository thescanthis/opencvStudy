#include "pch.h"
#include "MainFrame.h"

#ifdef _DEBUG
#define _CRTDBG_MAP_ALLOC
#include <crtdbg.h>

// 파일/라인 추적용 new 오버로드 (이 .cpp에서만!)
#define DBG_NEW new(_NORMAL_BLOCK, __FILE__, __LINE__)
#define new DBG_NEW
#endif

class MyApp : public wxApp {
public:
    virtual bool OnInit() {

        frame = new MainFrame();
        frame->Show();
        SetTopWindow(frame);
        // OnExit() 같은 곳에서 강제로 찍고 싶으면

        return true;
    }

    virtual int OnExit() override {
        return wxApp::OnExit();
    }


private:
    MainFrame* frame = nullptr;
};

wxIMPLEMENT_APP(MyApp);