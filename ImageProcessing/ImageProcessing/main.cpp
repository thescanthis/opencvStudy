#include "pch.h"
#include "MainFrame.h"
#define _CRTDBG_MAP_ALLOC
#include <crtdbg.h>

class MyApp : public wxApp {
public:
    virtual bool OnInit() {
        _CrtSetDbgFlag(_CRTDBG_ALLOC_MEM_DF | _CRTDBG_LEAK_CHECK_DF);  // 자동 릭 체크
        wxInitAllImageHandlers(); // wxImage용 초기화

        //_CrtSetBreakAlloc(55556);  // ← 여기서 해당 릭 ID 지정
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