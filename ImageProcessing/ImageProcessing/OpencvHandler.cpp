#include "pch.h"
#include "OpencvHandler.h"
#include "VideoPanel.h"

OpencvHandler::OpencvHandler()
{

}

OpencvHandler::~OpencvHandler()
{
}

wxBitmap OpencvHandler::LoadImageAsBitmap(IMAGE_STATE Type, const std::string& path, const wxSize& targetSize) {
    cv::Mat img = cv::imread(path, Type); 
    if (img.empty()) return wxBitmap();

    cv::cvtColor(img, img, cv::COLOR_BGR2RGB);
    
    cv::Mat resized;
    cv::resize(img, resized, cv::Size(targetSize.GetWidth(), targetSize.GetHeight()));

    wxImage wxImg(resized.cols, resized.rows);
    std::memcpy(wxImg.GetData(), resized.data, resized.cols * resized.rows * resized.channels());

    return wxBitmap(wxImg);
}

void OpencvHandler::LoadPlayVideo(const std::string& videoPath, VideoPanel* panel)
{
    ExitChk = false;
    StopFlag = false;
    cap.open(videoPath);
    if (!cap.isOpened())
        return;

    cv::Mat frame;
    while (!StopFlag.load()) {

        // 재생 중인지 확인
        {
            auto& pauseFlag = this->PauseFlag;
            auto& stopFlag = this->StopFlag;
            std::unique_lock<std::mutex> lock(cvMtx);
            cv.wait(lock, [&pauseFlag,&stopFlag] { return pauseFlag.load() || stopFlag.load(); });  // 재생 요청이 있을 때까지 대기
            if (StopFlag.load()) break;
        }

        if (!cap.read(frame)) break;

        cv::cvtColor(frame, frame, cv::COLOR_BGR2RGB);
       
        cv::Mat resized;
        cv::resize(frame, resized, cv::Size(panel->GetSize().x, panel->GetSize().y));

        wxImage wxImg(resized.cols, resized.rows, resized.data, true);
        wxBitmap bmp(wxImg);

        wxTheApp->CallAfter([panel, bmp]() {
            if (panel && panel->IsShown()) {
                panel->SetFrame(bmp);
            }
            });
    }

    ExitChk = true;
    cap.release();
}

void OpencvHandler::ObjectResize()
{
    cv::Mat resized;
    cv::Size targetSize();
}
