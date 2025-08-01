#include "pch.h"
#include "OpencvHandler.h"

OpencvHandler::OpencvHandler()
{
    VideoFlag = true;
    PauseFlag.store(true);
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

void OpencvHandler::LoadPlayVideo(const std::string& videoPath, wxPanel* panel)
{
    VideoFlag = false;
    PauseFlag.store(false);
    cv::VideoCapture cap(videoPath);
    if (!cap.isOpened()) {
        wxMessageBox("영상 파일을 열 수 없습니다.", "오류");
        return;
    }
    
    cv::Mat frame;
    wxSize targetSize = panel->GetSize();
    while (!VideoFlag && cap.read(frame)) {
        if (PauseFlag) {
            ResumeFrame = static_cast<int>(cap.get(cv::CAP_PROP_POS_FRAMES));
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
            continue;
        }

        // OpenCV -> RGB
        cv::cvtColor(frame, frame, cv::COLOR_BGR2RGB);
        
        //객체의 사이즈에 맞게 리사이즈
        cv::Mat resized;
        cv::resize(frame, OUT resized, cv::Size(targetSize.GetWidth(), targetSize.GetHeight()));

        wxImage wxImg(resized.cols, resized.rows, resized.data, true);
        std::memcpy(wxImg.GetData(), resized.data, resized.cols * resized.rows * resized.channels());
        wxBitmap bitmap(wxImg);
       
        wxTheApp->CallAfter([panel,bitmap]() {
            wxClientDC dc(panel);
            wxBufferedDC bufferedDC(&dc);
            bufferedDC.Clear();  // flickering 방지
            bufferedDC.DrawBitmap(bitmap, 0, 0, false);
            });
    }
    
    cap.release();
}