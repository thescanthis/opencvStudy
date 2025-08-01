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
        wxMessageBox("���� ������ �� �� �����ϴ�.", "����");
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
        
        //��ü�� ����� �°� ��������
        cv::Mat resized;
        cv::resize(frame, OUT resized, cv::Size(targetSize.GetWidth(), targetSize.GetHeight()));

        wxImage wxImg(resized.cols, resized.rows, resized.data, true);
        std::memcpy(wxImg.GetData(), resized.data, resized.cols * resized.rows * resized.channels());
        wxBitmap bitmap(wxImg);
       
        wxTheApp->CallAfter([panel,bitmap]() {
            wxClientDC dc(panel);
            wxBufferedDC bufferedDC(&dc);
            bufferedDC.Clear();  // flickering ����
            bufferedDC.DrawBitmap(bitmap, 0, 0, false);
            });
    }
    
    cap.release();
}