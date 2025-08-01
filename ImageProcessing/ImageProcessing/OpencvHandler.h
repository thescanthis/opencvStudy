#pragma once
#include <wx/bitmap.h>
#include <wx/image.h>
#include <atomic>
#include <string>

class OpencvHandler 
{
public:
    OpencvHandler();
    ~OpencvHandler();
    enum IMAGE_STATE
    {
        IMAGE_GRAYSCALE,
        IMAGE_COLOR
    };

    // �̹��� �ε� �� wxBitmap���� ��ȯ
    wxBitmap LoadImageAsBitmap(IMAGE_STATE Tpye, const std::string& path,const wxSize& targetSize);
    void LoadPlayVideo(const std::string& videoPath, class VideoPanel* panel);
    void ObjectResize();

    std::atomic<bool> StopFlag = false;           // ��������
    std::atomic<bool> PauseFlag = false;          // �Ͻ����� ����
    std::condition_variable cv;
    std::mutex cvMtx;
    cv::VideoCapture cap;                         // ���� �����ϵ��� ���ȭ
    bool ExitChk = true;
};

