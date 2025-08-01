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

    // 이미지 로드 후 wxBitmap으로 변환
    wxBitmap LoadImageAsBitmap(IMAGE_STATE Tpye, const std::string& path,const wxSize& targetSize);
    void LoadPlayVideo(const std::string& videoPath, class VideoPanel* panel);
    void ObjectResize();

    std::atomic<bool> StopFlag = false;
    bool ExitChk = true;
};

