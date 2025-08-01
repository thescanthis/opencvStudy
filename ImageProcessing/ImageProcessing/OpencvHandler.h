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
    void LoadPlayVideo(const std::string& videoPath, wxPanel* panel);

    bool VideoFlag;
    std::atomic<bool> PauseFlag;
    std::atomic<int> ResumeFrame = 0;           // ������ ���� ������ ��ȣ
};

