#pragma once
#include "ImageUtils.h"
#include "DragSelectionOverlay.h"

class MainFrame : public wxFrame
{
public:
    MainFrame();
    ~MainFrame();
private:
    void OnOpen(wxCommandEvent&);

private:
    ImageCasting* m_panel;
    std::shared_ptr<DragSelectionOverlay> m_Overay;

    std::vector<cv::Mat> m_pages;
};

