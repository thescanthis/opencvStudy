#include "pch.h"
#include "MainFrame.h"
#include "PdfLoader.h"
#include "DebugAnalyzer.h"

MainFrame::MainFrame()
    : wxFrame(nullptr, wxID_ANY, "OpenCV 연동", wxDefaultPosition, wxSize(800, 600)) 
{
    // 메뉴
    wxMenu* fileMenu = new wxMenu;
    fileMenu->Append(wxID_OPEN, "&Open...\tCtrl+O");
    fileMenu->AppendSeparator();
    fileMenu->Append(wxID_EXIT, "E&xit");

    wxMenuBar* menuBar = new wxMenuBar;
    menuBar->Append(fileMenu, "&File");
    SetMenuBar(menuBar);

    CreateStatusBar();
    SetStatusText("Ready");

    // 이미지 패널
    m_panel = new ImageCasting(this);

    // 오버레이는 캔버스의 자식으로 올려서 겹치게
    m_Overay = std::make_shared<DragSelectionOverlay>(
        m_panel, // ← 타겟
        [this](const wxRect& rClient) {
            cv::Rect roi;
            if (m_panel->ClientRectToImageRect(rClient, roi)) {
                wxLogMessage("ROI(image): %d,%d %dx%d", roi.x, roi.y, roi.width, roi.height);
                // TODO: OCR/도형 분석
            }

            const cv::Mat& img = m_panel->MatOriginal();
            if (img.empty()) return;

            cv::Mat roiMat = img(roi).clone();

            static DebugAnalyzer analyzer;
            analyzer.AnalyzeAndShow(roiMat);
        }
    );

    // 이벤트 바인딩
    Bind(wxEVT_MENU, &MainFrame::OnOpen, this, wxID_OPEN);
    Bind(wxEVT_MENU, [&](wxCommandEvent&) { Close(true); }, wxID_EXIT);
}

MainFrame::~MainFrame()
{
    std::cout << "~MainFrame() 호출됨\n";

}

void MainFrame::OnOpen(wxCommandEvent&)
{
    wxFileDialog dlg(this, "파일 열기", "", "",
        "PDF 및 이미지 파일 (*.pdf;*.png;*.jpg;*.jpeg;*.bmp;*.tif;*.tiff)|*.pdf;*.png;*.jpg;*.jpeg;*.bmp;*.tif;*.tiff",
        wxFD_OPEN | wxFD_FILE_MUST_EXIST);

    if (dlg.ShowModal() != wxID_OK)
        return;

    wxString path = dlg.GetPath();
    wxString ext = path.AfterLast('.').Lower();

    if (ext == "pdf")
    {
        SetStatusText("PDF 로딩 중...");

        m_pages = LoadPdfAllPages_Poppler(path.wc_str(), 200);
        if (m_pages.empty())
        {
            wxMessageBox("PDF를 불러오지 못했습니다.", "오류", wxOK | wxICON_ERROR);
            SetStatusText("PDF 로드 실패");
            return;
        }

        // 첫 페이지 표시
        m_panel->SetImage(m_pages[0]);
        SetStatusText(wxString::Format("PDF 로드 완료 (%d 페이지)", (int)m_pages.size()));
    }
    else
    {
        cv::Mat img = cv::imread(path.ToStdString(), cv::IMREAD_UNCHANGED);
        if (img.empty())
        {
            wxMessageBox("이미지를 불러오지 못했습니다.", "오류", wxOK | wxICON_ERROR);
            SetStatusText("이미지 로드 실패");
            return;
        }

        m_pages.clear();
        m_panel->SetImage(img);
        SetStatusText(wxString::Format("이미지 로드 완료: %s", dlg.GetFilename()));
    }
}
