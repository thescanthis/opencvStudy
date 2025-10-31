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
    wxFileDialog dlg(this, "Open", "", "",
        "PDF and image files (*.pdf;*.png;*.jpg;*.jpeg;*.bmp;*.tif;*.tiff)|*.pdf;*.png;*.jpg;*.jpeg;*.bmp;*.tif;*.tiff",
        wxFD_OPEN | wxFD_FILE_MUST_EXIST);

    if (dlg.ShowModal() != wxID_OK)
        return;

    wxString path = dlg.GetPath();
    wxString ext = path.AfterLast('.').Lower();

    if (ext == "pdf")
    {
        SetStatusText("Loading PDF pages...");

        m_pages = LoadPdfAllPages_Poppler(path.wc_str(), 200);
        if (m_pages.empty())
        {
            wxMessageBox("Failed to load PDF.", "", wxOK | wxICON_ERROR);
            SetStatusText("PDF load failed");
            return;
        }

        SetStatusText(wxString::Format("PDF loaded (%d pages). Running shape extraction...", (int)m_pages.size()));
        ShowShapeExtraction(m_pages[0]);
    }
    else
    {
        cv::Mat img = cv::imread(path.ToStdString(), cv::IMREAD_UNCHANGED);
        if (img.empty())
        {
            wxMessageBox("Failed to load image.", "", wxOK | wxICON_ERROR);
            SetStatusText("Image load failed");
            return;
        }

        m_pages.clear();
        SetStatusText(wxString::Format("Image loaded: %s. Running shape extraction...", dlg.GetFilename()));
        ShowShapeExtraction(img);
    }
}

void MainFrame::ShowShapeExtraction(const cv::Mat& image)
{
    if (image.empty())
    {
        wxLogWarning("Shape extraction skipped: empty image");
        return;
    }

    ShapeExtractor extractor;
    auto shapes = extractor.Extract(image);
    cv::Mat visual = extractor.RenderResult(image, shapes);
    m_panel->SetImage(visual);

    size_t lineCount = 0;
    size_t circleCount = 0;
    size_t comboCount = 0;
    for (const auto& shape : shapes)
    {
        switch (shape.type)
        {
        case ShapeExtractor::ShapeType::Line:
            ++lineCount;
            break;
        case ShapeExtractor::ShapeType::Circle:
            ++circleCount;
            break;
        case ShapeExtractor::ShapeType::LineAndCircle:
            ++comboCount;
            break;
        }
    }

    wxLogMessage("Shape extraction result - Lines: %zu, Circles: %zu, Line+Circle: %zu",
        lineCount, circleCount, comboCount);

    SetStatusText(wxString::Format("Shapes - Lines: %zu, Circles: %zu, Line+Circle: %zu",
        lineCount, circleCount, comboCount));
}
