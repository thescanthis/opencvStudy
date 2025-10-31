#include "pch.h"
#include "MainFrame.h"
#include "PdfLoader.h"
#include "DebugAnalyzer.h"

MainFrame::MainFrame()
    : wxFrame(nullptr, wxID_ANY, "OpenCV ����", wxDefaultPosition, wxSize(800, 600)) 
{
    // �޴�
    wxMenu* fileMenu = new wxMenu;
    fileMenu->Append(wxID_OPEN, "&Open...\tCtrl+O");
    fileMenu->AppendSeparator();
    fileMenu->Append(wxID_EXIT, "E&xit");

    wxMenuBar* menuBar = new wxMenuBar;
    menuBar->Append(fileMenu, "&File");
    SetMenuBar(menuBar);

    CreateStatusBar();
    SetStatusText("Ready");

    // �̹��� �г�
    m_panel = new ImageCasting(this);

    // �������̴� ĵ������ �ڽ����� �÷��� ��ġ��
    m_Overay = std::make_shared<DragSelectionOverlay>(
        m_panel, // �� Ÿ��
        [this](const wxRect& rClient) {
            cv::Rect roi;
            if (m_panel->ClientRectToImageRect(rClient, roi)) {
                wxLogMessage("ROI(image): %d,%d %dx%d", roi.x, roi.y, roi.width, roi.height);
                // TODO: OCR/���� �м�
            }

            const cv::Mat& img = m_panel->MatOriginal();
            if (img.empty()) return;

            cv::Mat roiMat = img(roi).clone();

            static DebugAnalyzer analyzer;
            analyzer.AnalyzeAndShow(roiMat);
        }
    );

    // �̺�Ʈ ���ε�
    Bind(wxEVT_MENU, &MainFrame::OnOpen, this, wxID_OPEN);
    Bind(wxEVT_MENU, [&](wxCommandEvent&) { Close(true); }, wxID_EXIT);
}

MainFrame::~MainFrame()
{
    std::cout << "~MainFrame() ȣ���\n";

}

void MainFrame::OnOpen(wxCommandEvent&)
{
    wxFileDialog dlg(this, "���� ����", "", "",
        "PDF �� �̹��� ���� (*.pdf;*.png;*.jpg;*.jpeg;*.bmp;*.tif;*.tiff)|*.pdf;*.png;*.jpg;*.jpeg;*.bmp;*.tif;*.tiff",
        wxFD_OPEN | wxFD_FILE_MUST_EXIST);

    if (dlg.ShowModal() != wxID_OK)
        return;

    wxString path = dlg.GetPath();
    wxString ext = path.AfterLast('.').Lower();

    if (ext == "pdf")
    {
        SetStatusText("PDF �ε� ��...");

        m_pages = LoadPdfAllPages_Poppler(path.wc_str(), 200);
        if (m_pages.empty())
        {
            wxMessageBox("PDF�� �ҷ����� ���߽��ϴ�.", "����", wxOK | wxICON_ERROR);
            SetStatusText("PDF �ε� ����");
            return;
        }

        // ù ������ ǥ��
        m_panel->SetImage(m_pages[0]);
        SetStatusText(wxString::Format("PDF �ε� �Ϸ� (%d ������)", (int)m_pages.size()));
    }
    else
    {
        cv::Mat img = cv::imread(path.ToStdString(), cv::IMREAD_UNCHANGED);
        if (img.empty())
        {
            wxMessageBox("�̹����� �ҷ����� ���߽��ϴ�.", "����", wxOK | wxICON_ERROR);
            SetStatusText("�̹��� �ε� ����");
            return;
        }

        m_pages.clear();
        m_panel->SetImage(img);
        SetStatusText(wxString::Format("�̹��� �ε� �Ϸ�: %s", dlg.GetFilename()));
    }
}
