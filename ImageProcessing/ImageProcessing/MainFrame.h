#pragma once
class MainFrame : public wxFrame
{
public:
    MainFrame();
    ~MainFrame();
private:
    wxStaticText* text;
    wxStaticBitmap* imageDisplay;

    void OnButtonClicked(wxCommandEvent& event);
    void OnClose(wxCloseEvent& event);

    wxDECLARE_EVENT_TABLE();  // �̺�Ʈ ���̺� ����
};

