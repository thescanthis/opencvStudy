#include "pch.h"
// ShapeExtractor.cpp
#include "ShapeExtractor.h"
#include <wx/log.h>

void ShapeExtractor::Analyze(const cv::Mat& roi)
{
    m_lines.clear();
    m_circles.clear();
    m_contours.clear();

    if (roi.empty()) return;

    // 1) GRAY
    cv::Mat gray;
    if (roi.channels() == 3)      cv::cvtColor(roi, gray, cv::COLOR_BGR2GRAY);
    else if (roi.channels() == 4) cv::cvtColor(roi, gray, cv::COLOR_BGRA2GRAY);
    else                          gray = roi.clone();

    // 2) ��� ���� + ����ȭ (���� ����ȭ)
    cv::Mat eq;
    cv::Ptr<cv::CLAHE> clahe = cv::createCLAHE(2.0, cv::Size(8, 8));
    clahe->apply(gray, eq);

    cv::Mat bin;
    cv::threshold(eq, bin, 0, 255, cv::THRESH_BINARY | cv::THRESH_OTSU);

    // �� ���/���� ���� �ƴ϶�� ����
    if (cv::countNonZero(bin) > (int)bin.total() / 2) cv::bitwise_not(bin, bin);

    const int W = bin.cols, H = bin.rows;
    const int shortDim = std::min(W, H);

    // 3) ����/���� �и�(���������� ���� �⼱ ����)
    int kx = std::max(15, W / 60);  // ���� ���� Ŀ�� ����
    int ky = std::max(15, H / 60);  // ���� ���� Ŀ�� ����

    cv::Mat horiz, vert;
    cv::erode(bin, horiz, cv::getStructuringElement(cv::MORPH_RECT, cv::Size(kx, 1)));
    cv::dilate(horiz, horiz, cv::getStructuringElement(cv::MORPH_RECT, cv::Size(kx, 1)));

    cv::erode(bin, vert, cv::getStructuringElement(cv::MORPH_RECT, cv::Size(1, ky)));
    cv::dilate(vert, vert, cv::getStructuringElement(cv::MORPH_RECT, cv::Size(1, ky)));

    // 4) ���� ���� (����/ƴ �������� ���� ����)
    const double minLen = std::max(40.0, 0.25 * shortDim);           // �ּ� ����
    const int    thr = 80;                                        // ���� �Ӱ�
    const int    maxGap = std::max(4, shortDim / 200);               // ���� ���

    std::vector<cv::Vec4i> hl, vl;
    cv::HoughLinesP(horiz, hl, 1, CV_PI / 180, thr, minLen, maxGap);
    cv::HoughLinesP(vert, vl, 1, CV_PI / 180, thr, minLen, maxGap);

    auto pushLine = [&](const cv::Vec4i& L) {
        cv::Point a(L[0], L[1]), b(L[2], L[3]);
        double len = cv::norm(a - b);
        if (len < minLen) return;
        // ������ 0/90�Ʒ� ���� (���� Ư��)
        double ang = std::atan2(double(b.y - a.y), double(b.x - a.x));
        double deg = ang * 180.0 / CV_PI;
        if (std::abs(std::sin(ang)) < 0.173) { // ~10�� �̳��� ��������
            b.y = a.y; ang = 0.0;
        }
        else if (std::abs(std::cos(ang)) < 0.173) { // ~10�� �̳��� ��������
            b.x = a.x; ang = CV_PI / 2;
        }
        m_lines.push_back({ a,b,cv::norm(a - b),ang });
        };
    for (auto& L : hl) pushLine(L);
    for (auto& L : vl) pushLine(L);

    // 5) �� ���� (Ŀ���� ��/�� �� ��)
    std::vector<cv::Vec3f> cs;
    const int minR = std::max(3, shortDim / 120);
    const int maxR = std::max(minR + 2, shortDim / 35);
    cv::HoughCircles(gray, cs, cv::HOUGH_GRADIENT, 1.2,
        shortDim / 15,   // �ּ� �߽� ����
        120, 30,       // Canny ��/���� �Ӱ�
        minR, maxR);
    for (auto& c : cs)
        m_circles.push_back({ cv::Point(cvRound(c[0]), cvRound(c[1])), c[2] });

    // 6) �簢��/�󿵿� �ĺ� (�ܰ� ���� ���̱� ���� bin ���)
    std::vector<std::vector<cv::Point>> cnts;
    cv::findContours(bin, cnts, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
    for (auto& c : cnts)
    {
        double area = cv::contourArea(c);
        if (area < 25.0) continue;                            // �ʹ� ���� ���� ����

        std::vector<cv::Point> poly;
        cv::approxPolyDP(c, poly, 0.02 * cv::arcLength(c, true), true);
        cv::Rect r = cv::boundingRect(c);
        if (r.width < 5 || r.height < 5) continue;
        m_contours.push_back({ r, area });       // �����(�����)
        if (poly.size() == 4 && cv::isContourConvex(poly)) {
            // �� �ڽ�/�е���� ����
            m_rects.push_back({ r, area });
            
        }
    }
}

void ShapeExtractor::DebugPrint() const
{
    wxLogMessage("ShapeExtractor Debug:");
    wxLogMessage("  Lines: %zu  Circles: %zu  Contours: %zu  Rects: %zu",
        m_lines.size(), m_circles.size(), m_contours.size(), m_rects.size());

    for (size_t i = 0; i < m_lines.size(); ++i)
        wxLogMessage("  L%zu: (%d,%d)-(%d,%d) len=%.1f angle=%.1f",
            i, m_lines[i].p1.x, m_lines[i].p1.y, m_lines[i].p2.x, m_lines[i].p2.y,
            m_lines[i].length, m_lines[i].angle * 180.0 / M_PI);
}

cv::Mat ShapeExtractor::BuildMask(const cv::Size& sz, int lineThick, int circleThick, bool fillRects) const
{
    cv::Mat mask(sz, CV_8UC1, cv::Scalar(0));
        // ����
        for (const auto& L : m_lines)
         cv::line(mask, L.p1, L.p2, cv::Scalar(255), std::max(1, lineThick), cv::LINE_AA);
        // ��
        for (const auto& C : m_circles)
         cv::circle(mask, C.center, (int)std::round(C.radius), cv::Scalar(255),
            +std::max(1, circleThick), cv::LINE_AA);
        // �簢��(�е�/�� �ڽ�)
        for (const auto& R : m_rects) {
            if (fillRects) cv::rectangle(mask, R.r, cv::Scalar(255), cv::FILLED);
            else           cv::rectangle(mask, R.r, cv::Scalar(255), 1);
        
        }
    return mask;
}

