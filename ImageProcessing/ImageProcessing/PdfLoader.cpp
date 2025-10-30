#include "pch.h"
#include <poppler/cpp/poppler-document.h>
#include <poppler/cpp/poppler-page.h>
#include <poppler/cpp/poppler-image.h>
#include <poppler/cpp/poppler-page-renderer.h>
#ifdef _WIN32
#include <windows.h>
#endif
#include <memory>
#include <string>

static std::string WideToUtf8(const std::wstring& ws)
{
#ifdef _WIN32
    if (ws.empty()) return {};
    int n = WideCharToMultiByte(CP_UTF8, 0, ws.c_str(), (int)ws.size(), nullptr, 0, nullptr, nullptr);
    std::string s(n, 0);
    WideCharToMultiByte(CP_UTF8, 0, ws.c_str(), (int)ws.size(), s.data(), n, nullptr, nullptr);
    return s;
#else
    return std::string(ws.begin(), ws.end());
#endif
}

static cv::Mat RenderPageUsingRenderer(poppler::page* pg, int dpi)
{
    if (!pg) return {};

    poppler::page_renderer renderer;
    renderer.set_image_format(poppler::image::format_enum::format_argb32);
    poppler::image img = renderer.render_page(pg, dpi, dpi);
    if (!img.is_valid()) return {};

    int w = img.width();
    int h = img.height();
    const unsigned char* src = reinterpret_cast<const unsigned char*>(img.const_data());
    int stride = img.bytes_per_row();

    cv::Mat bgra(h, w, CV_8UC4, const_cast<unsigned char*>(src), stride);
    cv::Mat bgr;
    cv::cvtColor(bgra, bgr, cv::COLOR_BGRA2BGR);
    return bgr.clone();
}

std::vector<cv::Mat> LoadPdfAllPages_Poppler(const std::wstring& wpath, int dpi)
{
    std::vector<cv::Mat> pages;
    auto path = WideToUtf8(wpath);

    std::unique_ptr<poppler::document> doc(poppler::document::load_from_file(path));
    if (!doc) return pages;

    int pageCount = doc->pages();
    pages.reserve(pageCount);

    for (int i = 0; i < pageCount; ++i) {
        std::unique_ptr<poppler::page> pg(doc->create_page(i));
        if (!pg) continue;
        cv::Mat m = RenderPageUsingRenderer(pg.get(), dpi);
        if (!m.empty()) pages.push_back(std::move(m));
    }

    return pages;
}

cv::Mat LoadPdfPage_Poppler(const std::wstring& wpath, int pageIndex, int dpi)
{
    auto path = WideToUtf8(wpath);
    std::unique_ptr<poppler::document> doc(poppler::document::load_from_file(path));
    if (!doc) return {};
    if (pageIndex < 0 || pageIndex >= doc->pages()) return {};

    std::unique_ptr<poppler::page> pg(doc->create_page(pageIndex));
    if (!pg) return {};
    return RenderPageUsingRenderer(pg.get(), dpi);
}
