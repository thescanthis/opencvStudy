#pragma once
#include <wx/wx.h>
#include <wx/statbmp.h>    // wxStaticBitmap
#include <wx/dcbuffer.h>
#include <wx/dcgraph.h>   // wxGCDC  ← 이거!
#include <wx/graphics.h>  // (옵션) wxGraphicsContext
#include <opencv2/opencv.hpp>


#include <vector>
#include <algorithm>

#if _DEBUG
#pragma comment (lib,"wxbase32ud.lib")
#pragma comment (lib,"wxbase32ud_net.lib")
#pragma comment (lib,"wxbase32ud_xml.lib")
#pragma comment (lib,"wxexpatd.lib")
#pragma comment (lib,"wxjpegd.lib")
#pragma comment (lib,"wxmsw32ud_adv.lib")
#pragma comment (lib,"wxmsw32ud_aui.lib")
#pragma comment (lib,"wxmsw32ud_core.lib")
#pragma comment (lib,"wxmsw32ud_gl.lib")
#pragma comment (lib,"wxmsw32ud_html.lib")
#pragma comment (lib,"wxmsw32ud_media.lib")
#pragma comment (lib,"wxmsw32ud_propgrid.lib")
#pragma comment (lib,"wxmsw32ud_qa.lib")
#pragma comment (lib,"wxmsw32ud_ribbon.lib")
#pragma comment (lib,"wxmsw32ud_richtext.lib")
#pragma comment (lib,"wxmsw32ud_stc.lib")
#pragma comment (lib,"wxmsw32ud_webview.lib")
#pragma comment (lib,"wxmsw32ud_xrc.lib")
#pragma comment (lib,"wxpngd.lib")
#pragma comment (lib,"wxregexud.lib")
#pragma comment (lib,"wxscintillad.lib")
#pragma comment (lib,"wxtiffd.lib")
#pragma comment (lib,"wxzlibd.lib")
#pragma comment (lib,"Rpcrt4.lib")
#pragma comment (lib,"Comctl32.lib")
#elif Relelase Librarys
#endif

// OpenCV Debug Libraries - Version 4.13.0
#pragma comment(lib, "opencv_calib3d4130d.lib")
#pragma comment(lib, "opencv_core4130d.lib")
#pragma comment(lib, "opencv_dnn4130d.lib")
#pragma comment(lib, "opencv_features2d4130d.lib")
#pragma comment(lib, "opencv_flann4130d.lib")
#pragma comment(lib, "opencv_gapi4130d.lib")
#pragma comment(lib, "opencv_highgui4130d.lib")
#pragma comment(lib, "opencv_imgcodecs4130d.lib")
#pragma comment(lib, "opencv_imgproc4130d.lib")
#pragma comment(lib, "opencv_ml4130d.lib")
#pragma comment(lib, "opencv_objdetect4130d.lib")
#pragma comment(lib, "opencv_photo4130d.lib")
#pragma comment(lib, "opencv_stitching4130d.lib")
#pragma comment(lib, "opencv_video4130d.lib")
#pragma comment(lib, "opencv_videoio4130d.lib")
