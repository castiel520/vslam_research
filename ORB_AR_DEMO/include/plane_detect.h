/*
*plane detector
*based on plane detect cod from ORB_SLAM2.
*/


#ifndef PLANE_DETECT
#define PLANE_DETECT

#include <opencv2/core/core.hpp>
#include <string>
#include "ORB_SLAM/System.h"

using namespace ORB_SLAM2;

class Plane
{
public:
    Plane(const std::vector<MapPoint*> &vMPs, const cv::Mat &Tcw);
    Plane(const float &nx, const float &ny, const float &nz, const float &ox, const float &oy, const float &oz);

    void Recompute();

    //normal
    cv::Mat n;
    //origin
    cv::Mat o;
    //arbitrary orientation along normal
    float rang;
    //transformation from world to the plane
    cv::Mat Tpw;
    pangolin::OpenGlMatrix glTpw;
    //MapPoints that define the plane
    std::vector<MapPoint*> mvMPs;
    //camera pose when the plane was first observed (to compute normal direction)
    cv::Mat mTcw, XC;
};
#endif