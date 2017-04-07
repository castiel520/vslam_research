#ifndef UTILS_H
#define UTILS_H

#include <opencv2/opencv.hpp>
#include <vector>
#include <iomanip>
#include "stats.h"


using namespace std;
using namespace cv;

std::vector<Point2f> Points(std::vector<KeyPoint> keypoints);

std::vector<Point2f> Points(std::vector<KeyPoint> keypoints)
{
    std::vector<Point2f> res;
    for(unsigned i = 0; i < keypoints.size(); i++) {
        res.push_back(keypoints[i].pt);
    }
    return res;
}

#endif // UTILS_H
