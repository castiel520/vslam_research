#include "plane_detect.h"

#include <opencv2/highgui/highgui.hpp>
using namespace std;

const float eps = 1e-4;
cv::Mat ExpSO3(const float &x, const float &y, const float &z)
{
    cv::Mat I = cv::Mat::eye(3,3,CV_32F);
    const float d2 = x*x+y*y+z*z;
    const float d = sqrt(d2);
    cv::Mat W = (cv::Mat_<float>(3,3) << 0, -z, y,
                 z, 0, -x,
                 -y,  x, 0);
    if(d<eps)
        return (I + W + 0.5f*W*W);
    else
        return (I + W*sin(d)/d + W*W*(1.0f-cos(d))/d2);
}

cv::Mat ExpSO3(const cv::Mat &v)
{
    return ExpSO3(v.at<float>(0),v.at<float>(1),v.at<float>(2));
}
Plane::Plane(const std::vector<MapPoint *> &vMPs, const cv::Mat &Tcw):mvMPs(vMPs),mTcw(Tcw.clone())
{
    rang = -3.14f/2+((float)rand()/RAND_MAX)*3.14f;
    Recompute();
}

void Plane::Recompute()
{
    const int N = mvMPs.size();

    // Recompute plane with all points
    cv::Mat A = cv::Mat(N,4,CV_32F);
    A.col(3) = cv::Mat::ones(N,1,CV_32F);

    o = cv::Mat::zeros(3,1,CV_32F);

    int nPoints = 0;
    for(int i=0; i<N; i++)
    {
        MapPoint* pMP = mvMPs[i];
        if(!pMP->isBad())
        {
            cv::Mat Xw = pMP->GetWorldPos();
            o+=Xw;
            A.row(nPoints).colRange(0,3) = Xw.t();
            nPoints++;
        }
    }
    A.resize(nPoints);

    cv::Mat u,w,vt;
    cv::SVDecomp(A,w,u,vt,cv::SVD::MODIFY_A | cv::SVD::FULL_UV);

    float a = vt.at<float>(3,0);
    float b = vt.at<float>(3,1);
    float c = vt.at<float>(3,2);

    o = o*(1.0f/nPoints);
    const float f = 1.0f/sqrt(a*a+b*b+c*c);

    // Compute XC just the first time
    if(XC.empty())
    {
        cv::Mat Oc = -mTcw.colRange(0,3).rowRange(0,3).t()*mTcw.rowRange(0,3).col(3);
        XC = Oc-o;
    }

    if((XC.at<float>(0)*a+XC.at<float>(1)*b+XC.at<float>(2)*c)>0)
    {
        a=-a;
        b=-b;
        c=-c;
    }

    const float nx = a*f;
    const float ny = b*f;
    const float nz = c*f;

    n = (cv::Mat_<float>(3,1)<<nx,ny,nz);

    cv::Mat up = (cv::Mat_<float>(3,1) << 0.0f, 1.0f, 0.0f);

    cv::Mat v = up.cross(n);
    const float sa = cv::norm(v);
    const float ca = up.dot(n);
    const float ang = atan2(sa,ca);
    Tpw = cv::Mat::eye(4,4,CV_32F);


    Tpw.rowRange(0,3).colRange(0,3) = ExpSO3(v*ang/sa)*ExpSO3(up*rang);
    o.copyTo(Tpw.col(3).rowRange(0,3));

    glTpw.m[0] = Tpw.at<float>(0,0);
    glTpw.m[1] = Tpw.at<float>(1,0);
    glTpw.m[2] = Tpw.at<float>(2,0);
    glTpw.m[3]  = 0.0;

    glTpw.m[4] = Tpw.at<float>(0,1);
    glTpw.m[5] = Tpw.at<float>(1,1);
    glTpw.m[6] = Tpw.at<float>(2,1);
    glTpw.m[7]  = 0.0;

    glTpw.m[8] = Tpw.at<float>(0,2);
    glTpw.m[9] = Tpw.at<float>(1,2);
    glTpw.m[10] = Tpw.at<float>(2,2);
    glTpw.m[11]  = 0.0;

    glTpw.m[12] = Tpw.at<float>(0,3);
    glTpw.m[13] = Tpw.at<float>(1,3);
    glTpw.m[14] = Tpw.at<float>(2,3);
    glTpw.m[15]  = 1.0;

}

Plane::Plane(const float &nx, const float &ny, const float &nz, const float &ox, const float &oy, const float &oz)
{
    n = (cv::Mat_<float>(3,1)<<nx,ny,nz);
    o = (cv::Mat_<float>(3,1)<<ox,oy,oz);

    cv::Mat up = (cv::Mat_<float>(3,1) << 0.0f, 1.0f, 0.0f);

    cv::Mat v = up.cross(n);
    const float s = cv::norm(v);
    const float c = up.dot(n);
    const float a = atan2(s,c);
    Tpw = cv::Mat::eye(4,4,CV_32F);
    const float rang = -3.14f/2+((float)rand()/RAND_MAX)*3.14f;
    cout << rang;
    Tpw.rowRange(0,3).colRange(0,3) = ExpSO3(v*a/s)*ExpSO3(up*rang);
    o.copyTo(Tpw.col(3).rowRange(0,3));

    glTpw.m[0] = Tpw.at<float>(0,0);
    glTpw.m[1] = Tpw.at<float>(1,0);
    glTpw.m[2] = Tpw.at<float>(2,0);
    glTpw.m[3]  = 0.0;

    glTpw.m[4] = Tpw.at<float>(0,1);
    glTpw.m[5] = Tpw.at<float>(1,1);
    glTpw.m[6] = Tpw.at<float>(2,1);
    glTpw.m[7]  = 0.0;

    glTpw.m[8] = Tpw.at<float>(0,2);
    glTpw.m[9] = Tpw.at<float>(1,2);
    glTpw.m[10] = Tpw.at<float>(2,2);
    glTpw.m[11]  = 0.0;

    glTpw.m[12] = Tpw.at<float>(0,3);
    glTpw.m[13] = Tpw.at<float>(1,3);
    glTpw.m[14] = Tpw.at<float>(2,3);
    glTpw.m[15]  = 1.0;
}