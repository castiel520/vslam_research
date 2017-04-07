#define GLM_FORCE_RADIANS
#include <glm/glm.hpp>
#include <glm/gtx/transform.hpp>

#include <opencv2/opencv.hpp>
#include "planar_tracking.h"
#include "utils.h"

using namespace cv;
using namespace std;

glm::mat4 Tracker::getInitModelMatrix(){

    glm::mat4 initModelMatrix;
    Mat initR;
    Mat viewMatrix = cv::Mat::zeros(4, 4, CV_64FC1);
    Rodrigues(rvec, initR);

    for(unsigned int row=0; row<3; ++row)
    {
        for(unsigned int col=0; col<3; ++col)
        {
            viewMatrix.at<double>(row, col) = initR.at<double>(row, col);
        }
        viewMatrix.at<double>(row, 3) = tvec.at<double>(row, 0);
    }
    viewMatrix.at<double>(3, 3) = 1.0f;

    //viewMatrix = cvToGl * viewMatrix;

    viewMatrix.convertTo(viewMatrix, CV_32F);


    for (int i = 0; i < 4; i++){
        for (int j = 0; j < 4; j++) {
            initModelMatrix[i][j] = viewMatrix.at<float>(j,i);
        }
    }

    return initModelMatrix;

}

void Tracker::setFirstFrame(const char * first_frame_path)
{
    
    first_frame = imread(first_frame_path);
    std::vector<KeyPoint> kp;

    object_bb.push_back(Point2f(0,0));
    object_bb.push_back(Point2f(8.4,0));
    object_bb.push_back(Point2f(8.4,4.7));
    object_bb.push_back(Point2f(0,4.7));

    std::vector<Point2f> bb;
    bb.push_back(Point2f(0,0));
    bb.push_back(Point2f(672,0));
    bb.push_back(Point2f(672,376));
    bb.push_back(Point2f(0,376));

    Mat H;
    H = findHomography(bb, object_bb);
    
    detector->detectAndCompute(first_frame, noArray(), kp, first_desc);

    std::vector<Point2f> tmp_kp_orgn, tmp_kp_homo;

    first_kp = kp;

    for (int i = 0; i <= kp.size(); i++) {
        tmp_kp_orgn.push_back(Point2f(kp[i].pt));
    }

    perspectiveTransform(tmp_kp_orgn, tmp_kp_homo, H);

    for (int i = 0; i <= kp.size(); i++) {
        first_kp[i].pt = tmp_kp_homo[i];
    }
}

bool Tracker::process(const Mat frame_left, bool slamMode)
{

    if (slamMode)
        return 1;
    std::vector<KeyPoint> kp;
    std::vector<Point3f> ObjectPoints;
    std::vector<Point2f> ImagePoints;
    Mat desc;
    
    detector->detectAndCompute(frame_left, noArray(), kp, desc);
    
    if(kp.size()<10)
    {
        return 0;
    }
    
    std::vector< std::vector<DMatch> > matches;
    std::vector<KeyPoint> matched1, matched2;
    matcher->knnMatch(first_desc, desc, matches, 2);
    for(unsigned i = 0; i < matches.size(); i++) {
        if(matches[i][0].distance < 0.8f * matches[i][1].distance) {
            matched1.push_back(first_kp[matches[i][0].queryIdx]);
            matched2.push_back(      kp[matches[i][0].trainIdx]);
        }
    }

    Mat inlier_mask, homography;

    int thd = 10;//(int)(0.04*first_desc.rows);

    if(matched1.size() >= thd) {
        homography = findHomography(Points(matched1), Points(matched2),
                                    RANSAC, 10.0f, inlier_mask);
    }

    if(matched1.size() < thd) {
        printf(" < thd\n");
        return 0;
    }

    if(homography.empty())
    {
        printf(" no homography\n");
        return 0;
    }


    for(unsigned i = 0; i < matched1.size(); i++) {
        if(inlier_mask.at<uchar>(i)) {
            ObjectPoints.push_back(Point3f(matched1[i].pt.x, matched1[i].pt.y, 0));
            ImagePoints.push_back(Point2f(matched2[i].pt.x, matched2[i].pt.y));
        }
    }
    solvePnP(ObjectPoints, ImagePoints, K, noArray(), rvec, tvec,CV_ITERATIVE);

    return 1;
}
std::vector<cv::Point3d> Tracker::get_3d_model_points()
{
    std::vector<cv::Point3d> modelPoints;

    modelPoints.push_back(cv::Point3d(0.0f, 0.0f, 0.0f)); //The first must be (0,0,0) while using POSIT
    modelPoints.push_back(cv::Point3d(0.0f, -3.30f, -6.50f));
    modelPoints.push_back(cv::Point3d(-22.50f, 17.0f, -13.50f));
    modelPoints.push_back(cv::Point3d(22.50f, 17.0f, -13.50f));
    modelPoints.push_back(cv::Point3d(-15.00f, -15.0f, -12.50f));
    modelPoints.push_back(cv::Point3d(15.00f, -15.0f, -12.50f));
    
    return modelPoints;
}

std::vector<cv::Point2d> Tracker::get_2d_image_points(dlib::full_object_detection &d)
{
    std::vector<cv::Point2d> image_points;
    image_points.push_back( cv::Point2d( d.part(30).x(), d.part(30).y() ) );    // Nose tip
    image_points.push_back( cv::Point2d( d.part(8).x(), d.part(8).y() ) );      // Chin
    image_points.push_back( cv::Point2d( d.part(36).x(), d.part(36).y() ) );    // Left eye left corner
    image_points.push_back( cv::Point2d( d.part(45).x(), d.part(45).y() ) );    // Right eye right corner
    image_points.push_back( cv::Point2d( d.part(48).x(), d.part(48).y() ) );    // Left Mouth corner
    image_points.push_back( cv::Point2d( d.part(54).x(), d.part(54).y() ) );    // Right mouth corner
    return image_points;
}

bool Tracker::process_face(const Mat frame_left, bool slamMode)
{
    if (slamMode)
    {
        return true;
    }
    
    Mat im_small;

    if (frame_left.empty())
    {
        return false;
    }

    cv::resize(frame_left,im_small,cv::Size(),0.25,0.25);
    cv::Size size = frame_left.size();

    dlib::frontal_face_detector detector = dlib::get_frontal_face_detector();

    int count = 0;
    std::vector<dlib::rectangle> faces;

    dlib::cv_image<dlib::bgr_pixel> cimg_small(im_small);
    dlib::cv_image<dlib::bgr_pixel> cimg(frame_left);

    faces = detector(cimg_small);
    std::vector<cv::Point3d> model_points = get_3d_model_points();

    for (int i = 0; i < faces.size(); ++i)
    {
        dlib::rectangle r(
            (long)(faces[i].left() * 4),
            (long)(faces[i].top() * 4),
            (long)(faces[i].right() * 4),
            (long)(faces[i].bottom() * 4)
            );
        dlib::full_object_detection shape = pose_model(cimg, r);
        std::vector<cv::Point2d> image_points = get_2d_image_points(shape);
        cv::solvePnP(model_points, image_points, K, noArray(), rvec, tvec);
        //cv::imshow("Fast Facial Landmark Detector", frame_left);
        return true;
    }
    return false;
}