#ifndef PLANAR_TRACKING_H
#define PLANAR_TRACKING_H

#include <dlib/opencv.h>
#include <dlib/image_processing/frontal_face_detector.h>
#include <dlib/image_processing/render_face_detections.h>
#include <dlib/image_processing.h>
#include <dlib/gui_widgets.h>
#include <dlib/image_io.h>
#include <dlib/image_transforms.h>

using namespace std;
using namespace cv;

class Tracker
{
public:
    Tracker(Ptr<Feature2D> _detector, Ptr<DescriptorMatcher> _matcher, Mat _K) :
            detector(_detector),
            matcher(_matcher),
            K(_K)
            {
                dlib::deserialize("shape_predictor_68_face_landmarks.dat") >> pose_model;
            }
    //Tracker(){}

    void setFirstFrame(const char * first_frame_path);

    bool process(const Mat frame_left, bool slamMode);
    bool process_face(const Mat frame_left, bool slamMode);
    glm::mat4 getInitModelMatrix();

protected:
    std::vector<cv::Point2d> get_2d_image_points(dlib::full_object_detection &d);
    std::vector<cv::Point3d> get_3d_model_points();

protected:
    Mat K, rvec, tvec;
    Ptr<Feature2D> detector;
    Ptr<DescriptorMatcher> matcher;
    Mat first_frame, first_desc;
    std::vector<KeyPoint> first_kp;
    std::vector<Point2f> object_bb;
    dlib::shape_predictor pose_model;
};

#endif //BOB_AR_PLANAR_TRACKING_H
