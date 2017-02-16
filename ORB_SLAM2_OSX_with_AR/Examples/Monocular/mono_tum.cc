#include <iostream>
#include <algorithm> 
#include <fstream>
#include <chrono>
#include <opencv2/core/core.hpp>
#include <opencv2/opencv.hpp>
#include <System.h>

using namespace std;
int main(int argc, char **argv) 
{
   cv::VideoCapture inputVideo(0);
   if (!inputVideo.isOpened())
   {
     cerr << endl  <<"Could not open camera feed."  << endl;
     return -1;
   }
   // Create SLAM system. It initializes all system threads and gets ready to process frames.
   ORB_SLAM2::System SLAM("Vocabulary/ORBvoc.txt","Examples/Monocular/TUM1.yaml",ORB_SLAM2::System::MONOCULAR,false);

   cout << endl << "-------" << endl;
   cout << "Start processing sequence ..." << endl;
   inputVideo.set(CV_CAP_PROP_FRAME_WIDTH, 640);
   inputVideo.set(CV_CAP_PROP_FRAME_HEIGHT, 480);
   // Main loop
   int timeStamps=0;

  while (inputVideo.grab()) {
    cv::Mat image;
    inputVideo.retrieve(image);
     // Pass the image to the SLAM system
     SLAM.TrackMonocular(image, timeStamps);
     timeStamps++;
    }
    
   // Stop all threads
   SLAM.Shutdown();

    // Save camera trajectory
    SLAM.SaveKeyFrameTrajectoryTUM("KeyFrameTrajectory.txt");
    return 0;
}

