/**
* This file is part of ORB-SLAM2.
*
* Copyright (C) 2014-2016 Raúl Mur-Artal <raulmur at unizar dot es> (University of Zaragoza)
* For more information see <https://github.com/raulmur/ORB_SLAM2>
*
* ORB-SLAM2 is free software: you can redistribute it and/or modify
* it under the terms of the GNU General Public License as published by
* the Free Software Foundation, either version 3 of the License, or
* (at your option) any later version.
*
* ORB-SLAM2 is distributed in the hope that it will be useful,
* but WITHOUT ANY WARRANTY; without even the implied warranty of
* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
* GNU General Public License for more details.
*
* You should have received a copy of the GNU General Public License
* along with ORB-SLAM2. If not, see <http://www.gnu.org/licenses/>.
*/


#include<iostream>
#include<algorithm>
#include<fstream>
#include<chrono>

#include<opencv2/core/core.hpp>

#include<System.h>

using namespace std;

void LoadImages(const string &strImagePath, const string &strPathTimes,
                vector<string> &vstrImages, vector<double> &vTimeStamps);

int main(int argc, char **argv)
{
  cv::VideoCapture cap(0);
  if (!cap.isOpened())
  {
    cerr << endl << "Could not open camera feed." << endl;
    return -1;
  }

  std::string voc_file = "D:\\opensource\\ORB_SLAM2\\Vocabulary\\ORBvoc.txt";
  std::string yaml_file = "D:\\opensource\\ORB_SLAM2\\Examples\\Monocular\\TUM1.yaml";

  std::cout << voc_file << std::endl;
  std::cout << yaml_file << std::endl;

  std::cout << "Hello" << std::endl;
  // Create SLAM system. It initializes all system threads and gets ready to process frames.
  ORB_SLAM2::System SLAM(voc_file, yaml_file, ORB_SLAM2::System::MONOCULAR, true);

  cout << endl << "-------" << endl;
  cout << "Start processing sequence ..." << endl;

  // Main loop
  int timeStamps = 0;
  for (;; timeStamps++)
  {
    //Create a new Mat
    cv::Mat frame;

    //Send the captured frame to the new Mat
    cap >> frame;

    // Pass the image to the SLAM system
    SLAM.TrackMonocular(frame, timeStamps);
  }

  // Stop all threads
  SLAM.Shutdown();

  // Save camera trajectory
  SLAM.SaveKeyFrameTrajectoryTUM("KeyFrameTrajectory.txt");
  return 0;

}

void LoadImages(const string &strImagePath, const string &strPathTimes,
                vector<string> &vstrImages, vector<double> &vTimeStamps)
{
    ifstream fTimes;
    fTimes.open(strPathTimes.c_str());
    vTimeStamps.reserve(5000);
    vstrImages.reserve(5000);
    while(!fTimes.eof())
    {
        string s;
        getline(fTimes,s);
        if(!s.empty())
        {
            stringstream ss;
            ss << s;
            vstrImages.push_back(strImagePath + "/" + ss.str() + ".png");
            double t;
            ss >> t;
            vTimeStamps.push_back(t/1e9);

        }
    }
}
