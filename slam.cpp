// slam.cpp : Defines the entry point for the console application.
//

#include "stdafx.h"

#include <opencv2/core.hpp> 
#include <opencv2/imgcodecs.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/xfeatures2d/nonfree.hpp>

#include <opencv2/highgui/highgui.hpp>

#include <g2o/core/sparse_optimizer.h>
#include <g2o/core/block_solver.h>
#include <g2o/core/optimization_algorithm_levenberg.h>
#include <g2o/core/robust_kernel.h>
#include <g2o/core/robust_kernel_impl.h>
#include <g2o/solvers/eigen/linear_solver_eigen.h>
#include <g2o/types/sba/types_six_dof_expmap.h>
#include <g2o/types/slam3d/se3quat.h>

int findCorrespondingPoints(const cv::Mat& img1, const cv::Mat& img2, std::vector<cv::Point2f>& points1, std::vector<cv::Point2f>& points2)
{
  auto orb = cv::ORB::create();
  
  //auto sift = cv::xfeatures2d::SIFT::create();

  std::vector<cv::KeyPoint> kp1, kp2;

  cv::Mat desp1, desp2;

  orb->detectAndCompute(img1, cv::Mat(), kp1, desp1);

  orb->detectAndCompute(img2, cv::Mat(), kp2, desp2);

  //sift->detectAndCompute(img1, cv::Mat(), kp1, desp1);

  //sift->detectAndCompute(img2, cv::Mat(), kp2, desp2);

  auto matcher = cv::DescriptorMatcher::create("BruteForce-Hamming");

  std::vector<std::vector<cv::DMatch>> matches_knn;

  matcher->knnMatch(desp1, desp2, matches_knn, 2);

  std::vector<cv::DMatch> matches;
  for (size_t i = 0; i < matches_knn.size(); i++)
  {
    if (matches_knn[i][0].distance < matches_knn[i][1].distance * 0.8)
    {
      matches.push_back(matches_knn[i][0]);
    }
  }

  for (const auto& match : matches)
  {
    points1.push_back(kp1[match.queryIdx].pt);
    points2.push_back(kp2[match.trainIdx].pt);
  }

  return true;
}
int main()
{

  std::string first_image = "D:\\project\\slam\\slam\\1.png";
  std::string second_image = "D:\\project\\slam\\slam\\2.png";

  cv::Mat img1 = cv::imread(first_image);
  cv::Mat img2 = cv::imread(second_image);

  std::vector<cv::Point2f> pts1, pts2;

  findCorrespondingPoints(img1, img2, pts1, pts2);


  g2o::SparseOptimizer optimizer;

  g2o::BlockSolver_6_3::LinearSolverType* linear_solver = new g2o::LinearSolverEigen<g2o::BlockSolver_6_3::PoseMatrixType>();

  g2o::BlockSolver_6_3* block_solver = new g2o::BlockSolver_6_3(linear_solver);

  g2o::OptimizationAlgorithmLevenberg* algorithm = new g2o::OptimizationAlgorithmLevenberg(block_solver);

  optimizer.setAlgorithm(algorithm);
  optimizer.setVerbose(false);

  // add camera parameter
  g2o::CameraParameters* camera = new g2o::CameraParameters(518, g2o::Vector2D(325.5, 253.5), 0.0);;
  camera->setId(0);
  optimizer.addParameter(camera);


  // add pos node
  for (int i = 0; i < 2; i++)
  {
    g2o::VertexSE3Expmap* v = new g2o::VertexSE3Expmap();
    v->setEstimate(g2o::SE3Quat());
    v->setId(i);

    if (0 == i)
    {
      v->setFixed(true);
    }

    optimizer.addVertex(v);
  }

  // add landmark node
  for (int i = 0; i < pts1.size(); i++)
  {
    g2o::VertexSBAPointXYZ* v = new g2o::VertexSBAPointXYZ();
    v->setId(2 + i);
    
    //g2o::CameraParameters* camera = new g2o::CameraParameters(518, g2o::Vector2D(325.5, 253.5), 0.0);;
    double x = (pts1[i].x - 325.5) / 518;
    double y = (pts1[i].y - 253.5) / 519;

    v->setEstimate(g2o::Vector3D(x,y,1));
    v->setMarginalized(true);
    optimizer.addVertex(v);
  }

  // add Edge for first frame
  for (int i = 0; i < pts1.size(); i++)
  {
    g2o::EdgeProjectXYZ2UV* e = new g2o::EdgeProjectXYZ2UV();
    e->setVertex(1, optimizer.vertex(0));
    e->setVertex(0, optimizer.vertex(i + 2));
    e->setMeasurement(Eigen::Vector2d(pts1[i].x, pts1[i].y));
    e->setInformation(Eigen::Matrix2d::Identity());
    e->setParameterId(0, 0);
    e->setRobustKernel(new g2o::RobustKernelHuber());
    optimizer.addEdge(e);
  }

  // add Edge for first frame
  for (int i = 0; i < pts2.size(); i++)
  {
    g2o::EdgeProjectXYZ2UV* e = new g2o::EdgeProjectXYZ2UV();
    e->setVertex(1, optimizer.vertex(1));
    e->setVertex(0, optimizer.vertex(i + 2));
    e->setMeasurement(Eigen::Vector2d(pts2[i].x, pts2[i].y));
    e->setInformation(Eigen::Matrix2d::Identity());
    e->setParameterId(0, 0);
    e->setRobustKernel(new g2o::RobustKernelHuber());
    optimizer.addEdge(e);
  }

  optimizer.setVerbose(true);
  optimizer.initializeOptimization();
  optimizer.optimize(10);

  g2o::VertexSE3Expmap* v = dynamic_cast<g2o::VertexSE3Expmap*>(optimizer.vertex(1));
  Eigen::Isometry3d pose = v->estimate();

  auto pos_matrix = pose.matrix();
  return 0;
}

