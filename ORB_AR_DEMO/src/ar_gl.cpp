#include <iostream>
#include <opencv2/highgui.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc/imgproc.hpp>

//include GLEW
#include <GL/glew.h>

//#include GLFW
#include <GLFW/glfw3.h>
GLFWwindow *window;     

#define GLM_FORCE_RADIANS
// Include GLM
#include <glm/glm.hpp>
#include <glm/gtx/transform.hpp>
#include <glm/gtc/type_ptr.hpp>

using namespace std;
using namespace cv;
#include "ORB_SLAM/System.h"

#include "objloader.hpp"
#include "shader.hpp"
#include "texture.hpp"

#include "plane_detect.h"

#include <mutex>
#include <thread>
#include <cstdlib>
//global var
int width, height;
cv::Mat K;
cv::Mat DistCoef;
float scale_factor =1.0f;
Plane* DetectPlane(const cv::Mat Tcw, const std::vector<MapPoint*> &vMPs, const int iterations)
{
    // Retrieve 3D points
    vector<cv::Mat> vPoints;
    vPoints.reserve(vMPs.size());
    vector<MapPoint*> vPointMP;
    vPointMP.reserve(vMPs.size());

    for(size_t i=0; i<vMPs.size(); i++)
    {
        MapPoint* pMP=vMPs[i];
        if(pMP)
        {
            if(pMP->Observations()>5)
            {
                vPoints.push_back(pMP->GetWorldPos());
                vPointMP.push_back(pMP);
            }
        }
    }

    const int N = vPoints.size();

    if(N<50)
        return NULL;


    // Indices for minimum set selection
    vector<size_t> vAllIndices;
    vAllIndices.reserve(N);
    vector<size_t> vAvailableIndices;

    for(int i=0; i<N; i++)
    {
        vAllIndices.push_back(i);
    }

    float bestDist = 1e10;
    vector<float> bestvDist;

    //RANSAC
    for(int n=0; n<iterations; n++)
    {
        vAvailableIndices = vAllIndices;

        cv::Mat A(3,4,CV_32F);
        A.col(3) = cv::Mat::ones(3,1,CV_32F);

        // Get min set of points
        for(short i = 0; i < 3; ++i)
        {
            int randi = DUtils::Random::RandomInt(0, vAvailableIndices.size()-1);

            int idx = vAvailableIndices[randi];

            A.row(i).colRange(0,3) = vPoints[idx].t();

            vAvailableIndices[randi] = vAvailableIndices.back();
            vAvailableIndices.pop_back();
        }

        cv::Mat u,w,vt;
        cv::SVDecomp(A,w,u,vt,cv::SVD::MODIFY_A | cv::SVD::FULL_UV);

        const float a = vt.at<float>(3,0);
        const float b = vt.at<float>(3,1);
        const float c = vt.at<float>(3,2);
        const float d = vt.at<float>(3,3);

        vector<float> vDistances(N,0);

        const float f = 1.0f/sqrt(a*a+b*b+c*c+d*d);

        for(int i=0; i<N; i++)
        {
            vDistances[i] = fabs(vPoints[i].at<float>(0)*a+vPoints[i].at<float>(1)*b+vPoints[i].at<float>(2)*c+d)*f;
        }

        vector<float> vSorted = vDistances;
        sort(vSorted.begin(),vSorted.end());

        int nth = max((int)(0.2*N),20);
        const float medianDist = vSorted[nth];

        if(medianDist<bestDist)
        {
            bestDist = medianDist;
            bestvDist = vDistances;
        }
    }

    // Compute threshold inlier/outlier
    const float th = 1.4*bestDist;
    vector<bool> vbInliers(N,false);
    int nInliers = 0;
    for(int i=0; i<N; i++)
    {
        if(bestvDist[i]<th)
        {
            nInliers++;
            vbInliers[i]=true;
        }
    }

    vector<MapPoint*> vInlierMPs(nInliers,NULL);
    int nin = 0;
    for(int i=0; i<N; i++)
    {
        if(vbInliers[i])
        {
            vInlierMPs[nin] = vPointMP[i];
            nin++;
        }
    }

    return new Plane(vInlierMPs,Tcw);
}

int main(int argc, char **argv)
{
  //hardcode for now
  width = 640;
  height = 480;

  VideoCapture inputVideo(1);

  ORB_SLAM2::System SLAM(argv[1],argv[2],ORB_SLAM2::System::MONOCULAR,false);

  //get K and DistCoef from yaml
  cv::FileStorage fSettings(argv[2], cv::FileStorage::READ);
  float fps = fSettings["Camera.fps"];

  float f_x = fSettings["Camera.fx"];
  float f_y = fSettings["Camera.fy"];
  float c_x = fSettings["Camera.cx"];
  float c_y = fSettings["Camera.cy"];

  K = cv::Mat::eye(3,3,CV_32F);
  K.at<float>(0,0) = f_x;
  K.at<float>(1,1) = f_y;
  K.at<float>(0,2) = c_x;
  K.at<float>(1,2) = c_y;

  DistCoef = cv::Mat::zeros(4,1,CV_32F);
  DistCoef.at<float>(0) = fSettings["Camera.k1"];
  DistCoef.at<float>(1) = fSettings["Camera.k2"];
  DistCoef.at<float>(2) = fSettings["Camera.p1"];
  DistCoef.at<float>(3) = fSettings["Camera.p2"];
  const float k3 = fSettings["Camera.k3"];
  if(k3!=0)
  {
      DistCoef.resize(5);
      DistCoef.at<float>(4) = k3;
  }

  /////////////////////////////////working
  /// area/////////////////////////////////////////////////////////////////////////////
  // Initialise GLFW
  if (!glfwInit()) {
    fprintf(stderr, "Failed to initialize GLFW\n");
    getchar();
    return -1;
  }

  glfwWindowHint(GLFW_SAMPLES, 4);
  glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
  glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
  glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT,
                 GL_TRUE);  // To make MacOS happy; should not be needed
  glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

  // Open a window and create its OpenGL context
  window = glfwCreateWindow(640, 480, "ORB AR", NULL, NULL);
  if (window == NULL) {
    fprintf(stderr,
            "Failed to open GLFW window. If you have an Intel GPU, they are "
            "not 3.3 compatible. Try the 2.1 version of the tutorials.\n");
    getchar();
    glfwTerminate();
    return -1;
  }
  glfwMakeContextCurrent(window);
  printf("OpenGL version supported by this platform (%s): \n",
         glGetString(GL_VERSION));

  // Initialize GLEW
  glewExperimental = GL_TRUE;  // Needed for core profile
  if (glewInit() != GLEW_OK) {
    fprintf(stderr, "Failed to initialize GLEW\n");
    getchar();
    glfwTerminate();
    return -1;
  }
  glClearColor(0.0f, 0.0f, 0.4f, 0.0f);
  glEnable(GL_DEPTH_TEST);
  glDepthFunc(GL_LESS);
  glEnable(GL_CULL_FACE);

  GLuint VertexArrayID;
  glGenVertexArrays(1, &VertexArrayID);
  glBindVertexArray(VertexArrayID);
  GLuint programID =
      LoadShaders("./Shaders/TransformVertexShader.vertexshader",
                  "./Shaders/TextureFragmentShader.fragmentshader");
  GLint MatrixID = glGetUniformLocation(programID, "MVP");

  int width, height;
  GLuint Texture =
      png_texture_load("./SpongeBob/spongebob.png", &width, &height);
  GLuint Texture1;
  glGenTextures(1, &Texture1);

  GLuint TextureID = glGetUniformLocation(programID, "myTextureSampler");
  std::vector<glm::vec3> vertices;
  std::vector<glm::vec2> uvs;
  std::vector<glm::vec3> normals;  // Won't be used at the moment.
  loadOBJ("./SpongeBob/spongebob.obj", vertices, uvs, normals);

  GLuint vertexbuffer;
  glGenBuffers(1, &vertexbuffer);
  glBindBuffer(GL_ARRAY_BUFFER, vertexbuffer);
  glBufferData(GL_ARRAY_BUFFER, vertices.size() * sizeof(glm::vec3),
               &vertices[0], GL_STATIC_DRAW);

  GLuint uvbuffer;
  glGenBuffers(1, &uvbuffer);
  glBindBuffer(GL_ARRAY_BUFFER, uvbuffer);
  glBufferData(GL_ARRAY_BUFFER, uvs.size() * sizeof(glm::vec2), &uvs[0],
               GL_STATIC_DRAW);

  static const GLfloat g_vertex_buffer_data[] = {
      1.0f, -1.0f, 0.0f, 1.0f,  1.0f, 0.0f, -1.0f, -1.0f, 0.0f,
      1.0f, 1.0f,  0.0f, -1.0f, 1.0f, 0.0f, -1.0f, -1.0f, 0.0f,
  };

  static const GLfloat g_uv_buffer_data[] = {
      1.0f, 0.0f, 1.0f, 1.0f, 0.0f, 0.0f, 1.0f, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f,
  };

  GLuint colorbuffer;
  glGenBuffers(1, &colorbuffer);
  glBindBuffer(GL_ARRAY_BUFFER, colorbuffer);
  // glBufferData(GL_ARRAY_BUFFER, cube_uvs.size() * sizeof(glm::vec2),
  // &cube_uvs[0], GL_STATIC_DRAW);
  glBufferData(GL_ARRAY_BUFFER, sizeof(g_uv_buffer_data), g_uv_buffer_data,
               GL_STATIC_DRAW);

  GLuint cubebuffer;
  glGenBuffers(1, &cubebuffer);
  glBindBuffer(GL_ARRAY_BUFFER, cubebuffer);
  glBufferData(GL_ARRAY_BUFFER, sizeof(g_vertex_buffer_data),
               g_vertex_buffer_data, GL_STATIC_DRAW);
  /////////////////////////////////working
  /// area/////////////////////////////////////////////////////////////////////////////
  //pure rendering code, consider encapsulate rendering part
  inputVideo.set(CV_CAP_PROP_FRAME_WIDTH, 640);
  inputVideo.set(CV_CAP_PROP_FRAME_HEIGHT, 480);
  cv::Mat im,imu,imcopy;
  
   cv::Mat Tcw; //camera pose
  int index = 0;
  bool onlyfirstplane = true;

  Plane* pPlane = nullptr;
  while(inputVideo.grab())
  {
    cv::Mat im;
    //grab image, consider to put the grab_image to a seprate thread
    inputVideo.retrieve(im);
    Tcw = SLAM.TrackMonocular(im,index++);
    int state = SLAM.GetTrackingState();
    vector<ORB_SLAM2::MapPoint*> vMPs = SLAM.GetTrackedMapPoints();
    vector<cv::KeyPoint> vKeys = SLAM.GetTrackedKeyPointsUn();
    cv::undistort(im,imu,K,DistCoef);

    im.copyTo(imcopy);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    glUseProgram(programID);
    //computeMatricesFromInputs();

    glActiveTexture(GL_TEXTURE1);
    Texture1 = loadframe_opencv(imcopy, Texture1);
    glBindTexture(GL_TEXTURE_2D, Texture1);

    glUniform1i(TextureID, 1);
    // get mv matrix
    glm::mat4 MVP = glm::mat4(1.0);

    glUniformMatrix4fv(MatrixID, 1, GL_FALSE, &MVP[0][0]);
    glDisable(GL_DEPTH_TEST);

    glEnableVertexAttribArray(0);
    glBindBuffer(GL_ARRAY_BUFFER, cubebuffer);
    glVertexAttribPointer(0,         // attribute
                          3,         // size
                          GL_FLOAT,  // type
                          GL_FALSE,  // normalized?
                          0,         // stride
                          (void *)0  // array buffer offset
                          );

    // 2nd attribute buffer : UVs
    glEnableVertexAttribArray(1);
    glBindBuffer(GL_ARRAY_BUFFER, colorbuffer);
    glVertexAttribPointer(1,         // attribute
                          2,         // size
                          GL_FLOAT,  // type
                          GL_FALSE,  // normalized?
                          0,         // stride
                          (void *)0  // array buffer offset
                          );
    glDrawArrays(GL_TRIANGLES, 0, 6);

    glDisableVertexAttribArray(0);
    glDisableVertexAttribArray(1);

    glfwPollEvents();
    //get View Matrix
    glm::mat4 V;

    Mat R, tvec;
    if(!Tcw.empty())
    {
      Tcw.rowRange(0,3).colRange(0,3).copyTo(R);
      Tcw.rowRange(0,3).col(3).copyTo(tvec);
      R.convertTo(R, CV_64F);
      tvec.convertTo(tvec, CV_64F);
    }


    Mat viewMatrix = cv::Mat::zeros(4, 4, CV_64FC1);
    cv::Mat cvToGl = cv::Mat::zeros(4, 4, CV_64F);
    cvToGl.at<double>(0, 0) = 1.0f;
    cvToGl.at<double>(1, 1) = -1.0f;
    cvToGl.at<double>(2, 2) = -1.0f;
    cvToGl.at<double>(3, 3) = 1.0f;

    if(!Tcw.empty()){
      for (unsigned int row = 0; row < 3; ++row) {
        for (unsigned int col = 0; col < 3; ++col) {
          viewMatrix.at<double>(row, col) = R.at<double>(row, col);
        }
        viewMatrix.at<double>(row, 3) = tvec.at<double>(row, 0);
      }
      viewMatrix.at<double>(3, 3) = 1.0f;

      viewMatrix = cvToGl * viewMatrix;
      viewMatrix.convertTo(viewMatrix, CV_32F);

      for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 4; j++) {
          V[i][j] = viewMatrix.at<float>(j, i);
        }
      }
    }
    else
    {
      for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 4; j++) {
          V[i][j] = cvToGl.at<float>(j, i);
        }
      }
    }

    float near_plane = 0.1;
    float far_plane = 100;

    //Project Matrix
    glm::mat4 ProjectionMatrix;

    float projection_matrix[16];
    projection_matrix[0] = 2*f_x/width;
    projection_matrix[1] = 0.0f;
    projection_matrix[2] = 0.0f;
    projection_matrix[3] = 0.0f;
    
    projection_matrix[4] = 0.0f;
    projection_matrix[5] = 2*f_y/height;
    projection_matrix[6] = 0.0f;
    projection_matrix[7] = 0.0f;
    
    projection_matrix[8] = 1.0f - 2*c_x/width;
    projection_matrix[9] = 2*c_y/height - 1.0f;
    projection_matrix[10] = -(far_plane + near_plane)/(far_plane - near_plane);
    projection_matrix[11] = -1.0f;
    
    projection_matrix[12] = 0.0f;
    projection_matrix[13] = 0.0f;
    projection_matrix[14] = -2.0f*far_plane*near_plane/(far_plane - near_plane);
    projection_matrix[15] = 0.0f;
    
    //ProjectionMatrix = glm::make_mat4(projection_matrix);
    ProjectionMatrix = glm::perspective(glm::radians(61.5f),4.0f / 3.0f,0.1f,1000.0f);
    //Model Matrix
    glm::mat4 ModelMatrix;
    glm::mat4 TranslateMatrix;
    glm::mat4 ScalingMatrix;
    
    if (glfwGetKey(window,GLFW_KEY_UP) == GLFW_PRESS)
        onlyfirstplane = true;
    if (onlyfirstplane)
        pPlane = DetectPlane(Tcw,vMPs,50);
    cv::Mat Tpw;

    if (pPlane)
    {
      onlyfirstplane =false;
      Tpw = pPlane->Tpw;
      for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 4; j++) {
          ModelMatrix[i][j] = Tpw.at<float>(j, i);
        }
      }
      ModelMatrix[0][3] = 0.0;
      ModelMatrix[1][3] = 0.0;
      ModelMatrix[2][3] = 0.0;
      ModelMatrix[3][3] = 1.0;
      ModelMatrix = glm::rotate(ModelMatrix, glm::radians(180.0f), glm::vec3( -1, 0, 0));

    }
    else
    {
      ModelMatrix = glm::rotate(glm::mat4(1.0), glm::radians(90.0f), glm::vec3( -1, 0, 0));
    }
    if (glfwGetKey(window,GLFW_KEY_DOWN) == GLFW_PRESS)
          scale_factor *= 2;
    //ModelMatrix = glm::rotate(glm::mat4(1.0), glm::radians(90.0f), glm::vec3( -1, 0, 0));
    //ModelMatrix = glm::rotate(ModelMatrix, glm::radians(0.0), glm::vec3( 0, 1, 0));
    TranslateMatrix = glm::translate(glm::mat4(1.0f), glm::vec3(0,0,0));
    ScalingMatrix = glm::scale(glm::mat4(1.0f), glm::vec3(scale_factor));
    ModelMatrix = ModelMatrix * ScalingMatrix;
    MVP = ProjectionMatrix * V * ModelMatrix;
    glEnable(GL_DEPTH_TEST);

    glUniformMatrix4fv(MatrixID, 1, GL_FALSE, &MVP[0][0]);

    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, Texture);

    glUniform1i(TextureID, 0);

    glEnableVertexAttribArray(0);
    glBindBuffer(GL_ARRAY_BUFFER, vertexbuffer);
    glVertexAttribPointer(0,         // attribute
                          3,         // size
                          GL_FLOAT,  // type
                          GL_FALSE,  // normalized?
                          0,         // stride
                          (void *)0  // array buffer offset
                          );

    glEnableVertexAttribArray(1);
    glBindBuffer(GL_ARRAY_BUFFER, uvbuffer);
    glVertexAttribPointer(1,         // attribute
                          2,         // size
                          GL_FLOAT,  // type
                          GL_FALSE,  // normalized?
                          0,         // stride
                          (void *)0  // array buffer offset
                          );

    glDrawArrays(GL_TRIANGLES, 0, vertices.size());

    glfwSwapBuffers(window);

    glfwPollEvents();

    //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    char key = (char)waitKey(10);
    if (key == 27) break;
    
  }
   // Stop all threads
  SLAM.Shutdown();

    // Save camera trajectory
  SLAM.SaveKeyFrameTrajectoryTUM("KeyFrameTrajectory.txt");
  
  glDeleteBuffers(1, &vertexbuffer);
  glDeleteBuffers(1, &uvbuffer);
  glDeleteProgram(programID);
  glDeleteTextures(1, &TextureID);
  glDeleteVertexArrays(1, &VertexArrayID);

  // Close OpenGL window and terminate GLFW
  glfwTerminate();
  
  return 0;
}
