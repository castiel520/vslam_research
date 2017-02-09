#include <iostream>
#include <opencv2/aruco.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/opencv.hpp>

// Include GLEW
#include <GL/glew.h>

// Include GLFW
#include <GLFW/glfw3.h>
GLFWwindow *window;

// Include GLM
#include <glm/glm.hpp>
#include <glm/gtx/transform.hpp>

using namespace glm;
using namespace std;
using namespace cv;

#include "GL_Rendering/controls.hpp"
#include "GL_Rendering/objloader.hpp"
#include "GL_Rendering/shader.hpp"
#include "GL_Rendering/texture.hpp"

namespace {
const char *about = "Basic marker detection";
const char *keys =
    "{l        | 0.1   | Marker side lenght (in meters). Needed for correct "
    "scale in camera pose }"
    "{dp       |       | File of marker detector parameters }"
    "{r        |       | show rejected candidates too }";
}

/**
 */
static bool readCameraParameters(string filename, Mat &camMatrix,
                                 Mat &distCoeffs) {
  FileStorage fs(filename, FileStorage::READ);
  if (!fs.isOpened()) return false;
  fs["camera_matrix"] >> camMatrix;
  fs["distortion_coefficients"] >> distCoeffs;
  return true;
}

/**
 */
static bool readDetectorParameters(string filename,
                                   Ptr<aruco::DetectorParameters> &params) {
  FileStorage fs(filename, FileStorage::READ);
  if (!fs.isOpened()) return false;
  fs["adaptiveThreshWinSizeMin"] >> params->adaptiveThreshWinSizeMin;
  fs["adaptiveThreshWinSizeMax"] >> params->adaptiveThreshWinSizeMax;
  fs["adaptiveThreshWinSizeStep"] >> params->adaptiveThreshWinSizeStep;
  fs["adaptiveThreshConstant"] >> params->adaptiveThreshConstant;
  fs["minMarkerPerimeterRate"] >> params->minMarkerPerimeterRate;
  fs["maxMarkerPerimeterRate"] >> params->maxMarkerPerimeterRate;
  fs["polygonalApproxAccuracyRate"] >> params->polygonalApproxAccuracyRate;
  fs["minCornerDistanceRate"] >> params->minCornerDistanceRate;
  fs["minDistanceToBorder"] >> params->minDistanceToBorder;
  fs["minMarkerDistanceRate"] >> params->minMarkerDistanceRate;
  fs["doCornerRefinement"] >> params->doCornerRefinement;
  fs["cornerRefinementWinSize"] >> params->cornerRefinementWinSize;
  fs["cornerRefinementMaxIterations"] >> params->cornerRefinementMaxIterations;
  fs["cornerRefinementMinAccuracy"] >> params->cornerRefinementMinAccuracy;
  fs["markerBorderBits"] >> params->markerBorderBits;
  fs["perspectiveRemovePixelPerCell"] >> params->perspectiveRemovePixelPerCell;
  fs["perspectiveRemoveIgnoredMarginPerCell"] >>
      params->perspectiveRemoveIgnoredMarginPerCell;
  fs["maxErroneousBitsInBorderRate"] >> params->maxErroneousBitsInBorderRate;
  fs["minOtsuStdDev"] >> params->minOtsuStdDev;
  fs["errorCorrectionRate"] >> params->errorCorrectionRate;
  return true;
}

/**
 */
int main(int argc, char *argv[]) {
  int dictionaryId = 10;
  bool showRejected = false;
  bool estimatePose = true;
  float markerLength = 0.038;

  Ptr<aruco::DetectorParameters> detectorParams =
      aruco::DetectorParameters::create();
  bool readOk = readDetectorParameters("detector_params.yml", detectorParams);
  if (!readOk) {
    cerr << "Invalid detector parameters file" << endl;
    return 0;
  }
  detectorParams->doCornerRefinement = true;  // do corner refinement in markers

  Ptr<aruco::Dictionary> dictionary = aruco::getPredefinedDictionary(
      aruco::PREDEFINED_DICTIONARY_NAME(10));  // default to DICT_6X6_250

  Mat camMatrix, distCoeffs;
  if (estimatePose) {
    bool readOk = readCameraParameters("camera_intri", camMatrix, distCoeffs);
    if (!readOk) {
      cerr << "Invalid camera file" << endl;
      return 0;
    }
  }

  VideoCapture inputVideo(0);

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
  window = glfwCreateWindow(1280, 720, "ORB AR", NULL, NULL);
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
      LoadShaders("../Shaders/TransformVertexShader.vertexshader",
                  "../Shaders/TextureFragmentShader.fragmentshader");
  GLint MatrixID = glGetUniformLocation(programID, "MVP");

  int width, height;
  GLuint Texture =
      png_texture_load("../SpongeBob/spongebob.png", &width, &height);
  GLuint Texture1;
  glGenTextures(1, &Texture1);

  GLuint TextureID = glGetUniformLocation(programID, "myTextureSampler");
  std::vector<glm::vec3> vertices;
  std::vector<glm::vec2> uvs;
  std::vector<glm::vec3> normals;  // Won't be used at the moment.
  loadOBJ("../SpongeBob/spongebob.obj", vertices, uvs, normals);

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

  inputVideo.set(CV_CAP_PROP_FRAME_WIDTH, 1280);
  inputVideo.set(CV_CAP_PROP_FRAME_HEIGHT, 720);
  while (inputVideo.grab()) {
    Mat image, imageCopy;
    inputVideo.retrieve(image);

    vector<int> ids;
    vector<vector<Point2f> > corners, rejected;
    vector<Vec3d> rvecs, tvecs;

    // detect markers and estimate pose
    aruco::detectMarkers(image, dictionary, corners, ids, detectorParams,
                         rejected);
    if (estimatePose && ids.size() > 0)
      aruco::estimatePoseSingleMarkers(corners, markerLength, camMatrix,
                                       distCoeffs, rvecs, tvecs);

    image.copyTo(imageCopy);
    if (ids.size() > 0) {
      aruco::drawDetectedMarkers(imageCopy, corners, ids);
  }
      //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
      if (rvecs.size() > 0) {
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
        glUseProgram(programID);
        computeMatricesFromInputs();

        glActiveTexture(GL_TEXTURE1);
        Texture1 = loadframe_opencv(imageCopy, Texture1);
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
        cv::Mat rot_matrix(rvecs[0]);
        cv::Mat rotation;
        cv::Rodrigues(rot_matrix, rotation);
        cv::Mat translation(tvecs[0]);

        rotation.convertTo(rotation, CV_64F);
        translation.convertTo(translation, CV_64F);

        glm::mat4 V;
        Mat viewMatrix = cv::Mat::zeros(4, 4, CV_64FC1);
        cv::Mat cvToGl = cv::Mat::zeros(4, 4, CV_64F);
        cvToGl.at<double>(0, 0) = 1.0f;
        cvToGl.at<double>(1, 1) = -1.0f;
        cvToGl.at<double>(2, 2) = -1.0f;
        cvToGl.at<double>(3, 3) = 1.0f;

        for (unsigned int row = 0; row < 3; ++row) {
          for (unsigned int col = 0; col < 3; ++col) {
            viewMatrix.at<double>(row, col) = rotation.at<double>(row, col);
          }
          viewMatrix.at<double>(row, 3) = translation.at<double>(row, 0);
        }
        viewMatrix.at<double>(3, 3) = 1.0f;

        viewMatrix = cvToGl * viewMatrix;
        viewMatrix.convertTo(viewMatrix, CV_32F);

        for (int i = 0; i < 4; i++) {
          for (int j = 0; j < 4; j++) {
            V[i][j] = viewMatrix.at<float>(j, i);
          }
        }

        computeMatricesFromInputs();
        glm::mat4 ProjectionMatrix = getProjectionMatrix();
        glm::mat4 ModelMatrix = getModelMatrix();

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
    }
    glDeleteBuffers(1, &vertexbuffer);
    glDeleteBuffers(1, &uvbuffer);
    glDeleteProgram(programID);
    glDeleteTextures(1, &TextureID);
    glDeleteVertexArrays(1, &VertexArrayID);

    // Close OpenGL window and terminate GLFW
    glfwTerminate();
    return 0;
  }
