//author:@Baitao Shao 
//github: castiel520
//@Beijing 17/4/4

//standard includes
#include <stdio.h>
#include <string.h>
#include <ctime>
#include <chrono>
#include <thread>

#include <vector>

//opencv includes
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

//ZED Includes
#include <zed/Camera.hpp>

// Include OpenCV
#include <opencv2/opencv.hpp>

// Include GLEW
#include <GL/glew.h>

// Include GLFW
#include <GLFW/glfw3.h>
GLFWwindow* window;

#define GLM_FORCE_RADIANS
// Include GLM
#include <glm/glm.hpp>
#include <glm/gtx/transform.hpp>
#include <glm/gtc/type_ptr.hpp>
using namespace glm;
using namespace cv;

#include "shader.hpp"
#include "texture.hpp"
#include "controls.hpp"
#include "objloader.hpp"
#include "orb_slam.h"
#include "planar_tracking.h"

using namespace sl::zed;
using namespace std;

#include <iostream>
#include <algorithm>
#include <fstream>
#include <chrono>
#include <iomanip>
#include <math.h>
#include <fstream>
#include <map>
#include <string>
#include <vector>

#include "ORB_SLAM/System.h"

// include DevIL for image loading
#include <IL/il.h>

// assimp include files, for model loading
#include "assimp/Importer.hpp" 
#include "assimp/postprocess.h"
#include "assimp/scene.h"

const int FPS = 30;
const ZEDResolution_mode ZED_RES = VGA;//zed camera options are VGA, HD720, HD2K

glm::mat4 MVP;
Camera* zed;
VideoCapture cap_left(1);
cv::Mat Left;
cv::Mat Right;
int width, height;
bool stop_signal;

// Information to render each assimp node
struct MyMesh{

    GLuint vao;
    GLuint texIndex;
    GLuint uniformBlockIndex;
    int numFaces;
};

std::vector<struct MyMesh> myMeshes;

// This is for a shader uniform block
struct MyMaterial{

    float diffuse[4];
    float ambient[4];
    float specular[4];
    float emissive[4];
    float shininess;
    int texCount;
};

// Vertex Attribute Locations
GLuint vertexLoc=0, texCoordLoc=1;

// Uniform Bindings Points
GLuint matricesUniLoc = 1, materialUniLoc = 2;

// Create an instance of the Importer class
Assimp::Importer importer;

// the global Assimp scene object
const aiScene* scene = NULL;

std::map<std::string, GLuint> textureIdMap; 

bool enable_zed = true;
// Replace the model name by your model's filename
static const std::string modelname = "porsche-911.obj";

void grab_run()
{
    while (!stop_signal)
    {
        bool res = zed->grab(SENSING_MODE::STANDARD, 1, 1);

        if (!res)
        {
            slMat2cvMat(zed->retrieveImage(SIDE::LEFT)).copyTo(Left);
            //slMat2cvMat(zed->normalizeMeasure(MEASURE::DEPTH)).copyTo(Depth);
            slMat2cvMat(zed->retrieveImage(SIDE::RIGHT)).copyTo(Right);
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    delete zed;
}

bool Import3DFromFile( const std::string& pFile)
{

    //check if file exists
    std::ifstream fin(pFile.c_str());
    if(!fin.fail()) {
        fin.close();
    }
    else{
        printf("Couldn't open file: %s\n", pFile.c_str());
        printf("%s\n", importer.GetErrorString());
        return false;
    }

    scene = importer.ReadFile( pFile, aiProcessPreset_TargetRealtime_Quality);

    // If the import failed, report it
    if( !scene)
    {
        printf("%s\n", importer.GetErrorString());
        return false;
    }

    // Now we can access the file's contents.
    printf("Import of scene %s succeeded.",pFile.c_str());
    return true;
}


int LoadGLTextures(const aiScene* scene)
{
    ILboolean success;

    /* initialization of DevIL */
    ilInit(); 

    /* scan scene's materials for textures */
    for (unsigned int m=0; m<scene->mNumMaterials; ++m)
    {
        int texIndex = 0;
        aiString path;  // filename

        aiReturn texFound = scene->mMaterials[m]->GetTexture(aiTextureType_DIFFUSE, texIndex, &path);
        while (texFound == AI_SUCCESS) {
            //fill map with textures, OpenGL image ids set to 0
            textureIdMap[path.data] = 0; 
            // more textures?
            texIndex++;
            texFound = scene->mMaterials[m]->GetTexture(aiTextureType_DIFFUSE, texIndex, &path);
        }
    }

    int numTextures = textureIdMap.size();

    /* create and fill array with DevIL texture ids */
    ILuint* imageIds = new ILuint[numTextures];
    ilGenImages(numTextures, imageIds); 

    /* create and fill array with GL texture ids */
    GLuint* textureIds = new GLuint[numTextures];
    glGenTextures(numTextures, textureIds); /* Texture name generation */

    /* get iterator */
    std::map<std::string, GLuint>::iterator itr = textureIdMap.begin();
    int i=0;
    for (; itr != textureIdMap.end(); ++i, ++itr)
    {
        //save IL image ID
        std::string filename = (*itr).first;  // get filename
        (*itr).second = textureIds[i];    // save texture id for filename in map

        ilBindImage(imageIds[i]); /* Binding of DevIL image name */
        ilEnable(IL_ORIGIN_SET);
        ilOriginFunc(IL_ORIGIN_LOWER_LEFT); 
        success = ilLoadImage((ILstring)filename.c_str());

        if (success) {
            /* Convert image to RGBA */
            ilConvertImage(IL_RGBA, IL_UNSIGNED_BYTE); 

            /* Create and load textures to OpenGL */
            glBindTexture(GL_TEXTURE_2D, textureIds[i]); 
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR); 
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
            glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, ilGetInteger(IL_IMAGE_WIDTH),
                ilGetInteger(IL_IMAGE_HEIGHT), 0, GL_RGBA, GL_UNSIGNED_BYTE,
                ilGetData()); 
        }
        else 
            printf("Couldn't load Image: %s\n", filename.c_str());
    }
    /* Because we have already copied image data into texture data
    we can release memory used by image. */
    ilDeleteImages(numTextures, imageIds); 

    //Cleanup
    delete [] imageIds;
    delete [] textureIds;

    //return success;
    return true;
}

void set_float4(float f[4], float a, float b, float c, float d)
{
    f[0] = a;
    f[1] = b;
    f[2] = c;
    f[3] = d;
}

void color4_to_float4(const aiColor4D *c, float f[4])
{
    f[0] = c->r;
    f[1] = c->g;
    f[2] = c->b;
    f[3] = c->a;
}



void genVAOsAndUniformBuffer(const aiScene *sc) {

    struct MyMesh aMesh;
    struct MyMaterial aMat; 
    GLuint buffer;
    
    // For each mesh
    for (unsigned int n = 0; n < sc->mNumMeshes; ++n)
    {
        const aiMesh* mesh = sc->mMeshes[n];

        // create array with faces
        // have to convert from Assimp format to array
        unsigned int *faceArray;
        faceArray = (unsigned int *)malloc(sizeof(unsigned int) * mesh->mNumFaces * 3);
        unsigned int faceIndex = 0;

        for (unsigned int t = 0; t < mesh->mNumFaces; ++t) {
            const aiFace* face = &mesh->mFaces[t];

            memcpy(&faceArray[faceIndex], face->mIndices,3 * sizeof(unsigned int));
            faceIndex += 3;
        }
        aMesh.numFaces = sc->mMeshes[n]->mNumFaces;
        // generate Vertex Array for mesh
        glGenVertexArrays(1,&(aMesh.vao));
        glBindVertexArray(aMesh.vao);

        // buffer for faces
        glGenBuffers(1, &buffer);
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, buffer);
        glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(unsigned int) * mesh->mNumFaces * 3, faceArray, GL_STATIC_DRAW);

        // buffer for vertex positions
        if (mesh->HasPositions()) {
            glGenBuffers(1, &buffer);
            glBindBuffer(GL_ARRAY_BUFFER, buffer);
            glBufferData(GL_ARRAY_BUFFER, sizeof(float)*3*mesh->mNumVertices, mesh->mVertices, GL_STATIC_DRAW);
            glEnableVertexAttribArray(vertexLoc);
            glVertexAttribPointer(vertexLoc, 3, GL_FLOAT, 0, 0, 0);
        }
    
        // buffer for vertex texture coordinates
        if (mesh->HasTextureCoords(0)) {
            float *texCoords = (float *)malloc(sizeof(float)*2*mesh->mNumVertices);
            for (unsigned int k = 0; k < mesh->mNumVertices; ++k) {

                texCoords[k*2]   = mesh->mTextureCoords[0][k].x;
                texCoords[k*2+1] = mesh->mTextureCoords[0][k].y; 
                
            }
            glGenBuffers(1, &buffer);
            glBindBuffer(GL_ARRAY_BUFFER, buffer);
            glBufferData(GL_ARRAY_BUFFER, sizeof(float)*2*mesh->mNumVertices, texCoords, GL_STATIC_DRAW);
            glEnableVertexAttribArray(texCoordLoc);
            glVertexAttribPointer(texCoordLoc, 2, GL_FLOAT, 0, 0, 0);
        }

        // unbind buffers
        glBindVertexArray(0);
        glBindBuffer(GL_ARRAY_BUFFER,0);
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER,0);
    
        // create material uniform buffer
        aiMaterial *mtl = sc->mMaterials[mesh->mMaterialIndex];
            
        aiString texPath;   //contains filename of texture
        if(AI_SUCCESS == mtl->GetTexture(aiTextureType_DIFFUSE, 0, &texPath)){
                //bind texture
                unsigned int texId = textureIdMap[texPath.data];
                aMesh.texIndex = texId;
                aMat.texCount = 1;
            }
        else
            aMat.texCount = 0;

        float c[4];
        set_float4(c, 0.8f, 0.8f, 0.8f, 1.0f);
        aiColor4D diffuse;
        if(AI_SUCCESS == aiGetMaterialColor(mtl, AI_MATKEY_COLOR_DIFFUSE, &diffuse))
            color4_to_float4(&diffuse, c);
        memcpy(aMat.diffuse, c, sizeof(c));

        set_float4(c, 0.2f, 0.2f, 0.2f, 1.0f);
        aiColor4D ambient;
        if(AI_SUCCESS == aiGetMaterialColor(mtl, AI_MATKEY_COLOR_AMBIENT, &ambient))
            color4_to_float4(&ambient, c);
        memcpy(aMat.ambient, c, sizeof(c));

        set_float4(c, 0.0f, 0.0f, 0.0f, 1.0f);
        aiColor4D specular;
        if(AI_SUCCESS == aiGetMaterialColor(mtl, AI_MATKEY_COLOR_SPECULAR, &specular))
            color4_to_float4(&specular, c);
        memcpy(aMat.specular, c, sizeof(c));

        set_float4(c, 0.0f, 0.0f, 0.0f, 1.0f);
        aiColor4D emission;
        if(AI_SUCCESS == aiGetMaterialColor(mtl, AI_MATKEY_COLOR_EMISSIVE, &emission))
            color4_to_float4(&emission, c);
        memcpy(aMat.emissive, c, sizeof(c));

        float shininess = 0.0;
        unsigned int max;
        aiGetMaterialFloatArray(mtl, AI_MATKEY_SHININESS, &shininess, &max);
        aMat.shininess = shininess;

        glGenBuffers(1,&(aMesh.uniformBlockIndex));
        glBindBuffer(GL_UNIFORM_BUFFER,aMesh.uniformBlockIndex);
        glBufferData(GL_UNIFORM_BUFFER, sizeof(aMat), (void *)(&aMat), GL_STATIC_DRAW);

        myMeshes.push_back(aMesh);
    }
}

void recursive_render(const aiScene *sc, const aiNode* nd,GLint matrixid)
{
    aiMatrix4x4 m = nd->mTransformation;
    // OpenGL matrices are column major
    m.Transpose();

    float trans[16];
    memcpy(trans,&m,sizeof(float) * 16);
    
    glm::mat4 Trans = glm::make_mat4(trans);
    Trans = MVP * Trans;
    glUniformMatrix4fv(matrixid, 1, GL_FALSE, &MVP[0][0]);
    // draw all meshes assigned to this node
    
    for (unsigned int n=0; n < nd->mNumMeshes; ++n){
        // bind material uniform
        glBindBufferRange(GL_UNIFORM_BUFFER, materialUniLoc, myMeshes[nd->mMeshes[n]].uniformBlockIndex, 0, sizeof(struct MyMaterial)); 
        // bind texture
        glBindTexture(GL_TEXTURE_2D, myMeshes[nd->mMeshes[n]].texIndex);
        // bind VAO
        glBindVertexArray(myMeshes[nd->mMeshes[n]].vao);
        // draw
        glDrawElements(GL_TRIANGLES,myMeshes[nd->mMeshes[n]].numFaces*3,GL_UNSIGNED_INT,0);
    }

    // draw all children
    for (unsigned int n=0; n < nd->mNumChildren; ++n){
        recursive_render(sc, nd->mChildren[n],matrixid);
    }
}

int main(int argc, char **argv)
{
    if (argc > 1)
    {
      if (std::string(argv[1]) == "-zed")
      {
        enable_zed = true;
      }        
    }

    
    //if (enable_zed)
    //{
      zed = new Camera(ZED_RES, FPS);
      InitParams parameters;
      parameters.mode = QUALITY;
      parameters.unit = MILLIMETER;
      parameters.verbose = 1;

      ERRCODE err = zed->init(parameters);

      width = zed->getImageSize().width;
      height = zed->getImageSize().height;
      Left = cv::Mat(height, width, CV_8UC4, 1);
      Right = cv::Mat(height, width, CV_8UC4, 1);

      std::thread grab_thread(grab_run); // creates thread of execution
      /*
    }
    else
    {
      cap_left.set(CV_CAP_PROP_FRAME_WIDTH, 672);
      cap_left.set(CV_CAP_PROP_FRAME_HEIGHT, 376);
    }
    */

    // Initialise Tracking System
    bool success = initTracking("../Extrinsics.xml");
    cv::Mat K = getCameraMatrix();

    //ORB_SLAM2::System SLAM("../Vocabulary/ORBvoc.txt","../TUM1.yaml",ORB_SLAM2::System::MONOCULAR,false);
    //if (enable_zed)
    //{
      // Create SLAM system. It initializes all system threads and gets ready to process frames.
      ORB_SLAM2::System SLAM("../Vocabulary/ORBvoc.txt","../zed.yaml",ORB_SLAM2::System::STEREO,false);
    //}
    

    if (!success)
        return 0;

    // Initialise GLFW
    if( !glfwInit() )
    {
        fprintf( stderr, "Failed to initialize GLFW\n" );
        getchar();
        return -1;
    }

    glfwWindowHint(GLFW_SAMPLES, 4);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE); // To make MacOS happy; should not be needed
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

    // Open a window and create its OpenGL context
    window = glfwCreateWindow( 672, 376, "SLAM-AR", NULL, NULL);
    if( window == NULL ){
        fprintf( stderr, "Failed to open GLFW window. If you have an Intel GPU, they are not 3.3 compatible. Try the 2.1 version of the tutorials.\n" );
        getchar();
        glfwTerminate();
        return -1;
    }
    glfwMakeContextCurrent(window);
    printf("OpenGL version supported by this platform (%s): \n", glGetString(GL_VERSION));

    // Initialize GLEW
    glewExperimental = GL_TRUE; // Needed for core profile
    if (glewInit() != GLEW_OK) {
        fprintf(stderr, "Failed to initialize GLEW\n");
        getchar();
        glfwTerminate();
        return -1;
    }

    // Ensure we can capture the escape key being pressed below
    glfwSetInputMode(window, GLFW_STICKY_KEYS, GL_TRUE);

    // Dark blue background
    glClearColor(0.0f, 0.0f, 0.4f, 0.0f);

    // Enable depth test
    glEnable(GL_DEPTH_TEST);

    // Accept fragment if it closer to the camera than the former one
    glDepthFunc(GL_LESS);

    // Cull triangles which normal is not towards the camera
    glEnable(GL_CULL_FACE);
    glEnable(GL_MULTISAMPLE);
    GLuint VertexArrayID;

    
    // Create and compile our GLSL program from the shaders
    GLuint programID = LoadShaders( "../Shaders/TransformVertexShader.vertexshader", "../Shaders/TextureFragmentShader.fragmentshader" );

    // Get a handle for our "MVP" uniform
    GLint MatrixID = glGetUniformLocation(programID, "MVP");
    
    // Load the texture
    int width, height;
    GLuint Texture1;
    glGenTextures(1, &Texture1);
    

    // Get a handle for our "myTextureSampler" uniform
    GLuint TextureID  = glGetUniformLocation(programID, "myTextureSampler");

    static const GLfloat g_vertex_buffer_data[] = {
            1.0f, -1.0f, 0.0f,
            1.0f, 1.0f, 0.0f,
            -1.0f,  -1.0f, 0.0f,
            1.0f, 1.0f, 0.0f,
            -1.0f,  1.0f, 0.0f,
            -1.0f,  -1.0f, 0.0f,
    };

    static const GLfloat g_uv_buffer_data[] = {
            1.0f, 0.0f,
            1.0f, 1.0f,
            0.0f,  0.0f,
            1.0f, 1.0f,
            0.0f,  1.0f,
            0.0f,  0.0f,
    };

    GLuint colorbuffer;
    glGenBuffers(1, &colorbuffer);
    glBindBuffer(GL_ARRAY_BUFFER, colorbuffer);
    glBufferData(GL_ARRAY_BUFFER, sizeof(g_uv_buffer_data), g_uv_buffer_data, GL_STATIC_DRAW);

    GLuint cubebuffer;
    glGenBuffers(1, &cubebuffer);
    glBindBuffer(GL_ARRAY_BUFFER, cubebuffer);
    glBufferData(GL_ARRAY_BUFFER, sizeof(g_vertex_buffer_data), g_vertex_buffer_data, GL_STATIC_DRAW);
    
    bool slamMode = 0;

    Ptr<ORB> orb = ORB::create();
    orb->setScoreType(cv::ORB::FAST_SCORE);
    orb->setMaxFeatures(1000);
    Ptr<DescriptorMatcher> matcher = DescriptorMatcher::create("BruteForce-Hamming(2)");
    Tracker orb_tracker(orb, matcher, K);
    orb_tracker.setFirstFrame("../template.png");

    cv::Mat frame_left;
    //cv::Mat frame_depth;
    cv::Mat frame_right;
    bool good_before = false;
    int count = 0;

    if (!Import3DFromFile(modelname)) 
        return(0);

    LoadGLTextures(scene);
    genVAOsAndUniformBuffer(scene);     
    glClearColor(1.0f, 1.0f, 1.0f, 0.0f);
    do{
        cvtColor(Left, frame_left, CV_BGRA2BGR);
        cvtColor(Right, frame_right, CV_BGRA2BGR);
        
        glGenVertexArrays(1, &VertexArrayID);
        glBindVertexArray(VertexArrayID);

        // Clear the screen
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        // Use our shader
        glUseProgram(programID);
        
        // Bind our texture in Texture Unit 0
        glActiveTexture(GL_TEXTURE0);
        Texture1 = loadframe_opencv(frame_left, Texture1);
        glBindTexture(GL_TEXTURE_2D, Texture1);


        // Set our "myTextureSampler" sampler to user Texture Unit 0
        glUniform1i(TextureID, 0);
        MVP = glm::mat4(1.0);
        glUniformMatrix4fv(MatrixID, 1, GL_FALSE, &MVP[0][0]);

        glDisable(GL_DEPTH_TEST);

        // 1rst attribute buffer : vertices
        glEnableVertexAttribArray(0);
        glBindBuffer(GL_ARRAY_BUFFER, cubebuffer);
        glVertexAttribPointer(
                0,                  // attribute
                3,                  // size
                GL_FLOAT,           // type
                GL_FALSE,           // normalized?
                0,                  // stride
                (void*)0            // array buffer offset
        );

        // 2nd attribute buffer : UVs
        glEnableVertexAttribArray(1);
        glBindBuffer(GL_ARRAY_BUFFER, colorbuffer);
        glVertexAttribPointer(
                1,                                // attribute
                2,                                // size
                GL_FLOAT,                         // type
                GL_FALSE,                         // normalized?
                0,                                // stride
                (void*)0                          // array buffer offset
        );
        glDrawArrays(GL_TRIANGLES, 0, 6 );
        glDisableVertexAttribArray(0);
        glDisableVertexAttribArray(1);

        glBindVertexArray(0);
        glBindBuffer(GL_ARRAY_BUFFER,0);
        
        glfwPollEvents();
        if (glfwGetKey(window, GLFW_KEY_SPACE) == GLFW_PRESS)
        {
            slamMode = true;
        }
        if (count % 2 == 0)
        {
            success = orb_tracker.process(frame_left, slamMode);
        }
        
        if (!success){
            glfwPollEvents();
            if (glfwGetKey(window, GLFW_KEY_ESCAPE ) != GLFW_PRESS &&
                glfwWindowShouldClose(window) == 0){
                glfwSwapBuffers(window);
                continue;
            } else
                break;
        }
        

        glm::mat4 ViewMatrix;
        if (!slamMode)
        {
            ViewMatrix = getViewMatrix(slamMode);
        }
        else
        {
          if (enable_zed)
          {
            cv::Mat CameraPose = SLAM.TrackStereo(frame_left, frame_right, 1);
            if(!CameraPose.empty())
            {
                trackStereo(CameraPose);
                ViewMatrix = getViewMatrix(slamMode);
            }
          }
          else
          {
            cv::Mat CameraPose = SLAM.TrackMonocular(frame_left, 1);
            if(!CameraPose.empty())
            {
              trackStereo(CameraPose);
              ViewMatrix = getViewMatrix(slamMode);
            }
          }
        }

        // Compute the MVP matrix from keyboard and mouse input
        computeMatricesFromInputs();
        
        glm::mat4 ProjectionMatrix = getProjectionMatrix();

        glm::mat4 ModelMatrix = getModelMatrix();

        glm::mat4 initModelMatrix = orb_tracker.getInitModelMatrix();

        MVP = ProjectionMatrix * ViewMatrix * initModelMatrix * ModelMatrix;

        glEnable(GL_DEPTH_TEST);

        glUniform1i(TextureID,0);
        recursive_render(scene, scene->mRootNode,MatrixID);
        // Swap buffers
        glfwSwapBuffers(window);

        glfwPollEvents();
        if (glfwGetKey(window, GLFW_KEY_R) == GLFW_PRESS)
        {
            slamMode = false;
            SLAM.Reset();
        }

        glfwPollEvents();

    } // Check if the ESC key was pressed or the window was closed
    while( glfwGetKey(window, GLFW_KEY_ESCAPE ) != GLFW_PRESS &&
           glfwWindowShouldClose(window) == 0 );

    stop_signal = true;
    grab_thread.join();

    // Cleanup VBO and shader
    glDeleteProgram(programID);
    glDeleteTextures(1, &TextureID);
        // cleaning up
    textureIdMap.clear();  

    // clear myMeshes stuff
    for (unsigned int i = 0; i < myMeshes.size(); ++i) {
            
        glDeleteVertexArrays(1,&(myMeshes[i].vao));
        glDeleteTextures(1,&(myMeshes[i].texIndex));
        glDeleteBuffers(1,&(myMeshes[i].uniformBlockIndex));
    }
    // Close OpenGL window and terminate GLFW
    glfwTerminate();

    return 0;
}

