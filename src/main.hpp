/**
 * @file      main.hpp
 * @brief     Main file for CUDA rasterizer. Handles CUDA-GL interop for display.
 * @authors   Skeleton code: Yining Karl Li, Kai Ninomiya
 * @date      2012-2015
 * @copyright University of Pennsylvania
 */

#pragma once

#include <GL/glew.h>
#include <GLFW/glfw3.h>

#include <cstdlib>
#include <ctime>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <cuda_gl_interop.h>
#include <cuda_runtime.h>
#include <glm/glm.hpp>
#include <util/glslUtility.hpp>
#include <util/utilityCore.hpp>
#include <util/objloader.hpp>
#include "rasterize.h"

using namespace std;

//-------------------------------
//------------GL STUFF-----------
//-------------------------------
int frame;
int fpstracker;
double seconds;
int fps = 0;
GLuint positionLocation = 0;
GLuint texcoordsLocation = 1;
const char *attributeLocations[] = { "Position", "Tex" };
GLuint pbo = (GLuint)NULL;
GLuint displayImage;
uchar4 *dptr;

GLFWwindow *window;

//-------------------------------
//----------CUDA STUFF-----------
//-------------------------------

int width = 800;
int height = 800;

//-------------------------------
//-------------MAIN--------------
//-------------------------------

int main(int argc, char **argv);

//-------------------------------
//---------RUNTIME STUFF---------
//-------------------------------

void runCuda();

#ifdef __APPLE__
void display();
#else
void display();
void keyboard(unsigned char key, int x, int y);
#endif

//-------------------------------
//----------SETUP STUFF----------
//-------------------------------
bool init(obj *mesh);
void initPBO();
void initCuda();
void initTextures();
void initVAO();
GLuint initShader();

//-------------------------------
//---------CLEANUP STUFF---------
//-------------------------------

void cleanupCuda();
void deletePBO(GLuint *pbo);
void deleteTexture(GLuint *tex);

//------------------------------
//-------GLFW CALLBACKS---------
//------------------------------
void mainLoop();
void errorCallback(int error, const char *description);
void keyCallback(GLFWwindow *window, int key, int scancode, int action, int mods);
