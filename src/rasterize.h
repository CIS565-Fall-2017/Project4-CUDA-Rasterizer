/**
 * @file      rasterize.h
 * @brief     CUDA-accelerated rasterization pipeline.
 * @authors   Skeleton code: Yining Karl Li, Kai Ninomiya, Shuai Shao (Shrek)
 * @date      2012-2016
 * @copyright University of Pennsylvania & STUDENT
 */

#pragma once

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtx/transform.hpp>

namespace tinygltf{
	class Scene;
}


void rasterizeInit(int width, int height);
void rasterizeSetBuffers(const tinygltf::Scene & scene);

void rasterize(uchar4 *pbo);
void rasterizeFree();
