#pragma once

#include <glm/glm.hpp>

void kernelCleanup();
void cudaRasterizeCore(uchar4* pos, glm::vec2 resolution, float frame, float* vbo, int vbosize, float* cbo, int cbosize, int* ibo, int ibosize);
