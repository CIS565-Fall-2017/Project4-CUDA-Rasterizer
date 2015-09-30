#pragma once

#include <glm/glm.hpp>

void rasterizeInit(int width, int height);
void rasterizeSetBuffers(
        int bufIdxSize, int *bufIdx,
        int vertCount, float *bufPos, float *bufNor, float *bufCol);
void rasterize(uchar4 *pbo);
void rasterizeFree();
