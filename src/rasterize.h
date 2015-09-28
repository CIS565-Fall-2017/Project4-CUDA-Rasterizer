#pragma once

#include <glm/glm.hpp>

void rasterizeInit(int width, int height);
void rasterizeSet(
        int bufIdxSize, int *bufIdx,
        int vertCount, float *bufPos, float *bufNor, float *bufCol);
void rasterize(uchar4 *pbo);
void rasterizeFree();
