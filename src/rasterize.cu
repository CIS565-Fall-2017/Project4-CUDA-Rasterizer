/**
 * @file      rasterize.cu
 * @brief     CUDA-accelerated rasterization pipeline.
 * @authors   Skeleton code: Yining Karl Li, Kai Ninomiya, Shuai Shao (Shrek)
 * @date      2012-2015
 * @copyright University of Pennsylvania & STUDENT
 */

#include "rasterize.h"

#include <cmath>
#include <cstdio>
#include <cuda.h>
#include <thrust/random.h>
#include <util/checkCUDAError.h>
#include "rasterizeTools.h"

struct VertexIn {
    glm::vec3 pos;
    glm::vec3 nor;
    glm::vec3 col;
    // TODO (optional) add other vertex attributes (e.g. texture coordinates)
};
struct VertexOut {
    // TODO
};


typedef glm::vec3 VertexAttributePosition;
typedef glm::vec3 VertexAttributeNormal;
typedef glm::vec2 VertexAttributeTexcoord;

// VertexIn
static int* dev_indices;

static VertexAttributePosition* dev_vertexAttributePosition = NULL;
static VertexAttributeNormal* dev_vertexAttributeNormal = NULL;
static VertexAttributeTexcoord* dev_vertexAttributeTexcoord0 = NULL;




enum PrimitiveType{
	Triangle,
	Line,
	Point
};

struct Primitive {
	PrimitiveType primitiveType = Triangle;	// C++ 11 init
    VertexOut v[3];
};
struct Fragment {
    glm::vec3 color;
};

static int width = 0;
static int height = 0;
static int *dev_bufIdx = NULL;
//static VertexIn *dev_bufVertex = NULL;
static Primitive *dev_primitives = NULL;
static Fragment *dev_depthbuffer = NULL;
static glm::vec3 *dev_framebuffer = NULL;
static int bufIdxSize = 0;
static int vertCount = 0;

/**
 * Kernel that writes the image to the OpenGL PBO directly.
 */
__global__
void sendImageToPBO(uchar4 *pbo, int w, int h, glm::vec3 *image) {
    int x = (blockIdx.x * blockDim.x) + threadIdx.x;
    int y = (blockIdx.y * blockDim.y) + threadIdx.y;
    int index = x + (y * w);

    if (x < w && y < h) {
        glm::vec3 color;
        color.x = glm::clamp(image[index].x, 0.0f, 1.0f) * 255.0;
        color.y = glm::clamp(image[index].y, 0.0f, 1.0f) * 255.0;
        color.z = glm::clamp(image[index].z, 0.0f, 1.0f) * 255.0;
        // Each thread writes one pixel location in the texture (textel)
        pbo[index].w = 0;
        pbo[index].x = color.x;
        pbo[index].y = color.y;
        pbo[index].z = color.z;
    }
}

// Writes fragment colors to the framebuffer
__global__
void render(int w, int h, Fragment *depthbuffer, glm::vec3 *framebuffer) {
    int x = (blockIdx.x * blockDim.x) + threadIdx.x;
    int y = (blockIdx.y * blockDim.y) + threadIdx.y;
    int index = x + (y * w);

    if (x < w && y < h) {
        framebuffer[index] = depthbuffer[index].color;
    }
}

/**
 * Called once at the beginning of the program to allocate memory.
 */
void rasterizeInit(int w, int h) {
    width = w;
    height = h;
    cudaFree(dev_depthbuffer);
    cudaMalloc(&dev_depthbuffer,   width * height * sizeof(Fragment));
    cudaMemset(dev_depthbuffer, 0, width * height * sizeof(Fragment));
    cudaFree(dev_framebuffer);
    cudaMalloc(&dev_framebuffer,   width * height * sizeof(glm::vec3));
    cudaMemset(dev_framebuffer, 0, width * height * sizeof(glm::vec3));
    checkCUDAError("rasterizeInit");
}

/**
 * Set all of the buffers necessary for rasterization.
 * Note: store bufferView as char* in device memory.
 */
//void rasterizeSetBuffers(
//        int _bufIdxSize, int *bufIdx,
//        int _vertCount, float *bufPos, float *bufNor, float *bufCol) {
//	bufIdxSize = _bufIdxSize;
//	vertCount = _vertCount;
//
//	cudaFree(dev_bufIdx);
//	cudaMalloc(&dev_bufIdx, bufIdxSize * sizeof(int));
//	cudaMemcpy(dev_bufIdx, bufIdx, bufIdxSize * sizeof(int), cudaMemcpyHostToDevice);
//
//	VertexIn *bufVertex = new VertexIn[_vertCount];
//	for (int i = 0; i < vertCount; i++) {
//		int j = i * 3;
//		bufVertex[i].pos = glm::vec3(bufPos[j + 0], bufPos[j + 1], bufPos[j + 2]);
//		bufVertex[i].nor = glm::vec3(bufNor[j + 0], bufNor[j + 1], bufNor[j + 2]);
//		bufVertex[i].col = glm::vec3(bufCol[j + 0], bufCol[j + 1], bufCol[j + 2]);
//	}
//	cudaFree(dev_bufVertex);
//	cudaMalloc(&dev_bufVertex, vertCount * sizeof(VertexIn));
//	cudaMemcpy(dev_bufVertex, bufVertex, vertCount * sizeof(VertexIn), cudaMemcpyHostToDevice);
//
//	cudaFree(dev_primitives);
//	cudaMalloc(&dev_primitives, vertCount / 3 * sizeof(Triangle));
//	cudaMemset(dev_primitives, 0, vertCount / 3 * sizeof(Triangle));
//
//	checkCUDAError("rasterizeSetBuffers");
//}


static std::map<std::string, char*> bufferViewDevPointers;

static std::map<std::string, VertexAttributePosition*> mapBufferView2VertexAttributePosition;

static std::map<std::string, Primitive*> primitiveDevPointers;


// Buffer State

// Attribute State (bufferview pointer, byte offset, byte stride, count(vec2/vec3), primitive type)

void rasterizeSetBuffers(const tinygltf::Scene & scene) {

	// 1. copy all `bufferViews` to device memory
	{
		std::map<std::string, tinygltf::BufferView>::const_iterator it(
			scene.bufferViews.begin());
		std::map<std::string, tinygltf::BufferView>::const_iterator itEnd(
			scene.bufferViews.end());

		

		for (; it != itEnd; it++) {
			const tinygltf::BufferView &bufferView = it->second;
			if (bufferView.target == 0) {
				continue; // Unsupported bufferView.
			}

			const tinygltf::Buffer &buffer = scene.buffers[bufferView.buffer];

			
			char* dev_bufferView;
			cudaMalloc(&dev_bufferView, bufferView.byteLength);
			cudaMemcpy(dev_bufferView, &buffer.data.front() + bufferView.byteOffset, bufferView.byteLength, cudaMemcpyHostToDevice);

			bufferViewDevPointers.insert(std::pair<std::string, char*>(bufferView.name, dev_bufferView));

			
			//GLBufferState state;
			//glGenBuffers(1, &state.vb);
			//glBindBuffer(bufferView.target, state.vb);
			//glBufferData(bufferView.target, bufferView.byteLength,
			//	&buffer.data.at(0) + bufferView.byteOffset, GL_STATIC_DRAW);
			//glBindBuffer(bufferView.target, 0);

			//gBufferState[it->first] = state;
		}
	}



	// 2. meshes: indices, attributes
	{
		std::map<std::string, tinygltf::Mesh>::const_iterator it(scene.meshes.begin());
		std::map<std::string, tinygltf::Mesh>::const_iterator itEnd(scene.meshes.end());

		for (; it != itEnd; it++) {
			const tinygltf::Mesh & mesh = it->second;

			for (size_t i = 0; i < mesh.primitives.size(); i++) {
				const tinygltf::Primitive &primitive = mesh.primitives[i];

				if (primitive.indices.empty())
					return;

				std::map<std::string, std::string>::const_iterator it(
					primitive.attributes.begin());
				std::map<std::string, std::string>::const_iterator itEnd(
					primitive.attributes.end());

				// Assume TEXTURE_2D target for the texture object.
				//glBindTexture(GL_TEXTURE_2D, gMeshState[mesh.name].diffuseTex[i]);

				for (; it != itEnd; it++) {
					const tinygltf::Accessor &accessor = scene.accessors[it->second];
					//glBindBuffer(GL_ARRAY_BUFFER, gBufferState[accessor.bufferView].vb);
					//CheckErrors("bind buffer");
					int count = 1;
					if (accessor.type == TINYGLTF_TYPE_SCALAR) {
						count = 1;
					}
					else if (accessor.type == TINYGLTF_TYPE_VEC2) {
						count = 2;
					}
					else if (accessor.type == TINYGLTF_TYPE_VEC3) {
						count = 3;
					}
					else if (accessor.type == TINYGLTF_TYPE_VEC4) {
						count = 4;
					}
					// it->first would be "POSITION", "NORMAL", "TEXCOORD_0", ...
					if ((it->first.compare("POSITION") == 0) ||
						(it->first.compare("NORMAL") == 0) ||
						(it->first.compare("TEXCOORD_0") == 0)) {


						glVertexAttribPointer(
							gGLProgramState.attribs[it->first], count, accessor.componentType,
							GL_FALSE, accessor.byteStride, BUFFER_OFFSET(accessor.byteOffset));
						CheckErrors("vertex attrib pointer");
						glEnableVertexAttribArray(gGLProgramState.attribs[it->first]);
						CheckErrors("enable vertex attrib array");
					}
				}

				const tinygltf::Accessor &indexAccessor = scene.accessors[primitive.indices];
				glBindBuffer(GL_ELEMENT_ARRAY_BUFFER,
					gBufferState[indexAccessor.bufferView].vb);
				CheckErrors("bind buffer");
				int mode = -1;
				if (primitive.mode == TINYGLTF_MODE_TRIANGLES) {
					mode = GL_TRIANGLES;
				}
				else if (primitive.mode == TINYGLTF_MODE_TRIANGLE_STRIP) {
					mode = GL_TRIANGLE_STRIP;
				}
				else if (primitive.mode == TINYGLTF_MODE_TRIANGLE_FAN) {
					mode = GL_TRIANGLE_FAN;
				}
				else if (primitive.mode == TINYGLTF_MODE_POINTS) {
					mode = GL_POINTS;
				}
				else if (primitive.mode == TINYGLTF_MODE_LINE) {
					mode = GL_LINES;
				}
				else if (primitive.mode == TINYGLTF_MODE_LINE_LOOP) {
					mode = GL_LINE_LOOP;
				};
				glDrawElements(mode, indexAccessor.count, indexAccessor.componentType,
					BUFFER_OFFSET(indexAccessor.byteOffset));
				CheckErrors("draw elements");

				{
					std::map<std::string, std::string>::const_iterator it(
						primitive.attributes.begin());
					std::map<std::string, std::string>::const_iterator itEnd(
						primitive.attributes.end());

					for (; it != itEnd; it++) {
						if ((it->first.compare("POSITION") == 0) ||
							(it->first.compare("NORMAL") == 0) ||
							(it->first.compare("TEXCOORD_0") == 0)) {
							glDisableVertexAttribArray(gGLProgramState.attribs[it->first]);
						}
					}
				}
			}

		}

	}
	

	// attributes (of vertices)

	// TODO: textures

}

/**
 * Perform rasterization.
 */
void rasterize(uchar4 *pbo) {
    int sideLength2d = 8;
    dim3 blockSize2d(sideLength2d, sideLength2d);
    dim3 blockCount2d((width  - 1) / blockSize2d.x + 1,
                      (height - 1) / blockSize2d.y + 1);

    // TODO: Execute your rasterization pipeline here
    // (See README for rasterization pipeline outline.)

    // Copy depthbuffer colors into framebuffer
    render<<<blockCount2d, blockSize2d>>>(width, height, dev_depthbuffer, dev_framebuffer);
    // Copy framebuffer into OpenGL buffer for OpenGL previewing
    sendImageToPBO<<<blockCount2d, blockSize2d>>>(pbo, width, height, dev_framebuffer);
    checkCUDAError("rasterize");
}

/**
 * Called once at the end of the program to free CUDA memory.
 */
void rasterizeFree() {
    cudaFree(dev_bufIdx);
    dev_bufIdx = NULL;

    cudaFree(dev_bufVertex);
    dev_bufVertex = NULL;

    cudaFree(dev_primitives);
    dev_primitives = NULL;

    cudaFree(dev_depthbuffer);
    dev_depthbuffer = NULL;

    cudaFree(dev_framebuffer);
    dev_framebuffer = NULL;

    checkCUDAError("rasterizeFree");
}
