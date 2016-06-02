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

typedef unsigned short VertexIndex;
typedef glm::vec3 VertexAttributePosition;
typedef glm::vec3 VertexAttributeNormal;
typedef glm::vec2 VertexAttributeTexcoord;

//// VertexIn
//static int* dev_indices;
//
//static VertexAttributePosition* dev_vertexAttributePosition = NULL;
//static VertexAttributeNormal* dev_vertexAttributeNormal = NULL;
//static VertexAttributeTexcoord* dev_vertexAttributeTexcoord0 = NULL;




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









// Buffer State

// Attribute State (bufferview pointer, byte offset, byte stride, count(vec2/vec3), primitive type)

// 1. for mesh, for each primitive, create device buffer for indices and attributes (accessor), and bind all attribute(acessor) state
// 2. (kern) vertex shader (transform position)
// 3. for each primitive, do primitive assembly ( each attribute buffer => Primitive * dev_primitives)





struct PrimitiveDevBufPointers {
	VertexIndex* dev_indices;
	//VertexIn* dev_vertices;
	VertexAttributePosition* dev_position;
	VertexAttributeNormal* dev_normal;
	VertexAttributeTexcoord* dev_texcoord0;

	//TODO: add more attributes when necessary
};

//static std::map<std::string, > primitiveBufState;
//
static std::map<std::string, char*> bufferViewDevPointers;
//
//static std::map<std::string, VertexAttributePosition*> mapBufferView2VertexAttributePosition;
//

// <mesh_name, vector<Primitives> >
static std::map<std::string, std::vector<PrimitiveDevBufPointers>> mesh2PrimitiveVector;

static std::map<std::string, PrimitiveDevBufPointers> primitiveDevPointers;




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

			// ? __constant__
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



	// 2. for each meshes: for each primitive: build VertexIn with indices, attributes
	{
		std::map<std::string, tinygltf::Mesh>::const_iterator it(scene.meshes.begin());
		std::map<std::string, tinygltf::Mesh>::const_iterator itEnd(scene.meshes.end());

		// for each mesh
		for (; it != itEnd; it++) {
			const tinygltf::Mesh & mesh = it->second;

			auto res = mesh2PrimitiveVector.insert(std::pair<std::string, std::vector<PrimitiveDevBufPointers>>(mesh.name, std::vector<PrimitiveDevBufPointers>()));
			std::vector<PrimitiveDevBufPointers> & primitiveVector = res.first;

			// for each primitive
			for (size_t i = 0; i < mesh.primitives.size(); i++) {
				const tinygltf::Primitive &primitive = mesh.primitives[i];

				if (primitive.indices.empty())
					return;


				// TODO: ? now position, normal, etc data type is predefined
				VertexIndex* dev_indices;
				VertexAttributePosition* dev_position;
				VertexAttributeNormal* dev_normal;
				VertexAttributeTexcoord* dev_texcoord0;



				// ----------Attributes-------------

				std::map<std::string, std::string>::const_iterator it(
					primitive.attributes.begin());
				std::map<std::string, std::string>::const_iterator itEnd(
					primitive.attributes.end());

				// Assume TEXTURE_2D target for the texture object.
				//glBindTexture(GL_TEXTURE_2D, gMeshState[mesh.name].diffuseTex[i]);

				// for each attribute
				for (; it != itEnd; it++) {
					const tinygltf::Accessor &accessor = scene.accessors[it->second];
					const tinygltf::BufferView &bufferView = scene.bufferViews[accessor.bufferView];

					//glBindBuffer(GL_ARRAY_BUFFER, gBufferState[accessor.bufferView].vb);
					//CheckErrors("bind buffer");
					int n = 1;
					if (accessor.type == TINYGLTF_TYPE_SCALAR) {
						n = 1;
					}
					else if (accessor.type == TINYGLTF_TYPE_VEC2) {
						n = 2;
					}
					else if (accessor.type == TINYGLTF_TYPE_VEC3) {
						n = 3;
					}
					else if (accessor.type == TINYGLTF_TYPE_VEC4) {
						n = 4;
					}

					//// it->first would be "POSITION", "NORMAL", "TEXCOORD_0", ...
					//if ((it->first.compare("POSITION") == 0) ||
					//	(it->first.compare("NORMAL") == 0) ||
					//	(it->first.compare("TEXCOORD_0") == 0)) {


					//	glVertexAttribPointer(
					//		gGLProgramState.attribs[it->first], count, accessor.componentType,
					//		GL_FALSE, accessor.byteStride, BUFFER_OFFSET(accessor.byteOffset));
					//	CheckErrors("vertex attrib pointer");
					//	glEnableVertexAttribArray(gGLProgramState.attribs[it->first]);
					//	CheckErrors("enable vertex attrib array");
					//}
					char * dev_bufferView = bufferViewDevPointers.at(accessor.bufferView);



					if (it->first.compare("POSITION") == 0) {
						int byteLength = numVertices * n * sizeof(VertexAttributePosition);

						// ???????? TODO: byteStride ????????????????
						// TODO: use a kernel with stride to copy data
						cudaMalloc(&dev_position, byteLength);
						cudaMemcpy(
							dev_position,
							dev_bufferView + bufferView.byteOffset + accessor.byteOffset,
							byteLength,
							cudaMemcpyDeviceToDevice);
					} 
					else if (it->first.compare("NORMAL") == 0) {
						int byteLength = numVertices * n * sizeof(VertexAttributeNormal);
						cudaMalloc(&dev_normal, byteLength);
						cudaMemcpy(
							dev_normal,
							dev_bufferView + bufferView.byteOffset + accessor.byteOffset,
							byteLength,
							cudaMemcpyDeviceToDevice);
					}
					else if (it->first.compare("TEXCOORD_0") == 0) {
						int byteLength = numVertices * n * sizeof(VertexAttributeTexcoord);
						cudaMalloc(&dev_texcoord0, byteLength);
						cudaMemcpy(
							dev_texcoord0,
							dev_bufferView + bufferView.byteOffset + accessor.byteOffset,
							byteLength,
							cudaMemcpyDeviceToDevice);
					}
				}





				// ----------Indices-------------

				const tinygltf::Accessor &indexAccessor = scene.accessors[primitive.indices];
				const tinygltf::BufferView &bufferView = scene.bufferViews[indexAccessor.bufferView];
				char * dev_bufferView = bufferViewDevPointers.at(indexAccessor.bufferView);
				
				// !! assume type is SCALAR
				int byteLength = indexAccessor.count * sizeof(VertexIndex);

				cudaMalloc(&dev_indices, byteLength);
				cudaMemcpy(
					dev_indices,
					dev_bufferView + bufferView.byteOffset + indexAccessor.byteOffset,
					byteLength,
					cudaMemcpyDeviceToDevice);


				// ----------Materials-------------
				// TODO


				// at the end of the for loop of primitive
				// push dev pointers to map
				primitiveVector.push_back(PrimitiveDevBufPointers{
					dev_indices,
					dev_position,
					dev_normal,
					dev_texcoord0
				});
			}

		}

	}
	


	// Finally, cudaFree raw dev_bufferViews




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
