/**
 * @file      rasterize.cu
 * @brief     CUDA-accelerated rasterization pipeline.
 * @authors   Skeleton code: Yining Karl Li, Kai Ninomiya, Shuai Shao (Shrek)
 * @date      2012-2015
 * @copyright University of Pennsylvania & STUDENT
 */



#include <cmath>
#include <cstdio>
#include <cuda.h>
#include <cuda_runtime.h>
#include <thrust/random.h>
#include <util/checkCUDAError.h>
#include "rasterizeTools.h"

#include "rasterize.h"


#include <util/tiny_gltf_loader.h>

struct VertexOut {
	glm::vec4 pos;

	glm::vec3 worldPos;
	glm::vec3 worldNor;

	//glm::vec3 col;
	glm::vec2 texcoord0;
};

typedef unsigned short VertexIndex;
typedef glm::vec3 VertexAttributePosition;
typedef glm::vec3 VertexAttributeNormal;
typedef glm::vec2 VertexAttributeTexcoord;

typedef unsigned char BufferByte;

enum PrimitiveType{
	Point = 1,
	Line = 2,
	Triangle = 3
};

struct Primitive {
	PrimitiveType primitiveType = Triangle;	// C++ 11 init
    VertexOut v[3];
};
struct Fragment {
    glm::vec3 color;
};


struct PrimitiveDevBufPointers {

	int primitiveMode;	//from tinygltfloader
	PrimitiveType primitiveType;
	int numPrimitives;

	// Vertex In, const after loaded
	VertexIndex* dev_indices;
	VertexAttributePosition* dev_position;
	VertexAttributeNormal* dev_normal;
	VertexAttributeTexcoord* dev_texcoord0;

	// Vertex Out, changing for each frame
	VertexOut* dev_verticesOut;

	//TODO: add more attributes when necessary
};

static std::map<std::string, std::vector<PrimitiveDevBufPointers>> mesh2PrimitivesMap;


static int width = 0;
static int height = 0;

static int totalNumPrimitives = 0;
static Primitive *dev_primitives = NULL;
static Fragment *dev_depthbuffer = NULL;
static glm::vec3 *dev_framebuffer = NULL;


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











// Buffer State

// Attribute State (bufferview pointer, byte offset, byte stride, count(vec2/vec3), primitive type)

// 1. for mesh, for each primitive, create device buffer for indices and attributes (accessor), and bind all attribute(acessor) state
// 2. (kern) vertex shader (transform position)
// 3. for each primitive, do primitive assembly ( each attribute buffer => Primitive * dev_primitives)








__global__ 
void _deviceBufferCopy(int N, BufferByte* dev_dst, const BufferByte* dev_src, int byteStride, int byteOffset, int componentTypeByteSize) {
	
	int i = (blockIdx.x * blockDim.x) + threadIdx.x;

	if (i < N) {
		for (int j = 0; j < componentTypeByteSize; j++) {
			dev_dst[i + j] = dev_src[byteOffset + i * byteStride + j];
		}
	}
	

}


void rasterizeSetBuffers(const tinygltf::Scene & scene) {

	std::map<std::string, BufferByte*> bufferViewDevPointers;

	// 1. copy all `bufferViews` to device memory
	{
		std::map<std::string, tinygltf::BufferView>::const_iterator it(
			scene.bufferViews.begin());
		std::map<std::string, tinygltf::BufferView>::const_iterator itEnd(
			scene.bufferViews.end());

		for (; it != itEnd; it++) {
			const std::string key = it->first;
			const tinygltf::BufferView &bufferView = it->second;
			if (bufferView.target == 0) {
				continue; // Unsupported bufferView.
			}

			//const tinygltf::Buffer &buffer = scene.buffers[bufferView.buffer];
			const tinygltf::Buffer &buffer = scene.buffers.at(bufferView.buffer);

			// ? __constant__
			BufferByte* dev_bufferView;
			cudaMalloc(&dev_bufferView, bufferView.byteLength);
			cudaMemcpy(dev_bufferView, &buffer.data.front() + bufferView.byteOffset, bufferView.byteLength, cudaMemcpyHostToDevice);

			checkCUDAError("Set BufferView Device Mem");

			bufferViewDevPointers.insert(std::make_pair(key, dev_bufferView));
		}
	}



	// 2. for each meshes: for each primitive: build device buffer of indices, materail, and each attributes
	{
		std::map<std::string, tinygltf::Mesh>::const_iterator it(scene.meshes.begin());
		std::map<std::string, tinygltf::Mesh>::const_iterator itEnd(scene.meshes.end());

		// for each mesh
		for (; it != itEnd; it++) {
			const tinygltf::Mesh & mesh = it->second;

			//std::pair<std::map<std::string, std::vector<PrimitiveDevBufPointers>>::iterator, bool> res = mesh2PrimitivesMap.insert(std::pair<std::string, std::vector<PrimitiveDevBufPointers>>(mesh.name, std::vector<PrimitiveDevBufPointers>()));
			auto res = mesh2PrimitivesMap.insert(std::pair<std::string, std::vector<PrimitiveDevBufPointers>>(mesh.name, std::vector<PrimitiveDevBufPointers>()));
			std::vector<PrimitiveDevBufPointers> & primitiveVector = (res.first)->second;

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

				// ----------Indices-------------

				const tinygltf::Accessor &indexAccessor = scene.accessors.at(primitive.indices);
				const tinygltf::BufferView &bufferView = scene.bufferViews.at(indexAccessor.bufferView);
				BufferByte* dev_bufferView = bufferViewDevPointers.at(indexAccessor.bufferView);

				// !! assume type is SCALAR
				int n = 1;
				int numIndices = indexAccessor.count;
				int componentTypeByteSize = sizeof(VertexIndex);
				int byteLength = numIndices * n * componentTypeByteSize;


				cudaMalloc(&dev_indices, byteLength);

				dim3 numBlocks(128);
				dim3 numThreadsPerBlock((numIndices + numBlocks.x - 1) / numBlocks.x);
				cudaMalloc(&dev_position, byteLength);
				_deviceBufferCopy << <numBlocks, numThreadsPerBlock >> > (
					numIndices,
					(BufferByte*)dev_indices,
					dev_bufferView,
					indexAccessor.byteStride,
					bufferView.byteOffset + indexAccessor.byteOffset,
					componentTypeByteSize);


				checkCUDAError("Set Index Buffer");


				// ---------Primitive Info-------


				// !! LINE_STRIP is not supported in tinygltfloader
				int numPrimitives;
				PrimitiveType primitiveType;
				switch (primitive.mode) {
				case TINYGLTF_MODE_TRIANGLES:
					primitiveType = PrimitiveType::Triangle;
					numPrimitives = numIndices / 3;
					break;
				case TINYGLTF_MODE_TRIANGLE_STRIP:
					primitiveType = PrimitiveType::Triangle;
					numPrimitives = numIndices - 2;
					break;
				case TINYGLTF_MODE_TRIANGLE_FAN:
					primitiveType = PrimitiveType::Triangle;
					numPrimitives = numIndices - 2;
					break;
				case TINYGLTF_MODE_LINE:
					primitiveType = PrimitiveType::Line;
					numPrimitives = numIndices / 2;
					break;
				case TINYGLTF_MODE_LINE_LOOP:
					primitiveType = PrimitiveType::Line;
					numPrimitives = numIndices + 1;
					break;
				case TINYGLTF_MODE_POINTS:
					primitiveType = PrimitiveType::Point;
					numPrimitives = numIndices;
					break;
				default:
					// TODO: error
					break;
				};


				// ----------Attributes-------------

				//std::map<std::string, std::string>::const_iterator it(primitive.attributes.begin());
				auto it(primitive.attributes.begin());
				//std::map<std::string, std::string>::const_iterator itEnd(primitive.attributes.end());
				auto itEnd(primitive.attributes.end());

				// for each attribute
				for (; it != itEnd; it++) {
					const tinygltf::Accessor &accessor = scene.accessors.at(it->second);
					const tinygltf::BufferView &bufferView = scene.bufferViews.at(accessor.bufferView);

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

					BufferByte * dev_bufferView = bufferViewDevPointers.at(accessor.bufferView);
					BufferByte ** dev_attribute = NULL;
					
					int numVertices = accessor.count;
					int componentTypeByteSize;

					if (it->first.compare("POSITION") == 0) {
						componentTypeByteSize = sizeof(VertexAttributePosition);
						dev_attribute = (BufferByte**)&dev_position;
					} 
					else if (it->first.compare("NORMAL") == 0) {
						componentTypeByteSize = sizeof(VertexAttributeNormal);
						dev_attribute = (BufferByte**)&dev_normal;
					}
					else if (it->first.compare("TEXCOORD_0") == 0) {
						componentTypeByteSize = sizeof(VertexAttributeTexcoord);
						dev_attribute = (BufferByte**)&dev_texcoord0;
					}


					dim3 numBlocks(128);
					dim3 numThreadsPerBlock((numVertices + numBlocks.x - 1) / numBlocks.x);
					int byteLength = numVertices * n * componentTypeByteSize;
					cudaMalloc(dev_attribute, byteLength);
					_deviceBufferCopy << <numBlocks, numThreadsPerBlock >> > (
						numVertices,
						*dev_attribute,
						dev_bufferView,
						accessor.byteStride,
						bufferView.byteOffset + accessor.byteOffset,
						componentTypeByteSize);

					std::string msg = "Set Attribute Buffer: " + it->first;
					checkCUDAError(msg.c_str());
				}



				// ----------Materials-------------
				// TODO


				// at the end of the for loop of primitive
				// push dev pointers to map
				primitiveVector.push_back(PrimitiveDevBufPointers{
					primitive.mode,
					primitiveType,
					numPrimitives,

					dev_indices,
					dev_position,
					dev_normal,
					dev_texcoord0,

					NULL	//VertexOut
				});
			} // for each primitive

		} // for each mesh

	}
	


	// Finally, cudaFree raw dev_bufferViews
	{

		std::map<std::string, BufferByte*>::const_iterator it(bufferViewDevPointers.begin());
		std::map<std::string, BufferByte*>::const_iterator itEnd(bufferViewDevPointers.end());
			
			//bufferViewDevPointers

		for (; it != itEnd; it++) {
			cudaFree(it->second);
		}

		checkCUDAError("Free BufferView Device Mem");
	}


}



/**
* for one primitive
* ?? can combine with pritimitiveAssembly to make only one kernel call??
*/
__global__ 
void _vertexTransformAndAssembly(int N, PrimitiveDevBufPointers primitive, glm::mat4 M) {
	// TODO: delete for assignments

	int i = (blockIdx.x * blockDim.x) + threadIdx.x;
	if (i < N) {
		primitive.dev_verticesOut[i].pos = M * glm::vec4(primitive.dev_position[i], 1.0f);
		primitive.dev_verticesOut[i].worldPos = primitive.dev_position[i];
		primitive.dev_verticesOut[i].worldNor = primitive.dev_normal[i];
		primitive.dev_verticesOut[i].texcoord0 = primitive.dev_texcoord0[i];
	}
}



static int curPrimitiveBeginId = 0;

__global__ 
void primitiveAssembly(int N, int curPrimitiveBeginId, Primitive* dev_primitives, PrimitiveDevBufPointers primitive) {
	// TODO: delete for assignments

	// TODO: output to dev_primitives
	int i = (blockIdx.x * blockDim.x) + threadIdx.x;

	if (i < N) {
		int temp;
		if (primitive.primitiveMode == TINYGLTF_MODE_TRIANGLES) {
			temp = i / (int)primitive.primitiveType;
			dev_primitives[temp + curPrimitiveBeginId].v[temp % (int)primitive.primitiveType]
				= primitive.dev_verticesOut[primitive.dev_indices[i]];
		}
	}
	
	// TODO: other primitive types
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

    // deconstruct primitives attribute/indices device buffer

	auto it(mesh2PrimitivesMap.begin());
	auto itEnd(mesh2PrimitivesMap.end());
	for (; it != itEnd; ++it) {
		for (auto p = it->second.begin(); p != it->second.end(); ++p) {
			cudaFree(p->dev_indices);
			cudaFree(p->dev_position);
			cudaFree(p->dev_normal);
			cudaFree(p->dev_texcoord0);
			cudaFree(p->dev_verticesOut);
			//TODO: release other attributes and materials
		}
	}

	////////////

    cudaFree(dev_primitives);
    dev_primitives = NULL;

    cudaFree(dev_depthbuffer);
    dev_depthbuffer = NULL;

    cudaFree(dev_framebuffer);
    dev_framebuffer = NULL;

    checkCUDAError("rasterizeFree");
}
