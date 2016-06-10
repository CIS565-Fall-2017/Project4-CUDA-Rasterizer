/**
 * @file      rasterize.cu
 * @brief     CUDA-accelerated rasterization pipeline.
 * @authors   Skeleton code: Yining Karl Li, Kai Ninomiya, Shuai Shao (Shrek)
 * @date      2012-2016
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

namespace {

	struct VertexOut {
		glm::vec4 pos;

		glm::vec3 eyePos;	// for shading
		glm::vec3 eyeNor;	// normal will go wrong after perspective transform

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
		int primitiveMode;	//from tinygltfloader macro
		PrimitiveType primitiveType;
		int numPrimitives;
		int numIndices;
		int numVertices;

		// Vertex In, const after loaded
		VertexIndex* dev_indices;
		VertexAttributePosition* dev_position;
		VertexAttributeNormal* dev_normal;
		VertexAttributeTexcoord* dev_texcoord0;

		// Vertex Out, changing for each frame
		VertexOut* dev_verticesOut;

		//TODO: add more attributes when necessary
	};

}

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



// TODO: delete me for assignment !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
static int * dev_depth = NULL;
__global__
void initDepth(int w, int h, int * depth)
{
	int x = (blockIdx.x * blockDim.x) + threadIdx.x;
	int y = (blockIdx.y * blockDim.y) + threadIdx.y;

	if (x < w && y < h)
	{
		int index = x + (y * w);

		depth[index] = INT_MAX;

	}


}
//!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!1





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

	// TODO delete
	cudaFree(dev_depth);
	cudaMalloc(&dev_depth, width * height *sizeof(int));
}











// Buffer State

// Attribute State (bufferview pointer, byte offset, byte stride, count(vec2/vec3), primitive type)

// 1. for mesh, for each primitive, create device buffer for indices and attributes (accessor), and bind all attribute(acessor) state
// 2. (kern) vertex shader (transform position)
// 3. for each primitive, do primitive assembly ( each attribute buffer => Primitive * dev_primitives)






/**
* kern function with support for stride to sometimes replace cudaMemcpy
*/
__global__ 
void _deviceBufferCopy(int N, BufferByte* dev_dst, const BufferByte* dev_src, int byteStride, int byteOffset, int componentTypeByteSize) {
	
	int i = (blockIdx.x * blockDim.x) + threadIdx.x;

	if (i < N) {
		for (int j = 0; j < componentTypeByteSize; j++) {
			dev_dst[i * componentTypeByteSize + j] = dev_src[byteOffset + i * (byteStride == 0 ? componentTypeByteSize : byteStride) + j];
		}
	}
	

}


void rasterizeSetBuffers(const tinygltf::Scene & scene) {

	totalNumPrimitives = 0;

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

				dim3 numThreadsPerBlock(128);
				dim3 numBlocks((numIndices + numThreadsPerBlock.x - 1) / numThreadsPerBlock.x);
				cudaMalloc(&dev_indices, byteLength);
				_deviceBufferCopy << <numBlocks, numThreadsPerBlock >> > (
					numIndices,
					(BufferByte*)dev_indices,
					dev_bufferView,
					indexAccessor.byteStride,
					indexAccessor.byteOffset,
					componentTypeByteSize);


				checkCUDAError("Set Index Buffer");

				//!!!!!!!!!TODO: delete test
				cudaDeviceSynchronize();
				std::vector<VertexIndex> indicesWatch(numIndices);
				cudaMemcpy(&indicesWatch.at(0), dev_indices, numIndices*sizeof(VertexIndex), cudaMemcpyDeviceToHost);

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

				int numVertices = 0;
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
					
					numVertices = accessor.count;
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


					dim3 numThreadsPerBlock(128);
					dim3 numBlocks((numVertices + numThreadsPerBlock.x - 1) / numThreadsPerBlock.x);
					int byteLength = numVertices * n * componentTypeByteSize;
					cudaMalloc(dev_attribute, byteLength);
					_deviceBufferCopy << <numBlocks, numThreadsPerBlock >> > (
						numVertices,
						*dev_attribute,
						dev_bufferView,
						accessor.byteStride,
						accessor.byteOffset,
						componentTypeByteSize);

					std::string msg = "Set Attribute Buffer: " + it->first;
					checkCUDAError(msg.c_str());
				}

				// malloc for VertexOut
				VertexOut* dev_vertexOut;
				cudaMalloc(&dev_vertexOut, numVertices * sizeof(VertexOut));
				checkCUDAError("Malloc VertexOut Buffer");

				// ----------Materials-------------
				// TODO


				// at the end of the for loop of primitive
				// push dev pointers to map
				primitiveVector.push_back(PrimitiveDevBufPointers{
					primitive.mode,
					primitiveType,
					numPrimitives,
					numIndices,
					numVertices,

					dev_indices,
					dev_position,
					dev_normal,
					dev_texcoord0,

					dev_vertexOut	//VertexOut
				});

				totalNumPrimitives += numPrimitives;

			} // for each primitive

		} // for each mesh

	}
	

	// 3. Malloc for dev_primitives
	{
		cudaMalloc(&dev_primitives, totalNumPrimitives * sizeof(Primitive));
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
void _vertexTransformAndAssembly(int numVertices, PrimitiveDevBufPointers primitive, glm::mat4 MVP, glm::mat4 MV, glm::mat3 MV_normal, int width, int height) {
	// TODO: delete for assignments

	// vertex id
	int vid = (blockIdx.x * blockDim.x) + threadIdx.x;
	if (vid < numVertices) {
		primitive.dev_verticesOut[vid].pos = MVP * glm::vec4(primitive.dev_position[vid], 1.0f);
		glm::vec4 temp = MV * glm::vec4(primitive.dev_position[vid], 1.0f);
		primitive.dev_verticesOut[vid].eyePos = glm::vec3(temp) / temp.w;
		primitive.dev_verticesOut[vid].eyeNor = MV_normal * primitive.dev_normal[vid];
		//primitive.dev_verticesOut[vid].texcoord0 = primitive.dev_texcoord0[vid];

		// clipping space (to NDC -1,1) to viewport
		glm::vec4 & pos = primitive.dev_verticesOut[vid].pos;
		pos.x = 0.5f * (float)width * (pos.x / pos.w + 1.0f);
		pos.y = 0.5f * (float)height * (pos.y / pos.w + 1.0f);
		pos.z = 0.5f * (pos.z / pos.w + 1.0f);

		//perspective correct interpolation
		primitive.dev_verticesOut[vid].texcoord0 = primitive.dev_texcoord0[vid] / pos.w;
	}
}



static int curPrimitiveBeginId = 0;

__global__ 
void _primitiveAssembly(int numIndices, int curPrimitiveBeginId, Primitive* dev_primitives, PrimitiveDevBufPointers primitive) {
	// TODO: delete for assignments

	// index id
	int iid = (blockIdx.x * blockDim.x) + threadIdx.x;

	if (iid < numIndices) {
		int pid;	//id for cur primitives vector
		if (primitive.primitiveMode == TINYGLTF_MODE_TRIANGLES) {
			pid = iid / (int)primitive.primitiveType;
			dev_primitives[pid + curPrimitiveBeginId].v[iid % (int)primitive.primitiveType]
				= primitive.dev_verticesOut[primitive.dev_indices[iid]];
		}
	}
	
	// TODO: other primitive types
}











// -----------------------------------------------------------
// TODO: delete for assignment

struct Edge
{
	VertexOut v[2];

	float x, z;
	float dx, dz;


	//
	//VertexOut cur_v;	//used for interpolate between a scan line
	float gap_y;
};
//e.v[0] is the one with smaller y value
//scan from v[0] to v[1]
__device__
void constructEdge(Edge & e, const VertexOut & v0, const VertexOut & v1)
{
	if (v0.pos.y <= v1.pos.y)
	{
		e.v[0] = v0;
		e.v[1] = v1;
	}
	else
	{
		e.v[0] = v1;
		e.v[1] = v0;
	}

	e.gap_y = 0.0f;

}

__device__
float initEdge(Edge & e, float y)
{
	e.gap_y = e.v[1].pos.y - e.v[0].pos.y;

	e.dx = (e.v[1].pos.x - e.v[0].pos.x) / e.gap_y;
	e.dz = (e.v[1].pos.z - e.v[0].pos.z) / e.gap_y;
	e.x = e.v[0].pos.x + (y - e.v[0].pos.y) * e.dx;
	e.z = e.v[0].pos.z + (y - e.v[0].pos.y) * e.dz;

	return (y - e.v[0].pos.y) / e.gap_y;
}

__device__
void updateEdge(Edge & e)
{
	e.x += e.dx;
	e.z += e.dz;
}



__device__
void drawOneScanLine(int width, const Edge & e1, const Edge & e2, int y, 
	float u1, float u2, Fragment * fragments, int * depth, const Primitive & tri)
{

	// Find the starting and ending x coordinates and
	// clamp them to be within the visible region
	int x_left = (int)(ceilf(e1.x) + EPSILON);
	int x_right = (int)(ceilf(e2.x) + EPSILON);


	float x_left_origin = e1.x;
	float x_right_origin = e2.x;

	if (x_left < 0)
	{
		x_left = 0;
	}

	if (x_right > width)
	{
		x_right = width;
	}

	// Discard scanline with no actual rasterization and also
	// ensure that the length is larger than zero
	if (x_left >= x_right) { return; }


	//TODO: get two interpolated segment end points
	//VertexOut cur_v_e1 = interpolateVertexOut(e1.v[0], e1.v[1], u1);
	//VertexOut cur_v_e2 = interpolateVertexOut(e2.v[0], e2.v[1], u2);


	//Initialize attributes
	float dz = (e2.z - e1.z) / (e2.x - e1.x);
	float z = e1.z + (x_left_origin - e1.x) * dz;

	//Interpolate
	//printf("%d,%d\n", x_left, x_right);
	//float gap_x = x_right_origin - x_left_origin;
	for (int x = x_left; x < x_right; ++x)
	{

		int idx = x + y * width;

		//VertexOut p = interpolateVertexOut(cur_v_e1, cur_v_e2, ((float)x-x_left_origin) / gap_x);


		//using barycentric
		glm::vec3 t[3] = { glm::vec3(tri.v[0].pos), glm::vec3(tri.v[1].pos), glm::vec3(tri.v[2].pos) };
		glm::vec3 u = calculateBarycentricCoordinate(t, glm::vec2(x, y));

		VertexOut p;
		//p.pos = u.x * tri.v[0].pos + u.y * tri.v[1].pos + u.z * tri.v[2].pos;
		//p.pos.w = u.x * tri.v[0].pos.w + u.y * tri.v[1].pos.w + u.z * tri.v[2].pos.w;
		p.pos = u.x * tri.v[0].pos + u.y * tri.v[1].pos + u.z * tri.v[2].pos;
		p.eyeNor = u.x * tri.v[0].eyeNor + u.y * tri.v[1].eyeNor + u.z * tri.v[2].eyeNor;
		//p.pos.w = u.x * tri.v[0].pos.w + u.y * tri.v[1].pos.w + u.z * tri.v[2].pos.w;

		int z_int = (int)(z * INT_MAX);

		int* address = &depth[idx];

		atomicMin(address, z_int);

		if (*address == z_int)
		{
			//fragments[idx].depth = z;
			//fragments[idx].color = glm::vec3(p.pos.z);

			//fragments[idx].color = glm::vec3(p.pos.z);

			fragments[idx].color = p.eyeNor;

			//fragments[idx].color = glm::vec3(1.0f, 1.0f, 1.0f);

			//fragments[idx].has_fragment = true;

		}



		z += dz;
	}
}


/**
* Rasterize the area between two edges as the left and right limit.
* e1 - longest y span
*/
__device__
void drawAllScanLines(int width, int height, Edge  e1, Edge  e2, 
	Fragment * fragments, int * depth, const Primitive &  tri)
{
	// Discard horizontal edge as there is nothing to rasterize
	if (e2.v[1].pos.y - e2.v[0].pos.y == 0.0f) { return; }

	// Find the starting and ending y positions and
	// clamp them to be within the visible region
	int y_bot = (int)(ceilf(e2.v[0].pos.y) + EPSILON);
	int y_top = (int)(ceilf(e2.v[1].pos.y) + EPSILON);



	float y_bot_origin = ceilf(e2.v[0].pos.y);
	float y_top_origin = floorf(e2.v[1].pos.y);

	if (y_bot < 0)
	{
		y_bot = 0;

	}

	if (y_top > height)
	{
		y_top = height;
	}


	//Initialize edge's structure
	float u1_base = initEdge(e1, y_bot_origin);
	initEdge(e2, y_bot_origin);


	//printf("%f,%f\n", e1.v[0].uv.x / e1.v[0].divide_w_clip, e1.v[0].uv.y / e1.v[0].divide_w_clip );

	for (int y = y_bot; y < y_top; ++y)
	{

		float u2 = ((float)y - y_bot_origin) / e2.gap_y;
		float u1 = u1_base + ((float)y - y_bot_origin) / e1.gap_y;
		if (e1.x <= e2.x)
		{
			drawOneScanLine(width, e1, e2, y, u1, u2, fragments, depth, tri);
		}
		else
		{
			drawOneScanLine(width, e2, e1, y, u2, u1, fragments, depth, tri);
		}

		//update edge
		updateEdge(e1);
		updateEdge(e2);
	}
}

/**
* Each thread handles one triangle
* rasterization
*/
__global__
void kernScanLineForOneTriangle(int num_tri, int width, int height
, Primitive * triangles, Fragment * depth_fragment, int * depth)
{
	int triangleId = blockDim.x * blockIdx.x + threadIdx.x;

	if (triangleId >= num_tri)
	{
		return;
	}


	Primitive tri = triangles[triangleId];	//copy


	


	bool outside = true;

	//currently tri.v are in clipped coordinates
	//need to transform to viewport coordinate
	for (int i = 0; i < 3; i++)
	{


		////////
		if (tri.v[i].pos.x < (float)width && tri.v[i].pos.x >= 0
			&& tri.v[i].pos.y < (float)height && tri.v[i].pos.y >= 0)
		{
			outside = false;
			// test------------------------------------
			int idx = tri.v[i].pos.x + tri.v[i].pos.y * width;
			depth_fragment[idx].color = glm::vec3(1.0f, 1.0f, 1.0f);
			//printf("%d", triangleId);
		}
	}


	//discard triangles that are totally out of the viewport
	if (outside)
	{
		return;
	}
	/////




	


	//build edge
	// for line scan
	Edge edges[3];

	constructEdge(edges[0], tri.v[0], tri.v[1]);
	constructEdge(edges[1], tri.v[1], tri.v[2]);
	constructEdge(edges[2], tri.v[2], tri.v[0]);


	//Find the edge with longest y span
	float maxLength = 0.0f;
	int longEdge = -1;
	for (int i = 0; i < 3; ++i)
	{
		float length = edges[i].v[1].pos.y - edges[i].v[0].pos.y;
		if (length > maxLength)
		{
			maxLength = length;
			longEdge = i;
		}
	}


	// get indices for other two shorter edges
	int shortEdge0 = (longEdge + 1) % 3;
	int shortEdge1 = (longEdge + 2) % 3;

	// Rasterize two parts separately
	drawAllScanLines(width, height, edges[longEdge], edges[shortEdge0], depth_fragment, depth, tri);
	drawAllScanLines(width, height, edges[longEdge], edges[shortEdge1], depth_fragment, depth, tri);



}










/**
 * Perform rasterization.
 */
void rasterize(uchar4 *pbo, const glm::mat4 & MVP, const glm::mat4 & MV, const glm::mat3 MV_normal) {
    int sideLength2d = 8;
    dim3 blockSize2d(sideLength2d, sideLength2d);
    dim3 blockCount2d((width  - 1) / blockSize2d.x + 1,
		(height - 1) / blockSize2d.y + 1);

	// TODO: Execute your rasterization pipeline here
	// (See README for rasterization pipeline outline.)

	// Vertex Process & primitive assembly
	{
		curPrimitiveBeginId = 0;
		dim3 numThreadsPerBlock(128);

		auto it = mesh2PrimitivesMap.begin();
		auto itEnd = mesh2PrimitivesMap.end();

		for (; it != itEnd; ++it) {
			auto p = (it->second).begin();	// each primitive
			auto pEnd = (it->second).end();
			for (; p != pEnd; ++p) {
				dim3 numBlocksForVertices((p->numVertices + numThreadsPerBlock.x - 1) / numThreadsPerBlock.x);
				dim3 numBlocksForIndices((p->numIndices + numThreadsPerBlock.x - 1) / numThreadsPerBlock.x);

				_vertexTransformAndAssembly << < numBlocksForVertices, numThreadsPerBlock >> >(p->numVertices, *p, MVP, MV, MV_normal, width, height);
				checkCUDAError("Vertex Processing");
				cudaDeviceSynchronize();
				_primitiveAssembly << < numBlocksForIndices, numThreadsPerBlock >> >
					(p->numIndices, 
					curPrimitiveBeginId, 
					dev_primitives, 
					*p);
				checkCUDAError("Primitive Assembly");

				curPrimitiveBeginId += p->numPrimitives;
			}
		}

		checkCUDAError("Vertex Processing and Primitive Assembly");



		// test, copy back to host memory
		std::vector<Primitive> hst_primitives(totalNumPrimitives);
		cudaMemcpy(&hst_primitives.at(0), dev_primitives, totalNumPrimitives * sizeof(Primitive), cudaMemcpyDeviceToHost);
		checkCUDAError("mem test");

		
		
	}
	
	// !!!!!!!!!!!!!!!!Rasterize: temp test
	cudaMemset(dev_depthbuffer, 0, width * height * sizeof(Fragment));
	initDepth << <blockCount2d, blockSize2d >> >(width, height, dev_depth);
	{
		dim3 numThreadsPerBlock(64);
		dim3 numBlocks((totalNumPrimitives + numThreadsPerBlock.x - 1) / numThreadsPerBlock.x);
		kernScanLineForOneTriangle << <numBlocks, numThreadsPerBlock >> >(totalNumPrimitives, width, height, dev_primitives, dev_depthbuffer, dev_depth);
	}



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
