Instructions - CUDA-RASTERIZER
========================

This is due **Tuesday, October 25, evening at midnight**.

**Summary:**
In this project, you will use CUDA to implement a simplified
rasterized graphics pipeline, similar to the OpenGL pipeline. You will
implement vertex shading, primitive assembly, rasterization, fragment shading,
and a framebuffer. More information about the rasterized graphics pipeline can
be found in the class slides and in the CIS 560 lecture notes.

The base code provided includes a glTF loader (tinygltfloader) and much of the I/O and
bookkeeping code. It also includes some functions that you may find useful,
described below. The core rasterization pipeline is left for you to implement.

You are not required to use this base code if you don't want
to. You may also change any part of the base code as you please.
**This is YOUR project.**

**Recommendation:**
Every image you save should automatically get a different
filename. Don't delete all of them! For the benefit of your README, keep a
bunch of them around so you can pick a few to document your progress.


### Contents

* `src/` C++/CUDA source files.
* `util/` C++ utility files.
* `gltfs/` Example glTF test files
  * `gltfs/triangle/triangle.gltf` (1 triangle only, start with this)
  * `gltfs/box/box.gltf` (8 vertices, 12 triangles, start with this)
  * `gltfs/cow/cow.gltf`
  * `gltfs/duck/duck.gltf` (has a diffuse texture)
  * `gltfs/checkerboard/checkerboard.gltf` (has a diffuse texture, can be used for testing perspective correct interpolation)
  * `gltfs/CesiumMilkTruck/CesiumMilkTruck.gltf` (has several textures)
  * `gltfs/flower/flower.gltf` (model with a lot of layers from most angles)
  * `gltfs/2_cylinder_engine/2_cylinder_engine.gltf` (relatively complex model, no texture, need to rescale to show in the center of screen)
* `renders/` Test implementation render result of duck.gltf.
* `external/` Includes and static libraries for 3rd party libraries.

### Running the code

The main function requires a glTF model file. Call the program with
one as an argument: `cis565_rasterizer gltfs/duck/duck.gltf`.
(In Visual Studio, `../gltfs/duck/duck.gltf`.)

If you are using Visual Studio, you can set this in the Debugging > Command
Arguments section in the Project properties. Note that this value is different
for every different configuration type. You can also set the argument for all
configurations to simplify this. Make sure you get the path right; read
the console for errors.

You can also launch the built program from command line to feed the arguments.

## Requirements

**Ask on the mailing list for any clarifications.**

In this project, you are given the following code:

* A tiny glTF loader for loading glTF format models and converting them to
OpenGL-style buffers of index and vertex attribute data.
    * [glTF](https://github.com/KhronosGroup/glTF) is a standard model format (Patrick being one of the major contributors).
    No need to worry about its details because it's all done for you, unless
    you would like to support primitives besides triangles.
* Structs for some parts of the pipeline.
* Fragment-buffer-to-framebuffer copy.
* CUDA-GL interop.
* A simple interactive camera using the mouse. 

You need to implement the following features/pipeline stages:

* Vertex shading. (`_vertexTransformAndAssembly` in `rasterize.cu`)
* Primitive assembly with support for triangles read from buffers of index and
  vertex data. (code already provided, simply uncomment it) (`_primitiveAssembly` in `rasterize.cu`)
* Rasterization. (create your own function, call it in `rasterize`)
* Fragment shading. (create your own function, call it in `rasterize`)
* A depth buffer for storing and depth testing fragments. (you've been provided with a `int * dev_depth`, you can always change to your own version)
* Fragment-to-depth-buffer writing (**with** atomics for race avoidance).
* (Fragment shader) simple lighting scheme, such as Lambert or Blinn-Phong. (`render` in `rasterize.cu`)

See below for more guidance.

You are also required to implement at least 2.0 "points" worth in extra features.
(point values are given in parentheses):

* (1.0) Use shared memory in a feature. Ideas and suggestions:
   * Look for areas where you may need to access the same "chunk" of memory multiple times in a single thread
   * post processing, with the fragment buffer divided into "tiles" - SSAO? Bloom? Toon Shading?
   * shared memory uniforms - an array of skinning matrices? an array of lights?
   * could you use shared memory for some kinds of texture-based shaders?
* (2.0) [Tile-based pipeline](https://github.com/CIS565-Fall-2015/cis565-fall-2015.github.io/blob/master/lectures/10-Mobile-Graphics.pptx?raw=true)
* Additional pipeline stages.
   * (1.0) Tessellation shader.
   * (1.0) Geometry shader, able to output a variable number of primitives per
     input primitive, optimized using stream compaction (thrust allowed).
   * (0.5 **if not doing geometry shader**) Backface culling, optimized using
     stream compaction (thrust allowed).
   * (1.0) Transform feedback.
   * (0.5) Blending (when writing into framebuffer).
* (1.0) Instancing: draw one set of vertex data multiple times, each run
  through the vertex shader with a different ID.
* (0.5) Correct color interpolation between points on a primitive.
* (1.0) UV texture mapping with bilinear texture filtering and perspective
  correct texture coordinates.
* Support for rasterizing additional primitives:
   * (0.5) Lines or line strips.
   * (0.5) Points.
   * For rasterizing lines and points, you may start with a toggle mode that
   switches your pipeline from displaying triangles to displaying a wireframe
   or a point cloud.
* Anti-aliasing
   * (0.5) SSAA - supersample antialiasing
   * (1.0) [MSAA](https://mynameismjp.wordpress.com/2012/10/24/msaa-overview/) - multisample antialiasing, and performance comparison with the former one
* (1.0) Occlusion queries.
* (1.0) Order-independent translucency using a k-buffer.

This extra feature list is not comprehensive. If you have a particular idea
you would like to implement, please **contact us first**.

**IMPORTANT:**
For each extra feature, please provide the following brief analysis:

* Concise overview write-up of the feature.
* Performance impact of adding the feature (slower or faster).
  * where is the performance hit?
  * where is the performance improvement?
* If you did something to accelerate the feature, what did you do and why?
* How might this feature be optimized beyond your current implementation?


## Base Code Tour

You will be working primarily in : `rasterize.cu`.
Areas that you need to complete are
marked with a `TODO` comment. Functions that are useful
for reference are marked with the comment `CHECKITOUT`.

* `src/rasterize.cu` contains the core rasterization pipeline.
  * A few pre-made structs are included for you to use, but those marked with
    TODO will also be needed for a simple rasterizer. As with any part of the
    base code, you may modify or replace these as you see fit.
  * `PrimitiveDevBufPointers` freshly loaded attribute buffers, texture pointer, etc. from glTF models.
    Everything is either a basic data type or already copied to device memory.
  * `VertexOut` assembled vertex with various attributes ranging from position to texcoord.
  * `Primitive` assembled primitive with vertices
  * `Fragment` final fragments passed depth/scissor/stencil test waiting to shade

* `src/rasterizeTools.h` contains various useful tools
  * Includes a number of barycentric coordinate related functions that you may
    find useful in implementing scanline based rasterization.

* `util/utilityCore.hpp` serves as a kitchen-sink of useful functions.


## Rasterization Pipeline

Possible pipelines are described below. Pseudo-type-signatures are given.
Not all of the pseudocode arrays will necessarily actually exist in practice.

### First-Try Pipeline

This describes a minimal version of *one possible* graphics pipeline, similar
to modern hardware (DX/OpenGL). Yours need not match precisely.  To begin, try
to write a minimal amount of code as described here. Verify some output after
implementing each pipeline step. This will reduce the necessary time spent
debugging.

Start out by testing with some simple models (`box.gltf`).

* Clear the fragment buffer with some default value.
* Vertex shading:
  * `VertexIn[n] vs_input -> VertexOut[n] vs_output`
  * A minimal vertex shader will apply no transformations at all - it draws world position
    directly in normalized device coordinates (-1 to 1 in each dimension).
* Primitive assembly.
  * `VertexOut[n] vs_output -> Triangle[t] primitives`
  * Code is actually provided, simply uncomment it. (since you might need to
    read through the gltf setup code to fully implement this on your own)
* Rasterization.
  * `Triangle[t] primitives -> Fragment[m] rasterized`
  * A scanline implementation is simple to start with.
  * Parallelize over triangles. For now, loop over every pixel in the
    fragment buffer in each thread.
  * Note that you won't have any real allocated array of size `m`.
* Fragments to depth buffer.
  * `Fragment[m] rasterized -> Fragment[width][height] depthbuffer`
    * `depthbuffer` is for storing and depth testing fragments.
  * Results in race conditions - don't bother to fix these until it works!
  * Can really be done inside/after the fragment shading, if you call the fragment
    shading from the rasterization kernel for every fragment (including those
    which get occluded). Doing this before fragment shading may be faster (why?)
    but means the fragment shader cannot change the depth.
* Fragment shading.
  * `Fragment[width][height] depthbuffer ->`
  * A super-simple test fragment shader: output same color for every fragment.
    * Also try displaying various debug views (normals, etc.)
* Fragment to framebuffer writing.
  * `-> vec3[width][height] framebuffer`
  * Simply saves the fragment shader results into the framebuffer
    (to be displayed on the screen).



### A Useful Pipeline

* Clear the fragment and depth buffers with some default values.
  * You should be able to pass a default value to the clear function, so that
    you can set the clear color (background), clear depth, etc.
* Vertex shading:
  * `VertexIn[n] vs_input -> VertexOut[n] vs_output`
  * Apply some vertex transformation (e.g. model-view-projection matrix using
    `glm::lookAt ` and `glm::perspective `).
* Primitive assembly.
  * `VertexOut[n] vs_output -> Triangle[t] primitives`
  * As above.
  * Other primitive types are optional.
* Rasterization.
  * `Triangle[t] primitives -> Fragment[m] rasterized`
  * You may choose to do a shared-memory tiled rasterization method,
    which should have lower global memory bandwidth.
    It will also change other parts of the pipeline - only try this AFTER you
    having a working "useful" pipeline, both for sanity and for the sake of
    comparison.
  * Parallelize over triangles, but now avoid looping over all pixels:
    * When rasterizing a triangle, only scan over the box around the triangle
      (`getAABBForTriangle`).
* Fragments to depth buffer.
  * `Fragment[m] rasterized -> Fragment[width][height] depthbuffer`
    * `depthbuffer` is for storing and depth testing fragments.
  * This can be done before fragment shading, which prevents the fragment
    shader from changing the depth of a fragment.
    * This order results in an optimization: it allows you to do depth tests
      before spending execution time in complex fragment shader code!
    * If you want to be able to change the depth of a fragment, you'll have to
      make an adaptation. For example, you can add a separate shader stage
      which occurs during rasterization, which can change the depth.
      Or, you can call the fragment shader from the rasterization step - but
      be aware that the performance will be much worse - occupancy will be low
      due to the variable run length of each thread.
  * Handle race conditions! Since multiple primitives write fragments to the
    same fragment in the depth buffer, races must be avoided by using CUDA
    atomics.
    * *Approach 1:* Lock the location in the depth buffer during the time that
      a thread is comparing old and new fragment depths (and possibly writing
      a new fragment). This should work in all cases, but be slower.
      See the section below on implementing this.
    * *Approach 2:* Convert your depth value to a fixed-point `int`, and use
      `atomicMin` to store it into an `int`-typed depth buffer `intdepth`. After
      that, the value which is stored at `intdepth[i]` is (usually) that of the
      fragment which should be stored into the `fragment` depth buffer.
      * This may result in some rare race conditions (e.g. across blocks).
    * The `flower.gltf` test file is good for testing race conditions.
* Fragment shading.
  * `Fragment[width][height] depthbuffer ->`
  * Add a shading method, such as Lambert or Blinn-Phong. Lights can be defined
    by kernel parameters (like GLSL uniforms).
* Fragment to framebuffer writing.
  * `-> vec3[width][height] framebuffer`
  * Simply copies the colors out of the depth buffer into the framebuffer
    (to be displayed on the screen).

This is a suggested sequence of pipeline steps, but you may choose to alter the
order of this sequence or merge entire kernels as you see fit.  For example, if
you decide that doing so has benefits, you can choose to merge the vertex shader
and primitive assembly kernels, or merge the perspective transform into another
kernel. There is not necessarily a right sequence of kernels, and you may
choose any sequence that works.  Please document in your README what sequence
you choose and why.


## Resources

### CUDA Mutexes

Adapted from
[this StackOverflow question](http://stackoverflow.com/questions/21341495/cuda-mutex-and-atomiccas).

```cpp
__global__ void kernelFunction(...) {
    // Get a pointer to the mutex, which should be 0 right now.
    unsigned int *mutex = ...;

    // Loop-wait until this thread is able to execute its critical section.
    bool isSet;
    do {
        isSet = (atomicCAS(mutex, 0, 1) == 0);
        if (isSet) {
            // Critical section goes here.
            // The critical section MUST be inside the wait loop;
            // if it is afterward, a deadlock will occur.
        }
        if (isSet) {
            mutex = 0;
        }
    } while (!isSet);
}
```

### Links

The following resources may be useful for this project.

* Line Rasterization slides, MIT EECS 6.837, Teller and Durand
  * [Slides](http://groups.csail.mit.edu/graphics/classes/6.837/F02/lectures/6.837-7_Line.pdf)
* High-Performance Software Rasterization on GPUs
  * [Paper (HPG 2011)](http://www.tml.tkk.fi/~samuli/publications/laine2011hpg_paper.pdf)
  * [Code](http://code.google.com/p/cudaraster/)
  * Note that looking over this code for reference with regard to the paper is
    fine, but we most likely will not grant any requests to actually
    incorporate any of this code into your project.
  * [Slides](http://bps11.idav.ucdavis.edu/talks/08-gpuSoftwareRasterLaineAndPantaleoni-BPS2011.pdf)
* The Direct3D 10 System (SIGGRAPH 2006) - for those interested in doing
  geometry shaders and transform feedback
  * [Paper](http://dl.acm.org/citation.cfm?id=1141947)
  * [Paper, through Penn Libraries proxy](http://proxy.library.upenn.edu:2247/citation.cfm?id=1141947)
* Multi-Fragment Eﬀects on the GPU using the k-Buﬀer - for those who want to do
  order-independent transparency using a k-buffer
  * [Paper](http://www.inf.ufrgs.br/~comba/papers/2007/kbuffer_preprint.pdf)
* FreePipe: A Programmable, Parallel Rendering Architecture for Efficient
  Multi-Fragment Effects (I3D 2010)
  * [Paper](https://sites.google.com/site/hmcen0921/cudarasterizer)
* Writing A Software Rasterizer In Javascript
  * [Part 1](http://simonstechblog.blogspot.com/2012/04/software-rasterizer-part-1.html)
  * [Part 2](http://simonstechblog.blogspot.com/2012/04/software-rasterizer-part-2.html)
* How OpenGL works: software rendering in 500 lines of code
  * [Wiki](https://github.com/ssloy/tinyrenderer/wiki)


## Third-Party Code Policy

* Use of any third-party code must be approved by asking on our Google Group.
* If it is approved, all students are welcome to use it. Generally, we approve
  use of third-party code that is not a core part of the project. For example,
  for the path tracer, we would approve using a third-party library for loading
  models, but would not approve copying and pasting a CUDA function for doing
  refraction.
* Third-party code **MUST** be credited in README.md.
* Using third-party code without its approval, including using another
  student's code, is an academic integrity violation, and will, at minimum,
  result in you receiving an F for the semester.


## README

* A brief description of the project and the specific features you implemented.
* At least one screenshot of your project running.
* A 30 second or longer video of your project running.
* A performance analysis (described below).

### Performance Analysis

The performance analysis is where you will investigate how to make your CUDA
programs more efficient using the skills you've learned in class. You must have
performed at least one experiment on your code to investigate the positive or
negative effects on performance.

We encourage you to get creative with your tweaks. Consider places in your code
that could be considered bottlenecks and try to improve them.

Provide summary of your optimizations (no more than one page), along with
tables and or graphs to visually explain any performance differences.

* Include a breakdown of time spent in each pipeline stage for a few different
  models. It is suggested that you use pie charts or 100% stacked bar charts.
* For optimization steps (like backface culling), include a performance
  comparison to show the effectiveness.


## Submit

If you have modified any of the `CMakeLists.txt` files at all (aside from the
list of `SOURCE_FILES`), mentions it explicity.
Beware of any build issues discussed on the Google Group.

Open a GitHub pull request so that we can see that you have finished.
The title should be "Project 4: YOUR NAME".
The template of the comment section of your pull request is attached below, you can do some copy and paste:  

* [Repo Link](https://link-to-your-repo)
* `Your PENNKEY`
* (Briefly) Mentions features that you've completed. Especially those bells and whistles you want to highlight
    * Feature 0
    * Feature 1
    * ...
* Feedback on the project itself, if any.
