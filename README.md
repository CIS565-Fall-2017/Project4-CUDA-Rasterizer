CUDA Rasterizer
===============

**University of Pennsylvania, CIS 565: GPU Programming and Architecture, Project 4**

* (TODO) YOUR NAME HERE
* Tested on: (TODO) Windows 22, i7-2222 @ 2.22GHz 22GB, GTX 222 222MB (Moore 2222 Lab)

### (TODO: Your README)

*DO NOT* leave the README to the last minute! It is a crucial part of the
project, and we will not be able to grade you without a good README.


Instructions (delete me)
========================

This is due Sunday, October 11, evening at midnight.

**Summary:** 
In this project, you will use CUDA to implement a simplified
rasterized graphics pipeline, similar to the OpenGL pipeline. You will
implement vertex shading, primitive assembly, rasterization, fragment shading,
and a framebuffer. More information about the rasterized graphics pipeline can
be found in the class slides and in the CIS 560 lecture notes.

The base code provided includes an OBJ loader and much of the I/O and
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
* `objs/` Example OBJ test files (# verts, # tris in buffers after loading)
  * `tri.obj` (3v, 1t): The simplest possible geometric object.
  * `cube.obj` (36v, 12t): A small model with low depth-complexity.
  * `suzanne.obj` (2904 verts, 968 tris): A medium model with low depth-complexity.
  * `suzanne_smooth.obj` (2904 verts, 968 tris): A medium model with low depth-complexity.
    This model has normals which must be interpolated.
  * `cow.obj` (17412 verts, 5804 tris): A large model with low depth-complexity.
  * `cow_smooth.obj` (17412 verts, 5804 tris): A large model with low depth-complexity.
    This model has normals which must be interpolated.
  * `flower.obj` (1920 verts, 640 tris): A medium model with very high depth-complexity.
  * `sponza.obj` (837,489 verts, 279,163 tris): A huge model with very high depth-complexity.
* `renders/` Debug render of an example OBJ.
* `external/` Includes and static libraries for 3rd party libraries.

### Running the code

The main function requires a scene description file. Call the program with
one as an argument: `cis565_rasterizer objs/cow.obj`.
(In Visual Studio, `../objs/cow.obj`.)

If you are using Visual Studio, you can set this in the Debugging > Command
Arguments section in the Project properties. Note that this value is different
for every different configuration type. Make sure you get the path right; read
the console for errors.

## Requirements

**Ask on the mailing list for any clarifications.**

In this project, you are given the following code:

* A library for loading standard Alias/Wavefront `.obj` format mesh
  files and converting them to OpenGL-style buffers of index and vertex data.
  * This library does NOT read materials, and provides all colors as white by
    default. You can use another library if you wish.
* Simple structs for some parts of the pipeline.
* Depth buffer to framebuffer copy.
* CUDA-GL interop.

You will need to implement the following features/pipeline stages:

* Vertex shading.
* (Vertex shader) perspective transformation.
* Primitive assembly with support for triangles read from buffers of index and
  vertex data.
* Rasterization.
* Fragment shading.
* A depth buffer for storing and depth testing fragments.
* Fragment to depth buffer writing (**with** atomics for race avoidance).
* (Fragment shader) simple lighting scheme, such as Lambert or Blinn-Phong.

See below for more guidance.

You are also required to implement at least "3.0" points in extra features.
(the parenthesized numbers must add to 3.0 or more):

* (1.0) Tile-based pipeline.
* Additional pipeline stages.
   * (1.0) Tessellation shader.
   * (1.0) Geometry shader, able to output a variable number of primitives per
     input primitive, optimized using stream compaction (thrust allowed).
   * (0.5 **if not doing geometry shader**) Backface culling, optimized using
     stream compaction (thrust allowed).
   * (1.0) Transform feedback.
   * (0.5) Scissor test.
   * (0.5) Blending (when writing into framebuffer).
* (1.0) Instancing: draw one set of vertex data multiple times, each run
  through the vertex shader with a different ID.
* (0.5) Correct color interpolation between points on a primitive.
* (1.0) UV texture mapping with bilinear texture filtering and perspective
  correct texture coordinates.
* Support for rasterizing additional primitives:
   * (0.5) Lines or line strips.
   * (0.5) Points.
* (1.0) Anti-aliasing.
* (1.0) Occlusion queries.
* (1.0) Order-independent translucency using a k-buffer.
* (0.5) **Mouse**-based interactive camera support.

This extra feature list is not comprehensive. If you have a particular idea
you would like to implement, please **contact us first**.

**IMPORTANT:**
For each extra feature, please provide the following brief analysis:

* Concise overview write-up of the feature.
* Performance impact of adding the feature (slower or faster).
* If you did something to accelerate the feature, what did you do and why?
* How might this feature be optimized beyond your current implementation?


## Base Code Tour

You will be working primarily in two files: `rasterize.cu`, and
`rasterizeTools.h`. Within these files, areas that you need to complete are
marked with a `TODO` comment. Areas that are useful to and serve as hints for
optional features are marked with `TODO (Optional)`. Functions that are useful
for reference are marked with the comment `CHECKITOUT`. **You should look at
all TODOs and CHECKITOUTs before starting!** There are not many.

* `src/rasterize.cu` contains the core rasterization pipeline. 
  * A few pre-made structs are included for you to use, but those marked with
    TODO will also be needed for a simple rasterizer. As with any part of the
    base code, you may modify or replace these as you see fit.

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

Start out by testing a single triangle (`tri.obj`).

* Clear the depth buffer with some default value.
* Vertex shading: 
  * `VertexIn[n] vs_input -> VertexOut[n] vs_output`
  * A minimal vertex shader will apply no transformations at all - it draws
    directly in normalized device coordinates (-1 to 1 in each dimension).
* Primitive assembly.
  * `VertexOut[n] vs_output -> Triangle[n/3] primitives`
  * Start by supporting ONLY triangles. For a triangle defined by indices
    `(a, b, c)` into `VertexOut` array `vo`, simply copy the appropriate values
    into a `Triangle` object `(vo[a], vo[b], vo[c])`.
* Rasterization.
  * `Triangle[n/3] primitives -> FragmentIn[m] fs_input`
  * A scanline implementation is simpler to start with.
* Fragment shading.
  * `FragmentIn[m] fs_input -> FragmentOut[m] fs_output`
  * A super-simple test fragment shader: output same color for every fragment.
    * Also try displaying various debug views (normals, etc.)
* Fragments to depth buffer.
  * `FragmentOut[m] -> FragmentOut[width][height]`
  * Results in race conditions - don't bother to fix these until it works!
  * Can really be done inside the fragment shader, if you call the fragment
    shader from the rasterization kernel for every fragment (including those
    which get occluded). **OR,** this can be done before fragment shading, which
    may be faster but means the fragment shader cannot change the depth.
* A depth buffer for storing and depth testing fragments.
  * `FragmentOut[width][height] depthbuffer`
  * An array of `fragment` objects.
  * At the end of a frame, it should contain the fragments drawn to the screen.
* Fragment to framebuffer writing.
  * `FragmentOut[width][height] depthbuffer -> vec3[width][height] framebuffer`
  * Simply copies the colors out of the depth buffer into the framebuffer
    (to be displayed on the screen).

### A Useful Pipeline

* Clear the depth buffer with some default value.
* Vertex shading: 
  * `VertexIn[n] vs_input -> VertexOut[n] vs_output`
  * Apply some vertex transformation (e.g. model-view-projection matrix using
    `glm::lookAt ` and `glm::perspective `).
* Primitive assembly.
  * `VertexOut[n] vs_output -> Triangle[n/3] primitives`
  * As above.
  * Other primitive types are optional.
* Rasterization.
  * `Triangle[n/3] primitives -> FragmentIn[m] fs_input`
  * You may choose to do a tiled rasterization method, which should have lower
    global memory bandwidth.
  * A scanline optimization: when rasterizing a triangle, only scan over the
    box around the triangle (`getAABBForTriangle`).
* Fragment shading.
  * `FragmentIn[m] fs_input -> FragmentOut[m] fs_output`
  * Add a shading method, such as Lambert or Blinn-Phong. Lights can be defined
    by kernel parameters (like GLSL uniforms).
* Fragments to depth buffer.
  * `FragmentOut[m] -> FragmentOut[width][height]`
  * Can really be done inside the fragment shader, if you call the fragment
    shader from the rasterization kernel for every fragment (including those
    which get occluded). **OR,** this can be done before fragment shading, which
    may be faster but means the fragment shader cannot change the depth.
    * This result in an optimization: it allows you to do depth tests before
     spending execution time in complex fragment shader code!
  * Handle race conditions! Since multiple primitives write fragments to the
    same fragment in the depth buffer, races must be avoided by using CUDA
    atomics.
    * *Approach 1:* Lock the location in the depth buffer during the time that
      a thread is comparing old and new fragment depths (and possibly writing
      a new fragment). This should work in all cases, but be slower.
    * *Approach 2:* Convert your depth value to a fixed-point `int`, and use
      `atomicMin` to store it into an `int`-typed depth buffer `intdepth`. After
      that, the value which is stored at `intdepth[i]` is (usually) that of the
      fragment which should be stored into the `fragment` depth buffer.
      * This may result in some rare race conditions (e.g. across blocks).
    * The `flower.obj` test file is good for testing race conditions.
* A depth buffer for storing and depth testing fragments.
  * `FragmentOut[width][height] depthbuffer`
  * An array of `fragment` objects.
  * At the end of a frame, it should contain the fragments drawn to the screen.
* Fragment to framebuffer writing.
  * `FragmentOut[width][height] depthbuffer -> vec3[width][height] framebuffer`
  * Simply copies the colors out of the depth buffer into the framebuffer
    (to be displayed on the screen).

This is a suggested sequence of pipeline steps, but you may choose to alter the
order of this sequence or merge entire kernels as you see fit.  For example, if
you decide that doing has benefits, you can choose to merge the vertex shader
and primitive assembly kernels, or merge the perspective transform into another
kernel. There is not necessarily a right sequence of kernels, and you may
choose any sequence that works.  Please document in your README what sequence
you choose and why.


## Resources

The following resources may be useful for this project:

* High-Performance Software Rasterization on GPUs:
  * [Paper (HPG 2011)](http://www.tml.tkk.fi/~samuli/publications/laine2011hpg_paper.pdf)
  * [Code](http://code.google.com/p/cudaraster/)
  * Note that looking over this code for reference with regard to the paper is
    fine, but we most likely will not grant any requests to actually
    incorporate any of this code into your project.
  * [Slides](http://bps11.idav.ucdavis.edu/talks/08-gpuSoftwareRasterLaineAndPantaleoni-BPS2011.pdf)
* The Direct3D 10 System (SIGGRAPH 2006) - for those interested in doing
  geometry shaders and transform feedback:
  * [Paper](http://dl.acm.org/citation.cfm?id=1141947)
  * [Paper, through Penn Libraries proxy](http://proxy.library.upenn.edu:2247/citation.cfm?id=1141947)
* Multi-Fragment Eﬀects on the GPU using the k-Buﬀer - for those who want to do
  order-independent transparency using a k-buffer:
  * [Paper](http://www.inf.ufrgs.br/~comba/papers/2007/kbuffer_preprint.pdf)
* FreePipe: A Programmable, Parallel Rendering Architecture for Efficient
  Multi-Fragment Effects (I3D 2010):
  * [Paper](https://sites.google.com/site/hmcen0921/cudarasterizer)
* Writing A Software Rasterizer In Javascript:
  * [Part 1](http://simonstechblog.blogspot.com/2012/04/software-rasterizer-part-1.html)
  * [Part 2](http://simonstechblog.blogspot.com/2012/04/software-rasterizer-part-2.html)


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

Replace the contents of this README.md in a clear manner with the following:

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
list of `SOURCE_FILES`), you must test that your project can build in Moore
100B/C. Beware of any build issues discussed on the Google Group.

1. Open a GitHub pull request so that we can see that you have finished.
   The title should be "Submission: YOUR NAME".
   * **ADDITIONALLY:**
     In the body of the pull request, include a link to your repository.
2. Send an email to the TA (gmail: kainino1+cis565@) with:
   * **Subject**: in the form of `[CIS565] Project N: PENNKEY`.
   * Direct link to your pull request on GitHub.
   * Estimate the amount of time you spent on the project.
   * If there were any outstanding problems, or if you did any extra
     work, *briefly* explain.
   * Feedback on the project itself, if any.
