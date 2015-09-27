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

This is due **INSTRUCTOR TODO** evening at midnight.

**Summary:** **INSTRUCTOR TODO**

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

**INSTRUCTOR TODO:** update according to any code changes. ADD SPONZA.

* `src/` C++/CUDA source files.
* `util/` C++ utility files.
* `objs/` Example OBJ test files: `tri.obj`, `cube.obj`, `cow.obj`, `sponza.obj`.
* `img/` Renders of example OBJs.
  (These probably won't match precisely with yours.)
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

In this project, you are given code for:

* A library for loading/reading standard Alias/Wavefront `.obj` format mesh
  files and converting them to OpenGL-style vertex and index buffers
* A suggested order of kernels with which to implement the graphics pipeline
* CUDA-GL interop

You will need to implement the following features/pipeline stages:

* Vertex shading.
* (Vertex shader) perspective transformation.
* Primitive assembly with support for triangle vertex and index buffers.
* Rasterization: **either** a scanline or a tiled approach.
* Fragment shading.
* A depth buffer for storing and depth testing fragments.
* Fragment to depth buffer writing (**with** atomics for race avoidance).
* (Fragment shader) simple lighting scheme, such as Lambert or Blinn-Phong.

See below for more guidance.

You are also required to implement at least "3.0" of the following features.
(the parenthesized numbers must add to 3.0 or more):

* Additional pipeline stages.
   * (1.0) Tessellation shader.
   * (1.0) Geometry shader.
   * (1.0) Transform feedback.
   * (0.5) Back-face culling with stream compaction.
   * (0.5) Scissor test.
   * (0.5) Blending.
* (1.0) Instancing
* (0.5) Correct color interpolation between points on a primitive.
* (1.0) UV texture mapping with bilinear texture filtering and perspective correct texture coordinates.
* Support for rasterizing additional primitives:
   * (0.5) Lines or line strips.
   * (0.5) Points.
* (1.0) Anti-aliasing
* (1.0) Occlusion queries
* (1.0) Order-independent translucency using a k-buffer
* (0.5) **Mouse**-based interactive camera support.

This extra feature list is not comprehensive. If you have a particular idea
you would like to implement, please **contact us first**.

**IMPORTANT:**
For each extra feature, please provide the following analysis:

* Concise overview write-up of the feature.
* Performance impact of adding the feature.
* If you did something to accelerate the feature, what did you do and why?
* How might this feature be optimized beyond your current implementation?


## Rasterization Pipeline

**INSTRUCTOR TODO**: update README to explain a minimal pipeline to see a
triangle, e.g., no depth test, draw in NDC, etc.

Possible pipelines are described below. Pseudo-type-signatures are given.
Not all of the pseudocode arrays will necessarily actually exist in practice.

### Minimal Pipeline

This describes a minimal version *one possible* graphics pipeline, similar to
modern hardware (DX/OpenGL). Yours need not match precisely.  To begin, try to
write a minimal amount of code as described here. This will reduce the
necessary time spent debugging.

* Vertex shading: 
  * `VertexIn[n] vs_input -> VertexOut[n] vs_output`
  * A minimal vertex shader will apply no transformations at all - it draws
    directly in normalized device coordinates (NDC).
* Primitive assembly.
  * `vertexOut[n] vs_output -> triangle[n/3] primitives`
  * Start by supporting ONLY triangles.
* Rasterization.
  * `triangle[n/3] primitives -> fragmentIn[m] fs_input`
  * Scanline: TODO
  * Tiled: TODO
* Fragment shading.
  * `fragmentIn[m] fs_input -> fragmentOut[m] fs_output`
  * A super-simple test fragment shader: output same color for every fragment.
    * Also try displaying various debug views (normals, etc.)
* Fragments to depth buffer.
  * `fragmentOut[m] -> fragmentOut[resolution]`
  * Can really be done inside the fragment shader.
  * Results in race conditions - don't bother to fix these until it works!
* A depth buffer for storing and depth testing fragments.
  * `fragmentOut[resolution] depthbuffer`
  * An array of `fragment` objects.
  * At the end of a frame, it should contain the fragments drawn to the screen.
* Fragment to framebuffer writing.
  * `fragmentOut[resolution] depthbuffer -> vec3[resolution] framebuffer`
  * Simply copies the colors out of the depth buffer into the framebuffer
    (to be displayed on the screen).

### Better Pipeline

INSTRUCTOR TODO

* Rasterization.
  * Scanline:
    * Optimization: scissor around rasterized triangle

* Fragments to depth buffer.
  * `fragmentOut[m] -> fragmentOut[resolution]`
  * Can really be done inside the fragment shader.
    * This allows you to do depth tests before spending execution time in
      complex fragment shader code.
  * When writing to the depth buffer, you will need to use atomics for race
    avoidance, to prevent different primitives from overwriting each other in
    the wrong order.


## Base Code Tour

**INSTRUCTOR TODO:** update according to any code changes.
TODO: simple structs for every part of the pipeline, intended to be changed?
(e.g. vertexPre, vertexPost, triangle = vertexPre[3], fragment).
TODO: autoformat code
TODO: pragma once
TODO: doxygen

You will be working primarily in two files: `rasterize.cu`, and
`rasterizeTools.h`. Within these files, areas that you need to complete are
marked with a `TODO` comment. Areas that are useful to and serve as hints for
optional features are marked with `TODO (Optional)`. Functions that are useful
for reference are marked with the comment `CHECKITOUT`.

* `src/rasterize.cu` contains the core rasterization pipeline. 
  * A suggested sequence of kernels exists in this file, but you may choose to
    alter the order of this sequence or merge entire kernels if you see fit.
    For example, if you decide that doing has benefits, you can choose to merge
    the vertex shader and primitive assembly kernels, or merge the perspective
    transform into another kernel. There is not necessarily a right sequence of
    kernels (although there are wrong sequences, such as placing fragment
    shading before vertex shading), and you may choose any sequence you want.
    Please document in your README what sequence you choose and why.
  * The provided kernels have had their input parameters removed beyond basic
    inputs such as the framebuffer. You will have to decide what inputs should
    go into each stage of the pipeline, and what outputs there should be. 

* `src/rasterizeTools.h` contains various useful tools, including a number of
  barycentric coordinate related functions that you may find useful in
  implementing scanline based rasterization...
  * A few pre-made structs are included for you to use, such as fragment and
    triangle. A simple rasterizer can be implemented with these structs as is.
    However, as with any part of the basecode, you may choose to modify, add
    to, use as-is, or outright ignore them as you see fit.
  * If you do choose to add to the fragment struct, be sure to include in your
    README a rationale for why. 

You will also want to familiarize yourself with:

* `src/main.cpp`, which contains code that transfers VBOs/CBOs/IBOs to the
  rasterization pipeline. Interactive camera work will also involve this file
  if you choose that feature.
* `util/utilityCore.h`, which serves as a kitchen-sink of useful functions


## Resources

**INSTRUCTOR TODO:** make sure these links work

The following resources may be useful for this project:

* High-Performance Software Rasterization on GPUs:
  * Paper (HPG 2011):
    http://www.tml.tkk.fi/~samuli/publications/laine2011hpg_paper.pdf
  * Code: http://code.google.com/p/cudaraster/
  * Note that looking over this code for reference with regard to the paper is
    fine, but we most likely will not grant any requests to actually
    incorporate any of this code into your project.
  * Slides:
    http://bps11.idav.ucdavis.edu/talks/08-gpuSoftwareRasterLaineAndPantaleoni-BPS2011.pdf
* The Direct3D 10 System (SIGGRAPH 2006) - for those interested in doing
  geometry shaders and transform feedback:
  * http://133.11.9.3/~takeo/course/2006/media/papers/Direct3D10_siggraph2006.pdf
* Multi-Fragment Eﬀects on the GPU using the k-Buﬀer - for those who want to do
  order-independent transparency using a k-buffer:
  * http://www.inf.ufrgs.br/~comba/papers/2007/kbuffer_preprint.pdf
* FreePipe: A Programmable, Parallel Rendering Architecture for Efficient
  Multi-Fragment Effects (I3D 2010):
  * https://sites.google.com/site/hmcen0921/cudarasterizer
* Writing A Software Rasterizer In Javascript:
  * Part 1:
    http://simonstechblog.blogspot.com/2012/04/software-rasterizer-part-1.html
  * Part 2:
    http://simonstechblog.blogspot.com/2012/04/software-rasterizer-part-2.html


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

Provide no more than a one page summary of your optimizations along with tables
and or graphs to visually explain any performance differences.

**INSTRUCTOR TODO**: require stage-by-stage performance analysis like this -
https://github.com/takfuruya/Project4-Rasterizer


## Submit

If you have modified any of the `CMakeLists.txt` files at all (aside from the
list of `SOURCE_FILES`), you must test that your project can build in Moore
100B/C. Beware of any build issues discussed on the Google Group.

1. Open a GitHub pull request so that we can see that you have finished.
   The title should be "Submission: YOUR NAME".
2. Send an email to the TA (gmail: kainino1+cis565@) with:
   * **Subject**: in the form of `[CIS565] Project N: PENNKEY`.
   * Direct link to your pull request on GitHub.
   * Estimate the amount of time you spent on the project.
   * If there were any outstanding problems, or if you did any extra
     work, *briefly* explain.
   * Feedback on the project itself, if any.
