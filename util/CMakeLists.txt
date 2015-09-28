set(SOURCE_FILES
    "stb.cpp"
    "utilityCore.hpp"
    "utilityCore.cpp"
    "glslUtility.hpp"
    "glslUtility.cpp"
    "obj.hpp"
    "obj.cpp"
    "objloader.hpp"
    "objloader.cpp"
   )

cuda_add_library(util
   ${SOURCE_FILES}
   OPTIONS -arch=sm_20
   )
