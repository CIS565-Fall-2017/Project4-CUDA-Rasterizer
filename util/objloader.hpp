/**
 * @file      objloader.hpp
 * @brief     An OBJ mesh loading library. Part of Yining Karl Li's OBJCORE.
 * @authors   Yining Karl Li
 * @date      2012
 * @copyright Yining Karl Li
 */

#pragma once

#include <stdlib.h>
#include "obj.hpp"

using namespace std;

class objLoader {
private:
    obj *geomesh;
public:
    objLoader(string, obj *);
    ~objLoader();

    //------------------------
    //-------GETTERS----------
    //------------------------

    obj *getMesh();
};
