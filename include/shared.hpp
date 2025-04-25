#ifndef SHARED_HPP
#define SHARED_HPP

#include <string>

namespace Shared{
    enum class ImgObjType{
        sugar_box,
        mustard_bottle,
        power_drill
    };

    std::string toString(ImgObjType type);
}

#endif