#ifndef SHARED_HPP
#define SHARED_HPP

#include <string>
#include <opencv2/highgui.hpp>

namespace Shared{
    enum class ImgObjType{
        sugar_box,
        mustard_bottle,
        power_drill
    };

    std::string toString(ImgObjType type);
    std::string getFolderNameData(ImgObjType type);
}

#endif
