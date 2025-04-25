#include <string>

#include "./../include/shared.hpp"

namespace Shared{
    std::string toString(ImgObjType type){
        switch (type)
        {
            case ImgObjType::sugar_box: return "sugar_box";
            case ImgObjType::mustard_bottle: return "mustard_bottle";
            case ImgObjType::power_drill: return "power_drill";
        }
    }
}