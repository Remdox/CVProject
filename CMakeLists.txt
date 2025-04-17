cmake_minimum_required(VERSION 2.8)
project(ProjectCV)
add_compile_options(-Wall -g)

find_package(OpenCV REQUIRED)

# 1ST LIB
add_library(marco_annunziata src/marco_annunziata.cpp)
target_include_directories(marco_annunziata PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include/
    ${OpenCV_INCLUDE_DIRS}
)
target_link_libraries(marco_annunziata PUBLIC ${OpenCV_LIBS})

#2ND LIB


#3RD LIB


# EXE

add_executable(intermediateProject src/intermediateProject.cpp)
target_link_libraries(intermediateProject
    marco_annunziata
    # ...
    ${OpenCV_LIBS}
)
