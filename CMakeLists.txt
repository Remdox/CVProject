cmake_minimum_required(VERSION 2.8)
project(ProjectCV)
add_compile_options(-Wall -g)

find_package(OpenCV REQUIRED)

# Shared
add_library(shared src/shared.cpp)
target_include_directories(shared PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include/
    ${OpenCV_INCLUDE_DIRS}
)
target_link_libraries(shared PUBLIC ${OpenCV_LIBS})

# 1ST LIB
add_library(marco_annunziata src/marco_annunziata.cpp)
target_include_directories(marco_annunziata PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include/
    ${OpenCV_INCLUDE_DIRS}
)
target_link_libraries(marco_annunziata PUBLIC ${OpenCV_LIBS})

#2ND LIB
add_library(hermann_serain src/hermann_serain.cpp)
target_include_directories(hermann_serain PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include/
    ${OpenCV_INCLUDE_DIRS}
)
target_link_libraries(hermann_serain PUBLIC ${OpenCV_LIBS})

#3RD LIB


# EXE

add_executable(intermediateProject src/intermediateProject.cpp)
target_link_libraries(intermediateProject
    marco_annunziata
    hermann_serain
    shared #Lasciare shared per ultimo
    ${OpenCV_LIBS}
)
