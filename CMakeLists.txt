cmake_minimum_required(VERSION 2.8)
project(DisplayImage)
add_compile_options(-Wall -g)

# Trova libreria esterna pre-compilata che poi aggiungiamo al linking con target_link_libraries
find_package(OpenCV REQUIRED)
# Puoi cercare il nome del package da mettere nell'istruzione sul sito di chi ha fatto la libreria. Cerca ad esempio "Using OpenCV with CMake".

# TASK1 (solo OPENCV)
add_executable(intermediateProject intermediateProject.cpp)
target_include_directories(intermediateProject PRIVATE ${OpenCV_INCLUDE_DIRS})
target_link_libraries(intermediateProject ${OpenCV_LIBS})
