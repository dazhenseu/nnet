cmake_minimum_required(VERSION 2.6)

project(nnet CXX)

option(ENABLE_AVX "Enables the use of AVX instructions." OFF)
option(ENABLE_FMA "Enables the use of FMA instructions." OFF)

if(ENABLE_AVX)
	add_definitions(-DHAVE_AVX -mavx)
endif()

if(ENABLE_FMA)
	add_definitions(-DHAVE_FMA -mfma)
endif()

add_definitions(-O3 -Wall)
add_definitions(-std=c++11)

include_directories(
	include
)

add_library(
	nnet
	source/core.cpp
	source/mnist.cpp
	source/cifar10.cpp
	source/stl10.cpp
	source/ActivationFunction.cpp
	source/Convolutional.cpp
	source/Dropout.cpp
	source/FeedForward.cpp
	source/FullyConnected.cpp
	source/Layer.cpp
	source/Loss.cpp
	source/MaxPool.cpp
	source/SlidingFullyConnected.cpp
	source/SpatialConvolutional.cpp
	source/UpdateRule.cpp
	source/vector.cpp
)

add_executable(
	mnistcnn
	examples/mnistcnn.cpp
	examples/evaluate.cpp
)

target_link_libraries(
	mnistcnn
	nnet
	fftw3f
)

add_executable(
	cifar10cnn
	examples/cifar10cnn.cpp
	examples/evaluate.cpp
)

target_link_libraries(
	cifar10cnn
	nnet
	fftw3f
)

configure_file(nnet-config.cmake.in nnet-config.cmake @ONLY)

install(TARGETS nnet DESTINATION ${CMAKE_INSTALL_PREFIX}/lib)
install(DIRECTORY include/nnet DESTINATION ${CMAKE_INSTALL_PREFIX}/include)
install(FILES "${PROJECT_BINARY_DIR}/nnet-config.cmake" DESTINATION ${CMAKE_INSTALL_PREFIX}/share/cmake/nnet)
