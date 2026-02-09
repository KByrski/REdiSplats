#include "shaders.cuh"

// *************************************************************************************************

extern "C" __global__ void __raygen__SH2() {
	__raygen__<2>();
}

extern "C" __global__ void __anyhit__SH2() {
	__anyhit__<2>();
}