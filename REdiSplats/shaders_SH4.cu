#include "shaders.cuh"

// *************************************************************************************************

extern "C" __constant__ LaunchParams<4> optixLaunchParams;

// *************************************************************************************************

extern "C" __global__ void __raygen__SH4() {
	__raygen__<4>();
}

extern "C" __global__ void __anyhit__SH4() {
	__anyhit__<4>();
}