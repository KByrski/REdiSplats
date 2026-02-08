#include "shaders.cuh"

// *************************************************************************************************

extern "C" __constant__ LaunchParams<3> optixLaunchParams;

// *************************************************************************************************

extern "C" __global__ void __raygen__SH3() {
	__raygen__<3>();
}

extern "C" __global__ void __anyhit__SH3() {
	__anyhit__<3>();
}