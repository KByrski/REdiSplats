#include "shaders.cuh"

// *************************************************************************************************

extern "C" __constant__ LaunchParams<0> optixLaunchParams;

// *************************************************************************************************

extern "C" __global__ void __raygen__SH0() {
	__raygen__<0>();
}

extern "C" __global__ void __anyhit__SH0() {
	__anyhit__<0>();
}