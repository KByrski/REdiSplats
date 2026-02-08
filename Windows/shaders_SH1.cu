#include "shaders.cuh"

// *************************************************************************************************

extern "C" __constant__ LaunchParams<1> optixLaunchParams;

// *************************************************************************************************

extern "C" __global__ void __raygen__SH1() {
	__raygen__<1>();
}

extern "C" __global__ void __anyhit__SH1() {
	__anyhit__<1>();
}