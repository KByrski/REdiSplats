#include "Header.cuh"

// *************************************************************************************************

bool GetSceneExtentOptiX(float &scene_extent_host) {
	cudaError_t error_CUDA;

	error_CUDA = cudaMemcpyFromSymbol(&scene_extent_host, scene_extent, sizeof(float));
	if (error_CUDA != cudaSuccess) goto Error;

	return true;
Error:
	return false;
}