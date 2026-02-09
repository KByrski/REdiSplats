#include "Header.cuh"

// *************************************************************************************************

bool GetSceneExtentOptiX(float &scene_extent_host) {
	cudaError_t error_CUDA;

	try {
		error_CUDA = cudaMemcpyFromSymbol(&scene_extent_host, scene_extent, sizeof(float));
		if (error_CUDA != cudaSuccess) throw 0;

		return true;
	} catch (...) {
		return false;
	}
}