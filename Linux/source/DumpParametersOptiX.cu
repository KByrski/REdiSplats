#include "Header.cuh"

// *************************************************************************************************

static void DumpToFile(const char *dirPath, int epochNum, const char *fName, void *buf, int size) {
	FILE *f;

	char fPath[256];
	sprintf(fPath, "%s/%s_iter_%d.checkpoint", dirPath, fName, epochNum);

	f = fopen(fPath, "wb");
	fwrite(buf, size, 1, f);
	fclose(f);
}

// *************************************************************************************************

template<int SH_degree>
bool DumpParametersOptiX(SOptiXRenderParams<SH_degree> &params_OptiX, char *dirPath) {
	cudaError_t error_CUDA;

	try {
		void *buf = malloc(sizeof(REAL4_G) * params_OptiX.numberOfGaussians);
		if (buf == NULL) throw 0;

		// *** *** *** *** ***

		error_CUDA = cudaMemcpy(buf, params_OptiX.GC_part_1_1, sizeof(float4) * params_OptiX.numberOfGaussians, cudaMemcpyDeviceToHost);
		if (error_CUDA != cudaSuccess) throw 0;
		DumpToFile(dirPath, params_OptiX.epoch, "gc1", buf, sizeof(float4) * params_OptiX.numberOfGaussians);

		error_CUDA = cudaMemcpy(buf, params_OptiX.GC_part_2_1, sizeof(float4) * params_OptiX.numberOfGaussians, cudaMemcpyDeviceToHost);
		if (error_CUDA != cudaSuccess) throw 0;
		DumpToFile(dirPath, params_OptiX.epoch, "gc2", buf, sizeof(float4) * params_OptiX.numberOfGaussians);

		error_CUDA = cudaMemcpy(buf, params_OptiX.GC_part_3_1, sizeof(float4) * params_OptiX.numberOfGaussians, cudaMemcpyDeviceToHost);
		if (error_CUDA != cudaSuccess) throw 0;
		DumpToFile(dirPath, params_OptiX.epoch, "gc3", buf, sizeof(float4) * params_OptiX.numberOfGaussians);

		// !!! !!! !!!
		error_CUDA = cudaMemcpy(buf, params_OptiX.GC_part_4_1, sizeof(float2) * params_OptiX.numberOfGaussians, cudaMemcpyDeviceToHost);
		if (error_CUDA != cudaSuccess) throw 0;
		DumpToFile(dirPath, params_OptiX.epoch, "gc4", buf, sizeof(float2) * params_OptiX.numberOfGaussians);

		// *** *** *** *** ***

		// Spherical harmonics
		if constexpr (SH_degree >= 1) {
			error_CUDA = cudaMemcpy(buf, params_OptiX.GC_SH_1, sizeof(float4) * params_OptiX.numberOfGaussians, cudaMemcpyDeviceToHost);
			if (error_CUDA != cudaSuccess) throw 0;
			DumpToFile(dirPath, params_OptiX.epoch, "gc_sh_1", buf, sizeof(float4) * params_OptiX.numberOfGaussians);

			error_CUDA = cudaMemcpy(buf, params_OptiX.GC_SH_2, sizeof(float4) * params_OptiX.numberOfGaussians, cudaMemcpyDeviceToHost);
			if (error_CUDA != cudaSuccess) throw 0;
			DumpToFile(dirPath, params_OptiX.epoch, "gc_sh_2", buf, sizeof(float4) * params_OptiX.numberOfGaussians);

			if constexpr (SH_degree >= 2) {
				error_CUDA = cudaMemcpy(buf, params_OptiX.GC_SH_3, sizeof(float4) * params_OptiX.numberOfGaussians, cudaMemcpyDeviceToHost);
				if (error_CUDA != cudaSuccess) throw 0;
				DumpToFile(dirPath, params_OptiX.epoch, "gc_sh_3", buf, sizeof(float4) * params_OptiX.numberOfGaussians);

				error_CUDA = cudaMemcpy(buf, params_OptiX.GC_SH_4, sizeof(float4) * params_OptiX.numberOfGaussians, cudaMemcpyDeviceToHost);
				if (error_CUDA != cudaSuccess) throw 0;
				DumpToFile(dirPath, params_OptiX.epoch, "gc_sh_4", buf, sizeof(float4) * params_OptiX.numberOfGaussians);

				error_CUDA = cudaMemcpy(buf, params_OptiX.GC_SH_5, sizeof(float4) * params_OptiX.numberOfGaussians, cudaMemcpyDeviceToHost);
				if (error_CUDA != cudaSuccess) throw 0;
				DumpToFile(dirPath, params_OptiX.epoch, "gc_sh_5", buf, sizeof(float4) * params_OptiX.numberOfGaussians);

				error_CUDA = cudaMemcpy(buf, params_OptiX.GC_SH_6, sizeof(float4) * params_OptiX.numberOfGaussians, cudaMemcpyDeviceToHost);
				if (error_CUDA != cudaSuccess) throw 0;
				DumpToFile(dirPath, params_OptiX.epoch, "gc_sh_6", buf, sizeof(float4) * params_OptiX.numberOfGaussians);

				if constexpr (SH_degree >= 3) {
					error_CUDA = cudaMemcpy(buf, params_OptiX.GC_SH_7, sizeof(float4) * params_OptiX.numberOfGaussians, cudaMemcpyDeviceToHost);
					if (error_CUDA != cudaSuccess) throw 0;
					DumpToFile(dirPath, params_OptiX.epoch, "gc_sh_7", buf, sizeof(float4) * params_OptiX.numberOfGaussians);

					error_CUDA = cudaMemcpy(buf, params_OptiX.GC_SH_8, sizeof(float4) * params_OptiX.numberOfGaussians, cudaMemcpyDeviceToHost);
					if (error_CUDA != cudaSuccess) throw 0;
					DumpToFile(dirPath, params_OptiX.epoch, "gc_sh_8", buf, sizeof(float4) * params_OptiX.numberOfGaussians);

					error_CUDA = cudaMemcpy(buf, params_OptiX.GC_SH_9, sizeof(float4) * params_OptiX.numberOfGaussians, cudaMemcpyDeviceToHost);
					if (error_CUDA != cudaSuccess) throw 0;
					DumpToFile(dirPath, params_OptiX.epoch, "gc_sh_9", buf, sizeof(float4) * params_OptiX.numberOfGaussians);

					error_CUDA = cudaMemcpy(buf, params_OptiX.GC_SH_10, sizeof(float4) * params_OptiX.numberOfGaussians, cudaMemcpyDeviceToHost);
					if (error_CUDA != cudaSuccess) throw 0;
					DumpToFile(dirPath, params_OptiX.epoch, "gc_sh_10", buf, sizeof(float4) * params_OptiX.numberOfGaussians);

					error_CUDA = cudaMemcpy(buf, params_OptiX.GC_SH_11, sizeof(float4) * params_OptiX.numberOfGaussians, cudaMemcpyDeviceToHost);
					if (error_CUDA != cudaSuccess) throw 0;
					DumpToFile(dirPath, params_OptiX.epoch, "gc_sh_11", buf, sizeof(float4) * params_OptiX.numberOfGaussians);

					if constexpr (SH_degree >= 4) {
						error_CUDA = cudaMemcpy(buf, params_OptiX.GC_SH_12, sizeof(float4) * params_OptiX.numberOfGaussians, cudaMemcpyDeviceToHost);
						if (error_CUDA != cudaSuccess) throw 0;
						DumpToFile(dirPath, params_OptiX.epoch, "gc_sh_12", buf, sizeof(float4) * params_OptiX.numberOfGaussians);

						error_CUDA = cudaMemcpy(buf, params_OptiX.GC_SH_13, sizeof(float4) * params_OptiX.numberOfGaussians, cudaMemcpyDeviceToHost);
						if (error_CUDA != cudaSuccess) throw 0;
						DumpToFile(dirPath, params_OptiX.epoch, "gc_sh_13", buf, sizeof(float4) * params_OptiX.numberOfGaussians);

						error_CUDA = cudaMemcpy(buf, params_OptiX.GC_SH_14, sizeof(float4) * params_OptiX.numberOfGaussians, cudaMemcpyDeviceToHost);
						if (error_CUDA != cudaSuccess) throw 0;
						DumpToFile(dirPath, params_OptiX.epoch, "gc_sh_14", buf, sizeof(float4) * params_OptiX.numberOfGaussians);

						error_CUDA = cudaMemcpy(buf, params_OptiX.GC_SH_15, sizeof(float4) * params_OptiX.numberOfGaussians, cudaMemcpyDeviceToHost);
						if (error_CUDA != cudaSuccess) throw 0;
						DumpToFile(dirPath, params_OptiX.epoch, "gc_sh_15", buf, sizeof(float4) * params_OptiX.numberOfGaussians);

						error_CUDA = cudaMemcpy(buf, params_OptiX.GC_SH_16, sizeof(float4) * params_OptiX.numberOfGaussians, cudaMemcpyDeviceToHost);
						if (error_CUDA != cudaSuccess) throw 0;
						DumpToFile(dirPath, params_OptiX.epoch, "gc_sh_16", buf, sizeof(float4) * params_OptiX.numberOfGaussians);

						error_CUDA = cudaMemcpy(buf, params_OptiX.GC_SH_17, sizeof(float4) * params_OptiX.numberOfGaussians, cudaMemcpyDeviceToHost);
						if (error_CUDA != cudaSuccess) throw 0;
						DumpToFile(dirPath, params_OptiX.epoch, "gc_sh_17", buf, sizeof(float4) * params_OptiX.numberOfGaussians);

						error_CUDA = cudaMemcpy(buf, params_OptiX.GC_SH_18, sizeof(float4) * params_OptiX.numberOfGaussians, cudaMemcpyDeviceToHost);
						if (error_CUDA != cudaSuccess) throw 0;
						DumpToFile(dirPath, params_OptiX.epoch, "gc_sh_18", buf, sizeof(float4) * params_OptiX.numberOfGaussians);
					} else {
						// !!! !!! !!!
						error_CUDA = cudaMemcpy(buf, params_OptiX.GC_SH_12, sizeof(float) * params_OptiX.numberOfGaussians, cudaMemcpyDeviceToHost);
						if (error_CUDA != cudaSuccess) throw 0;
						DumpToFile(dirPath, params_OptiX.epoch, "gc_sh_12", buf, sizeof(float) * params_OptiX.numberOfGaussians);
					}
				}
			} else {
				// !!! !!! !!!
				error_CUDA = cudaMemcpy(buf, params_OptiX.GC_SH_3, sizeof(float) * params_OptiX.numberOfGaussians, cudaMemcpyDeviceToHost);
				if (error_CUDA != cudaSuccess) throw 0;
				DumpToFile(dirPath, params_OptiX.epoch, "gc_sh_3", buf, sizeof(float) * params_OptiX.numberOfGaussians);
			}
		}

		// *** *** *** *** ***

		error_CUDA = cudaMemcpy(buf, params_OptiX.m11, sizeof(float4) * params_OptiX.numberOfGaussians, cudaMemcpyDeviceToHost);
		if (error_CUDA != cudaSuccess) throw 0;
		DumpToFile(dirPath, params_OptiX.epoch, "m1", buf, sizeof(float4) * params_OptiX.numberOfGaussians);

		error_CUDA = cudaMemcpy(buf, params_OptiX.m21, sizeof(float4) * params_OptiX.numberOfGaussians, cudaMemcpyDeviceToHost);
		if (error_CUDA != cudaSuccess) throw 0;
		DumpToFile(dirPath, params_OptiX.epoch, "m2", buf, sizeof(float4) * params_OptiX.numberOfGaussians);

		error_CUDA = cudaMemcpy(buf, params_OptiX.m31, sizeof(float4) * params_OptiX.numberOfGaussians, cudaMemcpyDeviceToHost);
		if (error_CUDA != cudaSuccess) throw 0;
		DumpToFile(dirPath, params_OptiX.epoch, "m3", buf, sizeof(float4) * params_OptiX.numberOfGaussians);

		// !!! !!! !!!
		error_CUDA = cudaMemcpy(buf, params_OptiX.m41, sizeof(float2) * params_OptiX.numberOfGaussians, cudaMemcpyDeviceToHost);
		if (error_CUDA != cudaSuccess) throw 0;
		DumpToFile(dirPath, params_OptiX.epoch, "m4", buf, sizeof(float2) * params_OptiX.numberOfGaussians);

		// *** *** *** *** ***

		// Spherical harmonics
		if constexpr (SH_degree >= 1) {
			error_CUDA = cudaMemcpy(buf, params_OptiX.m_SH_1, sizeof(float4) * params_OptiX.numberOfGaussians, cudaMemcpyDeviceToHost);
			if (error_CUDA != cudaSuccess) throw 0;
			DumpToFile(dirPath, params_OptiX.epoch, "m_sh_1", buf, sizeof(float4) * params_OptiX.numberOfGaussians);

			error_CUDA = cudaMemcpy(buf, params_OptiX.m_SH_2, sizeof(float4) * params_OptiX.numberOfGaussians, cudaMemcpyDeviceToHost);
			if (error_CUDA != cudaSuccess) throw 0;
			DumpToFile(dirPath, params_OptiX.epoch, "m_sh_2", buf, sizeof(float4) * params_OptiX.numberOfGaussians);

			if constexpr (SH_degree >= 2) {
				error_CUDA = cudaMemcpy(buf, params_OptiX.m_SH_3, sizeof(float4) * params_OptiX.numberOfGaussians, cudaMemcpyDeviceToHost);
				if (error_CUDA != cudaSuccess) throw 0;
				DumpToFile(dirPath, params_OptiX.epoch, "m_sh_3", buf, sizeof(float4) * params_OptiX.numberOfGaussians);

				error_CUDA = cudaMemcpy(buf, params_OptiX.m_SH_4, sizeof(float4) * params_OptiX.numberOfGaussians, cudaMemcpyDeviceToHost);
				if (error_CUDA != cudaSuccess) throw 0;
				DumpToFile(dirPath, params_OptiX.epoch, "m_sh_4", buf, sizeof(float4) * params_OptiX.numberOfGaussians);

				error_CUDA = cudaMemcpy(buf, params_OptiX.m_SH_5, sizeof(float4) * params_OptiX.numberOfGaussians, cudaMemcpyDeviceToHost);
				if (error_CUDA != cudaSuccess) throw 0;
				DumpToFile(dirPath, params_OptiX.epoch, "m_sh_5", buf, sizeof(float4) * params_OptiX.numberOfGaussians);

				error_CUDA = cudaMemcpy(buf, params_OptiX.m_SH_6, sizeof(float4) * params_OptiX.numberOfGaussians, cudaMemcpyDeviceToHost);
				if (error_CUDA != cudaSuccess) throw 0;
				DumpToFile(dirPath, params_OptiX.epoch, "m_sh_6", buf, sizeof(float4) * params_OptiX.numberOfGaussians);

				if constexpr (SH_degree >= 3) {
					error_CUDA = cudaMemcpy(buf, params_OptiX.m_SH_7, sizeof(float4) * params_OptiX.numberOfGaussians, cudaMemcpyDeviceToHost);
					if (error_CUDA != cudaSuccess) throw 0;
					DumpToFile(dirPath, params_OptiX.epoch, "m_sh_7", buf, sizeof(float4) * params_OptiX.numberOfGaussians);

					error_CUDA = cudaMemcpy(buf, params_OptiX.m_SH_8, sizeof(float4) * params_OptiX.numberOfGaussians, cudaMemcpyDeviceToHost);
					if (error_CUDA != cudaSuccess) throw 0;
					DumpToFile(dirPath, params_OptiX.epoch, "m_sh_8", buf, sizeof(float4) * params_OptiX.numberOfGaussians);

					error_CUDA = cudaMemcpy(buf, params_OptiX.m_SH_9, sizeof(float4) * params_OptiX.numberOfGaussians, cudaMemcpyDeviceToHost);
					if (error_CUDA != cudaSuccess) throw 0;
					DumpToFile(dirPath, params_OptiX.epoch, "m_sh_9", buf, sizeof(float4) * params_OptiX.numberOfGaussians);

					error_CUDA = cudaMemcpy(buf, params_OptiX.m_SH_10, sizeof(float4) * params_OptiX.numberOfGaussians, cudaMemcpyDeviceToHost);
					if (error_CUDA != cudaSuccess) throw 0;
					DumpToFile(dirPath, params_OptiX.epoch, "m_sh_10", buf, sizeof(float4) * params_OptiX.numberOfGaussians);

					error_CUDA = cudaMemcpy(buf, params_OptiX.m_SH_11, sizeof(float4) * params_OptiX.numberOfGaussians, cudaMemcpyDeviceToHost);
					if (error_CUDA != cudaSuccess) throw 0;
					DumpToFile(dirPath, params_OptiX.epoch, "m_sh_11", buf, sizeof(float4) * params_OptiX.numberOfGaussians);

					if constexpr (SH_degree >= 4) {
						error_CUDA = cudaMemcpy(buf, params_OptiX.m_SH_12, sizeof(float4) * params_OptiX.numberOfGaussians, cudaMemcpyDeviceToHost);
						if (error_CUDA != cudaSuccess) throw 0;
						DumpToFile(dirPath, params_OptiX.epoch, "m_sh_12", buf, sizeof(float4) * params_OptiX.numberOfGaussians);

						error_CUDA = cudaMemcpy(buf, params_OptiX.m_SH_13, sizeof(float4) * params_OptiX.numberOfGaussians, cudaMemcpyDeviceToHost);
						if (error_CUDA != cudaSuccess) throw 0;
						DumpToFile(dirPath, params_OptiX.epoch, "m_sh_13", buf, sizeof(float4) * params_OptiX.numberOfGaussians);

						error_CUDA = cudaMemcpy(buf, params_OptiX.m_SH_14, sizeof(float4) * params_OptiX.numberOfGaussians, cudaMemcpyDeviceToHost);
						if (error_CUDA != cudaSuccess) throw 0;
						DumpToFile(dirPath, params_OptiX.epoch, "m_sh_14", buf, sizeof(float4) * params_OptiX.numberOfGaussians);

						error_CUDA = cudaMemcpy(buf, params_OptiX.m_SH_15, sizeof(float4) * params_OptiX.numberOfGaussians, cudaMemcpyDeviceToHost);
						if (error_CUDA != cudaSuccess) throw 0;
						DumpToFile(dirPath, params_OptiX.epoch, "m_sh_15", buf, sizeof(float4) * params_OptiX.numberOfGaussians);

						error_CUDA = cudaMemcpy(buf, params_OptiX.m_SH_16, sizeof(float4) * params_OptiX.numberOfGaussians, cudaMemcpyDeviceToHost);
						if (error_CUDA != cudaSuccess) throw 0;
						DumpToFile(dirPath, params_OptiX.epoch, "m_sh_16", buf, sizeof(float4) * params_OptiX.numberOfGaussians);

						error_CUDA = cudaMemcpy(buf, params_OptiX.m_SH_17, sizeof(float4) * params_OptiX.numberOfGaussians, cudaMemcpyDeviceToHost);
						if (error_CUDA != cudaSuccess) throw 0;
						DumpToFile(dirPath, params_OptiX.epoch, "m_sh_17", buf, sizeof(float4) * params_OptiX.numberOfGaussians);

						error_CUDA = cudaMemcpy(buf, params_OptiX.m_SH_18, sizeof(float4) * params_OptiX.numberOfGaussians, cudaMemcpyDeviceToHost);
						if (error_CUDA != cudaSuccess) throw 0;
						DumpToFile(dirPath, params_OptiX.epoch, "m_sh_18", buf, sizeof(float4) * params_OptiX.numberOfGaussians);
					} else {
						// !!! !!! !!!
						error_CUDA = cudaMemcpy(buf, params_OptiX.m_SH_12, sizeof(float) * params_OptiX.numberOfGaussians, cudaMemcpyDeviceToHost);
						if (error_CUDA != cudaSuccess) throw 0;
						DumpToFile(dirPath, params_OptiX.epoch, "m_sh_12", buf, sizeof(float) * params_OptiX.numberOfGaussians);
					}
				}
			} else {
				// !!! !!! !!!
				error_CUDA = cudaMemcpy(buf, params_OptiX.m_SH_3, sizeof(float) * params_OptiX.numberOfGaussians, cudaMemcpyDeviceToHost);
				if (error_CUDA != cudaSuccess) throw 0;
				DumpToFile(dirPath, params_OptiX.epoch, "m_sh_3", buf, sizeof(float) * params_OptiX.numberOfGaussians);
			}
		}

		// *** *** *** *** ***

		error_CUDA = cudaMemcpy(buf, params_OptiX.v11, sizeof(float4) * params_OptiX.numberOfGaussians, cudaMemcpyDeviceToHost);
		if (error_CUDA != cudaSuccess) throw 0;
		DumpToFile(dirPath, params_OptiX.epoch, "v1", buf, sizeof(float4) * params_OptiX.numberOfGaussians);

		error_CUDA = cudaMemcpy(buf, params_OptiX.v21, sizeof(float4) * params_OptiX.numberOfGaussians, cudaMemcpyDeviceToHost);
		if (error_CUDA != cudaSuccess) throw 0;
		DumpToFile(dirPath, params_OptiX.epoch, "v2", buf, sizeof(float4) * params_OptiX.numberOfGaussians);

		error_CUDA = cudaMemcpy(buf, params_OptiX.v31, sizeof(float4) * params_OptiX.numberOfGaussians, cudaMemcpyDeviceToHost);
		if (error_CUDA != cudaSuccess) throw 0;
		DumpToFile(dirPath, params_OptiX.epoch, "v3", buf, sizeof(float4) * params_OptiX.numberOfGaussians);

		// !!! !!! !!!
		error_CUDA = cudaMemcpy(buf, params_OptiX.v41, sizeof(float2) * params_OptiX.numberOfGaussians, cudaMemcpyDeviceToHost);
		if (error_CUDA != cudaSuccess) throw 0;
		DumpToFile(dirPath, params_OptiX.epoch, "v4", buf, sizeof(float2) * params_OptiX.numberOfGaussians);

		// *** *** *** *** ***

		// Spherical harmonics
		if constexpr (SH_degree >= 1) {
			error_CUDA = cudaMemcpy(buf, params_OptiX.v_SH_1, sizeof(float4) * params_OptiX.numberOfGaussians, cudaMemcpyDeviceToHost);
			if (error_CUDA != cudaSuccess) throw 0;
			DumpToFile(dirPath, params_OptiX.epoch, "v_sh_1", buf, sizeof(float4) * params_OptiX.numberOfGaussians);

			error_CUDA = cudaMemcpy(buf, params_OptiX.v_SH_2, sizeof(float4) * params_OptiX.numberOfGaussians, cudaMemcpyDeviceToHost);
			if (error_CUDA != cudaSuccess) throw 0;
			DumpToFile(dirPath, params_OptiX.epoch, "v_sh_2", buf, sizeof(float4) * params_OptiX.numberOfGaussians);

			if constexpr (SH_degree >= 2) {
				error_CUDA = cudaMemcpy(buf, params_OptiX.v_SH_3, sizeof(float4) * params_OptiX.numberOfGaussians, cudaMemcpyDeviceToHost);
				if (error_CUDA != cudaSuccess) throw 0;
				DumpToFile(dirPath, params_OptiX.epoch, "v_sh_3", buf, sizeof(float4) * params_OptiX.numberOfGaussians);

				error_CUDA = cudaMemcpy(buf, params_OptiX.v_SH_4, sizeof(float4) * params_OptiX.numberOfGaussians, cudaMemcpyDeviceToHost);
				if (error_CUDA != cudaSuccess) throw 0;
				DumpToFile(dirPath, params_OptiX.epoch, "v_sh_4", buf, sizeof(float4) * params_OptiX.numberOfGaussians);

				error_CUDA = cudaMemcpy(buf, params_OptiX.v_SH_5, sizeof(float4) * params_OptiX.numberOfGaussians, cudaMemcpyDeviceToHost);
				if (error_CUDA != cudaSuccess) throw 0;
				DumpToFile(dirPath, params_OptiX.epoch, "v_sh_5", buf, sizeof(float4) * params_OptiX.numberOfGaussians);

				error_CUDA = cudaMemcpy(buf, params_OptiX.v_SH_6, sizeof(float4) * params_OptiX.numberOfGaussians, cudaMemcpyDeviceToHost);
				if (error_CUDA != cudaSuccess) throw 0;
				DumpToFile(dirPath, params_OptiX.epoch, "v_sh_6", buf, sizeof(float4) * params_OptiX.numberOfGaussians);

				if constexpr (SH_degree >= 3) {
					error_CUDA = cudaMemcpy(buf, params_OptiX.v_SH_7, sizeof(float4) * params_OptiX.numberOfGaussians, cudaMemcpyDeviceToHost);
					if (error_CUDA != cudaSuccess) throw 0;
					DumpToFile(dirPath, params_OptiX.epoch, "v_sh_7", buf, sizeof(float4) * params_OptiX.numberOfGaussians);

					error_CUDA = cudaMemcpy(buf, params_OptiX.v_SH_8, sizeof(float4) * params_OptiX.numberOfGaussians, cudaMemcpyDeviceToHost);
					if (error_CUDA != cudaSuccess) throw 0;
					DumpToFile(dirPath, params_OptiX.epoch, "v_sh_8", buf, sizeof(float4) * params_OptiX.numberOfGaussians);

					error_CUDA = cudaMemcpy(buf, params_OptiX.v_SH_9, sizeof(float4) * params_OptiX.numberOfGaussians, cudaMemcpyDeviceToHost);
					if (error_CUDA != cudaSuccess) throw 0;
					DumpToFile(dirPath, params_OptiX.epoch, "v_sh_9", buf, sizeof(float4) * params_OptiX.numberOfGaussians);

					error_CUDA = cudaMemcpy(buf, params_OptiX.v_SH_10, sizeof(float4) * params_OptiX.numberOfGaussians, cudaMemcpyDeviceToHost);
					if (error_CUDA != cudaSuccess) throw 0;
					DumpToFile(dirPath, params_OptiX.epoch, "v_sh_10", buf, sizeof(float4) * params_OptiX.numberOfGaussians);

					error_CUDA = cudaMemcpy(buf, params_OptiX.v_SH_11, sizeof(float4) * params_OptiX.numberOfGaussians, cudaMemcpyDeviceToHost);
					if (error_CUDA != cudaSuccess) throw 0;
					DumpToFile(dirPath, params_OptiX.epoch, "v_sh_11", buf, sizeof(float4) * params_OptiX.numberOfGaussians);

					if constexpr (SH_degree >= 4) {
						error_CUDA = cudaMemcpy(buf, params_OptiX.v_SH_12, sizeof(float4) * params_OptiX.numberOfGaussians, cudaMemcpyDeviceToHost);
						if (error_CUDA != cudaSuccess) throw 0;
						DumpToFile(dirPath, params_OptiX.epoch, "v_sh_12", buf, sizeof(float4) * params_OptiX.numberOfGaussians);

						error_CUDA = cudaMemcpy(buf, params_OptiX.v_SH_13, sizeof(float4) * params_OptiX.numberOfGaussians, cudaMemcpyDeviceToHost);
						if (error_CUDA != cudaSuccess) throw 0;
						DumpToFile(dirPath, params_OptiX.epoch, "v_sh_13", buf, sizeof(float4) * params_OptiX.numberOfGaussians);

						error_CUDA = cudaMemcpy(buf, params_OptiX.v_SH_14, sizeof(float4) * params_OptiX.numberOfGaussians, cudaMemcpyDeviceToHost);
						if (error_CUDA != cudaSuccess) throw 0;
						DumpToFile(dirPath, params_OptiX.epoch, "v_sh_14", buf, sizeof(float4) * params_OptiX.numberOfGaussians);

						error_CUDA = cudaMemcpy(buf, params_OptiX.v_SH_15, sizeof(float4) * params_OptiX.numberOfGaussians, cudaMemcpyDeviceToHost);
						if (error_CUDA != cudaSuccess) throw 0;
						DumpToFile(dirPath, params_OptiX.epoch, "v_sh_15", buf, sizeof(float4) * params_OptiX.numberOfGaussians);

						error_CUDA = cudaMemcpy(buf, params_OptiX.v_SH_16, sizeof(float4) * params_OptiX.numberOfGaussians, cudaMemcpyDeviceToHost);
						if (error_CUDA != cudaSuccess) throw 0;
						DumpToFile(dirPath, params_OptiX.epoch, "v_sh_16", buf, sizeof(float4) * params_OptiX.numberOfGaussians);

						error_CUDA = cudaMemcpy(buf, params_OptiX.v_SH_17, sizeof(float4) * params_OptiX.numberOfGaussians, cudaMemcpyDeviceToHost);
						if (error_CUDA != cudaSuccess) throw 0;
						DumpToFile(dirPath, params_OptiX.epoch, "v_sh_17", buf, sizeof(float4) * params_OptiX.numberOfGaussians);

						error_CUDA = cudaMemcpy(buf, params_OptiX.v_SH_18, sizeof(float4) * params_OptiX.numberOfGaussians, cudaMemcpyDeviceToHost);
						if (error_CUDA != cudaSuccess) throw 0;
						DumpToFile(dirPath, params_OptiX.epoch, "v_sh_18", buf, sizeof(float4) * params_OptiX.numberOfGaussians);
					} else {
						// !!! !!! !!!
						error_CUDA = cudaMemcpy(buf, params_OptiX.v_SH_12, sizeof(float) * params_OptiX.numberOfGaussians, cudaMemcpyDeviceToHost);
						if (error_CUDA != cudaSuccess) throw 0;
						DumpToFile(dirPath, params_OptiX.epoch, "v_sh_12", buf, sizeof(float) * params_OptiX.numberOfGaussians);
					}
				}
			} else {
				// !!! !!! !!!
				error_CUDA = cudaMemcpy(buf, params_OptiX.v_SH_3, sizeof(float) * params_OptiX.numberOfGaussians, cudaMemcpyDeviceToHost);
				if (error_CUDA != cudaSuccess) throw 0;
				DumpToFile(dirPath, params_OptiX.epoch, "v_sh_3", buf, sizeof(float) * params_OptiX.numberOfGaussians);
			}
		}

		// *** *** *** *** ***

		// !!! !!! !!!
		error_CUDA = cudaMemcpy(buf, params_OptiX.counter2, sizeof(unsigned) * 1, cudaMemcpyDeviceToHost);
		if (error_CUDA != cudaSuccess) throw 0;
		DumpToFile(dirPath, params_OptiX.epoch, "counter", buf, sizeof(unsigned) * 1);

		// *** *** *** *** ***

		free(buf);

		return true;
	} catch (...) {
		return false;
	}
}

// *************************************************************************************************

template bool DumpParametersOptiX<0>(SOptiXRenderParams<0> &params_OptiX, char *dirPath);

// *************************************************************************************************

template bool DumpParametersOptiX<1>(SOptiXRenderParams<1> &params_OptiX, char *dirPath);

// *************************************************************************************************

template bool DumpParametersOptiX<2>(SOptiXRenderParams<2> &params_OptiX, char *dirPath);

// *************************************************************************************************

template bool DumpParametersOptiX<3>(SOptiXRenderParams<3> &params_OptiX, char *dirPath);

// *************************************************************************************************

template bool DumpParametersOptiX<4>(SOptiXRenderParams<4> &params_OptiX, char *dirPath);