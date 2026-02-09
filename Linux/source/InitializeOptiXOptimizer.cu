#include "Header.cuh"

// *************************************************************************************************

template<int SH_degree>
bool InitializeOptiXOptimizer(
	SRenderParams<SH_degree> &params,
	SOptiXRenderParams<SH_degree> &params_OptiX,
	char *dirPath
) {
	cudaError_t error_CUDA;

	try {
		error_CUDA = cudaMalloc(&params_OptiX.bitmap_ref, sizeof(unsigned) * params_OptiX.width * params_OptiX.height * params.NUMBER_OF_POSES);
		if (error_CUDA != cudaSuccess) throw 0;

		// *** *** *** *** ***

		error_CUDA = cudaMalloc(&params_OptiX.dL_dparams_1, sizeof(REAL4_G) * params_OptiX.tmpArraysGroup2Size);
		if (error_CUDA != cudaSuccess) throw 0;

		error_CUDA = cudaMalloc(&params_OptiX.dL_dparams_2, sizeof(REAL4_G) * params_OptiX.tmpArraysGroup2Size);
		if (error_CUDA != cudaSuccess) throw 0;

		error_CUDA = cudaMalloc(&params_OptiX.dL_dparams_3, sizeof(REAL4_G) * params_OptiX.tmpArraysGroup2Size);
		if (error_CUDA != cudaSuccess) throw 0;

		// !!! !!! !!!
		error_CUDA = cudaMalloc(&params_OptiX.dL_dparams_4, sizeof(REAL2_G) * params_OptiX.tmpArraysGroup2Size);
		if (error_CUDA != cudaSuccess) throw 0;

		// Spherical harmonics
		if constexpr (SH_degree >= 1) {
			error_CUDA = cudaMalloc(&params_OptiX.dL_dparams_SH_1, sizeof(REAL4_G) * params_OptiX.tmpArraysGroup2Size);
			if (error_CUDA != cudaSuccess) throw 0;

			error_CUDA = cudaMalloc(&params_OptiX.dL_dparams_SH_2, sizeof(REAL4_G) * params_OptiX.tmpArraysGroup2Size);
			if (error_CUDA != cudaSuccess) throw 0;

			if constexpr (SH_degree >= 2) {
				error_CUDA = cudaMalloc(&params_OptiX.dL_dparams_SH_3, sizeof(REAL4_G) * params_OptiX.tmpArraysGroup2Size);
				if (error_CUDA != cudaSuccess) throw 0;

				error_CUDA = cudaMalloc(&params_OptiX.dL_dparams_SH_4, sizeof(REAL4_G) * params_OptiX.tmpArraysGroup2Size);
				if (error_CUDA != cudaSuccess) throw 0;

				error_CUDA = cudaMalloc(&params_OptiX.dL_dparams_SH_5, sizeof(REAL4_G) * params_OptiX.tmpArraysGroup2Size);
				if (error_CUDA != cudaSuccess) throw 0;

				error_CUDA = cudaMalloc(&params_OptiX.dL_dparams_SH_6, sizeof(REAL4_G) * params_OptiX.tmpArraysGroup2Size);
				if (error_CUDA != cudaSuccess) throw 0;

				if constexpr (SH_degree >= 3) {
					error_CUDA = cudaMalloc(&params_OptiX.dL_dparams_SH_7, sizeof(REAL4_G) * params_OptiX.tmpArraysGroup2Size);
					if (error_CUDA != cudaSuccess) throw 0;

					error_CUDA = cudaMalloc(&params_OptiX.dL_dparams_SH_8, sizeof(REAL4_G) * params_OptiX.tmpArraysGroup2Size);
					if (error_CUDA != cudaSuccess) throw 0;

					error_CUDA = cudaMalloc(&params_OptiX.dL_dparams_SH_9, sizeof(REAL4_G) * params_OptiX.tmpArraysGroup2Size);
					if (error_CUDA != cudaSuccess) throw 0;

					error_CUDA = cudaMalloc(&params_OptiX.dL_dparams_SH_10, sizeof(REAL4_G) * params_OptiX.tmpArraysGroup2Size);
					if (error_CUDA != cudaSuccess) throw 0;

					error_CUDA = cudaMalloc(&params_OptiX.dL_dparams_SH_11, sizeof(REAL4_G) * params_OptiX.tmpArraysGroup2Size);
					if (error_CUDA != cudaSuccess) throw 0;

					if constexpr (SH_degree >= 4) {
						error_CUDA = cudaMalloc(&params_OptiX.dL_dparams_SH_12, sizeof(REAL4_G) * params_OptiX.tmpArraysGroup2Size);
						if (error_CUDA != cudaSuccess) throw 0;

						error_CUDA = cudaMalloc(&params_OptiX.dL_dparams_SH_13, sizeof(REAL4_G) * params_OptiX.tmpArraysGroup2Size);
						if (error_CUDA != cudaSuccess) throw 0;

						error_CUDA = cudaMalloc(&params_OptiX.dL_dparams_SH_14, sizeof(REAL4_G) * params_OptiX.tmpArraysGroup2Size);
						if (error_CUDA != cudaSuccess) throw 0;

						error_CUDA = cudaMalloc(&params_OptiX.dL_dparams_SH_15, sizeof(REAL4_G) * params_OptiX.tmpArraysGroup2Size);
						if (error_CUDA != cudaSuccess) throw 0;

						error_CUDA = cudaMalloc(&params_OptiX.dL_dparams_SH_16, sizeof(REAL4_G) * params_OptiX.tmpArraysGroup2Size);
						if (error_CUDA != cudaSuccess) throw 0;

						error_CUDA = cudaMalloc(&params_OptiX.dL_dparams_SH_17, sizeof(REAL4_G) * params_OptiX.tmpArraysGroup2Size);
						if (error_CUDA != cudaSuccess) throw 0;

						error_CUDA = cudaMalloc(&params_OptiX.dL_dparams_SH_18, sizeof(REAL4_G) * params_OptiX.tmpArraysGroup2Size);
						if (error_CUDA != cudaSuccess) throw 0;
					} else {
						// !!! !!! !!!
						error_CUDA = cudaMalloc(&params_OptiX.dL_dparams_SH_12, sizeof(REAL_G) * params_OptiX.tmpArraysGroup2Size);
						if (error_CUDA != cudaSuccess) throw 0;
					}
				}
			} else {
				// !!! !!! !!!
				error_CUDA = cudaMalloc(&params_OptiX.dL_dparams_SH_3, sizeof(REAL_G) * params_OptiX.tmpArraysGroup2Size);
				if (error_CUDA != cudaSuccess) throw 0;
			}
		}

		// *** *** *** *** ***

		error_CUDA = cudaMalloc(&params_OptiX.loss_device, sizeof(double) * 1);
		if (error_CUDA != cudaSuccess) throw 0;

		// !!! !!! !!!
		error_CUDA = cudaMalloc(&params_OptiX.max_RSH, sizeof(float) * 1 * params_OptiX.width * params_OptiX.height);
		if (error_CUDA != cudaSuccess) throw 0;

		error_CUDA = cudaMalloc(&params_OptiX.max_GSH, sizeof(float) * 1 * params_OptiX.width * params_OptiX.height);
		if (error_CUDA != cudaSuccess) throw 0;

		error_CUDA = cudaMalloc(&params_OptiX.max_BSH, sizeof(float) * 1 * params_OptiX.width * params_OptiX.height);
		if (error_CUDA != cudaSuccess) throw 0;
		// !!! !!! !!!

		error_CUDA = cudaMalloc(&params_OptiX.Gaussians_indices, sizeof(int) * max_Gaussians_per_ray_host * params_OptiX.width * params_OptiX.height);
		if (error_CUDA != cudaSuccess) throw 0;

		// *** *** *** *** ***

		error_CUDA = cudaMalloc(&params_OptiX.m11, sizeof(float4) * params_OptiX.tmpArraysGroup1Size);
		if (error_CUDA != cudaSuccess) throw 0;

		error_CUDA = cudaMalloc(&params_OptiX.m21, sizeof(float4) * params_OptiX.tmpArraysGroup1Size);
		if (error_CUDA != cudaSuccess) throw 0;

		error_CUDA = cudaMalloc(&params_OptiX.m31, sizeof(float4) * params_OptiX.tmpArraysGroup1Size);
		if (error_CUDA != cudaSuccess) throw 0;

		// !!! !!! !!!
		error_CUDA = cudaMalloc(&params_OptiX.m41, sizeof(float2) * params_OptiX.tmpArraysGroup1Size);
		if (error_CUDA != cudaSuccess) throw 0;

		// Spherical harmonics
		if constexpr (SH_degree >= 1) {
			error_CUDA = cudaMalloc(&params_OptiX.m_SH_1, sizeof(float4) * params_OptiX.tmpArraysGroup1Size);
			if (error_CUDA != cudaSuccess) throw 0;

			error_CUDA = cudaMalloc(&params_OptiX.m_SH_2, sizeof(float4) * params_OptiX.tmpArraysGroup1Size);
			if (error_CUDA != cudaSuccess) throw 0;

			if constexpr (SH_degree >= 2) {
				error_CUDA = cudaMalloc(&params_OptiX.m_SH_3, sizeof(float4) * params_OptiX.tmpArraysGroup1Size);
				if (error_CUDA != cudaSuccess) throw 0;

				error_CUDA = cudaMalloc(&params_OptiX.m_SH_4, sizeof(float4) * params_OptiX.tmpArraysGroup1Size);
				if (error_CUDA != cudaSuccess) throw 0;

				error_CUDA = cudaMalloc(&params_OptiX.m_SH_5, sizeof(float4) * params_OptiX.tmpArraysGroup1Size);
				if (error_CUDA != cudaSuccess) throw 0;

				error_CUDA = cudaMalloc(&params_OptiX.m_SH_6, sizeof(float4) * params_OptiX.tmpArraysGroup1Size);
				if (error_CUDA != cudaSuccess) throw 0;

				if constexpr (SH_degree >= 3) {
					error_CUDA = cudaMalloc(&params_OptiX.m_SH_7, sizeof(float4) * params_OptiX.tmpArraysGroup1Size);
					if (error_CUDA != cudaSuccess) throw 0;

					error_CUDA = cudaMalloc(&params_OptiX.m_SH_8, sizeof(float4) * params_OptiX.tmpArraysGroup1Size);
					if (error_CUDA != cudaSuccess) throw 0;

					error_CUDA = cudaMalloc(&params_OptiX.m_SH_9, sizeof(float4) * params_OptiX.tmpArraysGroup1Size);
					if (error_CUDA != cudaSuccess) throw 0;

					error_CUDA = cudaMalloc(&params_OptiX.m_SH_10, sizeof(float4) * params_OptiX.tmpArraysGroup1Size);
					if (error_CUDA != cudaSuccess) throw 0;

					error_CUDA = cudaMalloc(&params_OptiX.m_SH_11, sizeof(float4) * params_OptiX.tmpArraysGroup1Size);
					if (error_CUDA != cudaSuccess) throw 0;

					if constexpr (SH_degree >= 4) {
						error_CUDA = cudaMalloc(&params_OptiX.m_SH_12, sizeof(float4) * params_OptiX.tmpArraysGroup1Size);
						if (error_CUDA != cudaSuccess) throw 0;

						error_CUDA = cudaMalloc(&params_OptiX.m_SH_13, sizeof(float4) * params_OptiX.tmpArraysGroup1Size);
						if (error_CUDA != cudaSuccess) throw 0;

						error_CUDA = cudaMalloc(&params_OptiX.m_SH_14, sizeof(float4) * params_OptiX.tmpArraysGroup1Size);
						if (error_CUDA != cudaSuccess) throw 0;

						error_CUDA = cudaMalloc(&params_OptiX.m_SH_15, sizeof(float4) * params_OptiX.tmpArraysGroup1Size);
						if (error_CUDA != cudaSuccess) throw 0;

						error_CUDA = cudaMalloc(&params_OptiX.m_SH_16, sizeof(float4) * params_OptiX.tmpArraysGroup1Size);
						if (error_CUDA != cudaSuccess) throw 0;

						error_CUDA = cudaMalloc(&params_OptiX.m_SH_17, sizeof(float4) * params_OptiX.tmpArraysGroup1Size);
						if (error_CUDA != cudaSuccess) throw 0;

						error_CUDA = cudaMalloc(&params_OptiX.m_SH_18, sizeof(float4) * params_OptiX.tmpArraysGroup1Size);
						if (error_CUDA != cudaSuccess) throw 0;
					} else {
						// !!! !!! !!!
						error_CUDA = cudaMalloc(&params_OptiX.m_SH_12, sizeof(float) * params_OptiX.tmpArraysGroup1Size);
						if (error_CUDA != cudaSuccess) throw 0;
					}
				}
			} else {
				// !!! !!! !!!
				error_CUDA = cudaMalloc(&params_OptiX.m_SH_3, sizeof(float) * params_OptiX.tmpArraysGroup1Size);
				if (error_CUDA != cudaSuccess) throw 0;
			}
		}

		// *** *** *** *** ***

		error_CUDA = cudaMalloc(&params_OptiX.v11, sizeof(float4) * params_OptiX.tmpArraysGroup1Size);
		if (error_CUDA != cudaSuccess) throw 0;

		error_CUDA = cudaMalloc(&params_OptiX.v21, sizeof(float4) * params_OptiX.tmpArraysGroup1Size);
		if (error_CUDA != cudaSuccess) throw 0;

		error_CUDA = cudaMalloc(&params_OptiX.v31, sizeof(float4) * params_OptiX.tmpArraysGroup1Size);
		if (error_CUDA != cudaSuccess) throw 0;

		// !!! !!! !!!
		error_CUDA = cudaMalloc(&params_OptiX.v41, sizeof(float2) * params_OptiX.tmpArraysGroup1Size);
		if (error_CUDA != cudaSuccess) throw 0;

		// Spherical harmonics
		if constexpr (SH_degree >= 1) {
			error_CUDA = cudaMalloc(&params_OptiX.v_SH_1, sizeof(float4) * params_OptiX.tmpArraysGroup1Size);
			if (error_CUDA != cudaSuccess) throw 0;

			error_CUDA = cudaMalloc(&params_OptiX.v_SH_2, sizeof(float4) * params_OptiX.tmpArraysGroup1Size);
			if (error_CUDA != cudaSuccess) throw 0;

			if constexpr (SH_degree >= 2) {
				error_CUDA = cudaMalloc(&params_OptiX.v_SH_3, sizeof(float4) * params_OptiX.tmpArraysGroup1Size);
				if (error_CUDA != cudaSuccess) throw 0;

				error_CUDA = cudaMalloc(&params_OptiX.v_SH_4, sizeof(float4) * params_OptiX.tmpArraysGroup1Size);
				if (error_CUDA != cudaSuccess) throw 0;

				error_CUDA = cudaMalloc(&params_OptiX.v_SH_5, sizeof(float4) * params_OptiX.tmpArraysGroup1Size);
				if (error_CUDA != cudaSuccess) throw 0;

				error_CUDA = cudaMalloc(&params_OptiX.v_SH_6, sizeof(float4) * params_OptiX.tmpArraysGroup1Size);
				if (error_CUDA != cudaSuccess) throw 0;

				if constexpr (SH_degree >= 3) {
					error_CUDA = cudaMalloc(&params_OptiX.v_SH_7, sizeof(float4) * params_OptiX.tmpArraysGroup1Size);
					if (error_CUDA != cudaSuccess) throw 0;

					error_CUDA = cudaMalloc(&params_OptiX.v_SH_8, sizeof(float4) * params_OptiX.tmpArraysGroup1Size);
					if (error_CUDA != cudaSuccess) throw 0;

					error_CUDA = cudaMalloc(&params_OptiX.v_SH_9, sizeof(float4) * params_OptiX.tmpArraysGroup1Size);
					if (error_CUDA != cudaSuccess) throw 0;

					error_CUDA = cudaMalloc(&params_OptiX.v_SH_10, sizeof(float4) * params_OptiX.tmpArraysGroup1Size);
					if (error_CUDA != cudaSuccess) throw 0;

					error_CUDA = cudaMalloc(&params_OptiX.v_SH_11, sizeof(float4) * params_OptiX.tmpArraysGroup1Size);
					if (error_CUDA != cudaSuccess) throw 0;

					if constexpr (SH_degree >= 4) {
						error_CUDA = cudaMalloc(&params_OptiX.v_SH_12, sizeof(float4) * params_OptiX.tmpArraysGroup1Size);
						if (error_CUDA != cudaSuccess) throw 0;

						error_CUDA = cudaMalloc(&params_OptiX.v_SH_13, sizeof(float4) * params_OptiX.tmpArraysGroup1Size);
						if (error_CUDA != cudaSuccess) throw 0;

						error_CUDA = cudaMalloc(&params_OptiX.v_SH_14, sizeof(float4) * params_OptiX.tmpArraysGroup1Size);
						if (error_CUDA != cudaSuccess) throw 0;

						error_CUDA = cudaMalloc(&params_OptiX.v_SH_15, sizeof(float4) * params_OptiX.tmpArraysGroup1Size);
						if (error_CUDA != cudaSuccess) throw 0;

						error_CUDA = cudaMalloc(&params_OptiX.v_SH_16, sizeof(float4) * params_OptiX.tmpArraysGroup1Size);
						if (error_CUDA != cudaSuccess) throw 0;

						error_CUDA = cudaMalloc(&params_OptiX.v_SH_17, sizeof(float4) * params_OptiX.tmpArraysGroup1Size);
						if (error_CUDA != cudaSuccess) throw 0;

						error_CUDA = cudaMalloc(&params_OptiX.v_SH_18, sizeof(float4) * params_OptiX.tmpArraysGroup1Size);
						if (error_CUDA != cudaSuccess) throw 0;
					} else {
						// !!! !!! !!!
						error_CUDA = cudaMalloc(&params_OptiX.v_SH_12, sizeof(float) * params_OptiX.tmpArraysGroup1Size);
						if (error_CUDA != cudaSuccess) throw 0;
					}
				}
			} else {
				// !!! !!! !!!
				error_CUDA = cudaMalloc(&params_OptiX.v_SH_3, sizeof(float) * params_OptiX.tmpArraysGroup1Size);
				if (error_CUDA != cudaSuccess) throw 0;
			}
		}

		// *** *** *** *** ***

		error_CUDA = cudaMalloc(&params_OptiX.counter1, sizeof(unsigned) * 1);
		if (error_CUDA != cudaSuccess) throw 0;

		error_CUDA = cudaMalloc(&params_OptiX.counter2, sizeof(unsigned) * 1);
		if (error_CUDA != cudaSuccess) throw 0;

		// ************************************************************************************************

		error_CUDA = cudaMemcpy(params_OptiX.bitmap_ref, params.bitmap_ref, sizeof(unsigned) * params_OptiX.width * params_OptiX.height * params.NUMBER_OF_POSES, cudaMemcpyHostToDevice);
		if (error_CUDA != cudaSuccess) throw 0;

		// ************************************************************************************************

		if (dirPath == NULL) {
			error_CUDA = cudaMemset(params_OptiX.m11, 0, sizeof(float4) * params_OptiX.tmpArraysGroup1Size);
			if (error_CUDA != cudaSuccess) throw 0;

			error_CUDA = cudaMemset(params_OptiX.m21, 0, sizeof(float4) * params_OptiX.tmpArraysGroup1Size);
			if (error_CUDA != cudaSuccess) throw 0;

			error_CUDA = cudaMemset(params_OptiX.m31, 0, sizeof(float4) * params_OptiX.tmpArraysGroup1Size);
			if (error_CUDA != cudaSuccess) throw 0;

			// !!! !!! !!!
			error_CUDA = cudaMemset(params_OptiX.m41, 0, sizeof(float2) * params_OptiX.tmpArraysGroup1Size);
			if (error_CUDA != cudaSuccess) throw 0;

			// Spherical harmonics
			if constexpr (SH_degree >= 1) {
				error_CUDA = cudaMemset(params_OptiX.m_SH_1, 0, sizeof(float4) * params_OptiX.numberOfGaussians);
				if (error_CUDA != cudaSuccess) throw 0;

				error_CUDA = cudaMemset(params_OptiX.m_SH_2, 0, sizeof(float4) * params_OptiX.numberOfGaussians);
				if (error_CUDA != cudaSuccess) throw 0;

				if constexpr (SH_degree >= 2) {
					error_CUDA = cudaMemset(params_OptiX.m_SH_3, 0, sizeof(float4) * params_OptiX.numberOfGaussians);
					if (error_CUDA != cudaSuccess) throw 0;

					error_CUDA = cudaMemset(params_OptiX.m_SH_4, 0, sizeof(float4) * params_OptiX.numberOfGaussians);
					if (error_CUDA != cudaSuccess) throw 0;

					error_CUDA = cudaMemset(params_OptiX.m_SH_5, 0, sizeof(float4) * params_OptiX.numberOfGaussians);
					if (error_CUDA != cudaSuccess) throw 0;

					error_CUDA = cudaMemset(params_OptiX.m_SH_6, 0, sizeof(float4) * params_OptiX.numberOfGaussians);
					if (error_CUDA != cudaSuccess) throw 0;

					if constexpr (SH_degree >= 3) {
						error_CUDA = cudaMemset(params_OptiX.m_SH_7, 0, sizeof(float4) * params_OptiX.numberOfGaussians);
						if (error_CUDA != cudaSuccess) throw 0;

						error_CUDA = cudaMemset(params_OptiX.m_SH_8, 0, sizeof(float4) * params_OptiX.numberOfGaussians);
						if (error_CUDA != cudaSuccess) throw 0;

						error_CUDA = cudaMemset(params_OptiX.m_SH_9, 0, sizeof(float4) * params_OptiX.numberOfGaussians);
						if (error_CUDA != cudaSuccess) throw 0;

						error_CUDA = cudaMemset(params_OptiX.m_SH_10, 0, sizeof(float4) * params_OptiX.numberOfGaussians);
						if (error_CUDA != cudaSuccess) throw 0;

						error_CUDA = cudaMemset(params_OptiX.m_SH_11, 0, sizeof(float4) * params_OptiX.numberOfGaussians);
						if (error_CUDA != cudaSuccess) throw 0;

						if constexpr (SH_degree >= 4) {
							error_CUDA = cudaMemset(params_OptiX.m_SH_12, 0, sizeof(float4) * params_OptiX.numberOfGaussians);
							if (error_CUDA != cudaSuccess) throw 0;

							error_CUDA = cudaMemset(params_OptiX.m_SH_13, 0, sizeof(float4) * params_OptiX.numberOfGaussians);
							if (error_CUDA != cudaSuccess) throw 0;

							error_CUDA = cudaMemset(params_OptiX.m_SH_14, 0, sizeof(float4) * params_OptiX.numberOfGaussians);
							if (error_CUDA != cudaSuccess) throw 0;

							error_CUDA = cudaMemset(params_OptiX.m_SH_15, 0, sizeof(float4) * params_OptiX.numberOfGaussians);
							if (error_CUDA != cudaSuccess) throw 0;

							error_CUDA = cudaMemset(params_OptiX.m_SH_16, 0, sizeof(float4) * params_OptiX.numberOfGaussians);
							if (error_CUDA != cudaSuccess) throw 0;

							error_CUDA = cudaMemset(params_OptiX.m_SH_17, 0, sizeof(float4) * params_OptiX.numberOfGaussians);
							if (error_CUDA != cudaSuccess) throw 0;

							error_CUDA = cudaMemset(params_OptiX.m_SH_18, 0, sizeof(float4) * params_OptiX.numberOfGaussians);
							if (error_CUDA != cudaSuccess) throw 0;
						} else {
							// !!! !!! !!!
							error_CUDA = cudaMemset(params_OptiX.m_SH_12, 0, sizeof(float) * params_OptiX.numberOfGaussians);
							if (error_CUDA != cudaSuccess) throw 0;
						}
					}
				} else {
					// !!! !!! !!!
					error_CUDA = cudaMemset(params_OptiX.m_SH_3, 0, sizeof(float) * params_OptiX.numberOfGaussians);
					if (error_CUDA != cudaSuccess) throw 0;
				}
			}

			// *** *** *** *** ***

			error_CUDA = cudaMemset(params_OptiX.v11, 0, sizeof(float4) * params_OptiX.tmpArraysGroup1Size);
			if (error_CUDA != cudaSuccess) throw 0;

			error_CUDA = cudaMemset(params_OptiX.v21, 0, sizeof(float4) * params_OptiX.tmpArraysGroup1Size);
			if (error_CUDA != cudaSuccess) throw 0;

			error_CUDA = cudaMemset(params_OptiX.v31, 0, sizeof(float4) * params_OptiX.tmpArraysGroup1Size);
			if (error_CUDA != cudaSuccess) throw 0;

			// !!! !!! !!!
			error_CUDA = cudaMemset(params_OptiX.v41, 0, sizeof(float2) * params_OptiX.tmpArraysGroup1Size);
			if (error_CUDA != cudaSuccess) throw 0;

			// Spherical harmonics
			if constexpr (SH_degree >= 1) {
				error_CUDA = cudaMemset(params_OptiX.v_SH_1, 0, sizeof(float4) * params_OptiX.numberOfGaussians);
				if (error_CUDA != cudaSuccess) throw 0;

				error_CUDA = cudaMemset(params_OptiX.v_SH_2, 0, sizeof(float4) * params_OptiX.numberOfGaussians);
				if (error_CUDA != cudaSuccess) throw 0;

				if constexpr (SH_degree >= 2) {
					error_CUDA = cudaMemset(params_OptiX.v_SH_3, 0, sizeof(float4) * params_OptiX.numberOfGaussians);
					if (error_CUDA != cudaSuccess) throw 0;

					error_CUDA = cudaMemset(params_OptiX.v_SH_4, 0, sizeof(float4) * params_OptiX.numberOfGaussians);
					if (error_CUDA != cudaSuccess) throw 0;

					error_CUDA = cudaMemset(params_OptiX.v_SH_5, 0, sizeof(float4) * params_OptiX.numberOfGaussians);
					if (error_CUDA != cudaSuccess) throw 0;

					error_CUDA = cudaMemset(params_OptiX.v_SH_6, 0, sizeof(float4) * params_OptiX.numberOfGaussians);
					if (error_CUDA != cudaSuccess) throw 0;

					if constexpr (SH_degree >= 3) {
						error_CUDA = cudaMemset(params_OptiX.v_SH_7, 0, sizeof(float4) * params_OptiX.numberOfGaussians);
						if (error_CUDA != cudaSuccess) throw 0;

						error_CUDA = cudaMemset(params_OptiX.v_SH_8, 0, sizeof(float4) * params_OptiX.numberOfGaussians);
						if (error_CUDA != cudaSuccess) throw 0;

						error_CUDA = cudaMemset(params_OptiX.v_SH_9, 0, sizeof(float4) * params_OptiX.numberOfGaussians);
						if (error_CUDA != cudaSuccess) throw 0;

						error_CUDA = cudaMemset(params_OptiX.v_SH_10, 0, sizeof(float4) * params_OptiX.numberOfGaussians);
						if (error_CUDA != cudaSuccess) throw 0;

						error_CUDA = cudaMemset(params_OptiX.v_SH_11, 0, sizeof(float4) * params_OptiX.numberOfGaussians);
						if (error_CUDA != cudaSuccess) throw 0;

						if constexpr (SH_degree >= 4) {
							error_CUDA = cudaMemset(params_OptiX.v_SH_12, 0, sizeof(float4) * params_OptiX.numberOfGaussians);
							if (error_CUDA != cudaSuccess) throw 0;

							error_CUDA = cudaMemset(params_OptiX.v_SH_13, 0, sizeof(float4) * params_OptiX.numberOfGaussians);
							if (error_CUDA != cudaSuccess) throw 0;

							error_CUDA = cudaMemset(params_OptiX.v_SH_14, 0, sizeof(float4) * params_OptiX.numberOfGaussians);
							if (error_CUDA != cudaSuccess) throw 0;

							error_CUDA = cudaMemset(params_OptiX.v_SH_15, 0, sizeof(float4) * params_OptiX.numberOfGaussians);
							if (error_CUDA != cudaSuccess) throw 0;

							error_CUDA = cudaMemset(params_OptiX.v_SH_16, 0, sizeof(float4) * params_OptiX.numberOfGaussians);
							if (error_CUDA != cudaSuccess) throw 0;

							error_CUDA = cudaMemset(params_OptiX.v_SH_17, 0, sizeof(float4) * params_OptiX.numberOfGaussians);
							if (error_CUDA != cudaSuccess) throw 0;

							error_CUDA = cudaMemset(params_OptiX.v_SH_18, 0, sizeof(float4) * params_OptiX.numberOfGaussians);
							if (error_CUDA != cudaSuccess) throw 0;
						} else {
							// !!! !!! !!!
							error_CUDA = cudaMemset(params_OptiX.v_SH_12, 0, sizeof(float) * params_OptiX.numberOfGaussians);
							if (error_CUDA != cudaSuccess) throw 0;
						}
					}
				} else {
					// !!! !!! !!!
					error_CUDA = cudaMemset(params_OptiX.v_SH_3, 0, sizeof(float) * params_OptiX.numberOfGaussians);
					if (error_CUDA != cudaSuccess) throw 0;
				}
			}

			// *** *** *** *** ***

			// !!! !!! !!!
			error_CUDA = cudaMemset(params_OptiX.counter2, 0, sizeof(unsigned) * 1);
			if (error_CUDA != cudaSuccess) throw 0;
		} else {
			void *buf = malloc(sizeof(float4) * params_OptiX.numberOfGaussians);
			if (buf == NULL) throw 0;

			// *** *** *** *** ***

			LoadFromFile(dirPath, params_OptiX.epoch, "m1", buf, sizeof(float4) * params_OptiX.numberOfGaussians);
			error_CUDA = cudaMemcpy(params_OptiX.m11, buf, sizeof(float4) * params_OptiX.numberOfGaussians, cudaMemcpyHostToDevice);
			if (error_CUDA != cudaSuccess) throw 0;

			LoadFromFile(dirPath, params_OptiX.epoch, "m2", buf, sizeof(float4) * params_OptiX.numberOfGaussians);
			error_CUDA = cudaMemcpy(params_OptiX.m21, buf, sizeof(float4) * params_OptiX.numberOfGaussians, cudaMemcpyHostToDevice);
			if (error_CUDA != cudaSuccess) throw 0;

			LoadFromFile(dirPath, params_OptiX.epoch, "m3", buf, sizeof(float4) * params_OptiX.numberOfGaussians);
			error_CUDA = cudaMemcpy(params_OptiX.m31, buf, sizeof(float4) * params_OptiX.numberOfGaussians, cudaMemcpyHostToDevice);
			if (error_CUDA != cudaSuccess) throw 0;

			LoadFromFile(dirPath, params_OptiX.epoch, "m4", buf, sizeof(float2) * params_OptiX.numberOfGaussians);
			error_CUDA = cudaMemcpy(params_OptiX.m41, buf, sizeof(float2) * params_OptiX.numberOfGaussians, cudaMemcpyHostToDevice);
			if (error_CUDA != cudaSuccess) throw 0;

			// *** *** *** *** ***

			// Spherical harmonics
			if constexpr (SH_degree >= 1) {
				LoadFromFile(dirPath, params_OptiX.epoch, "m_sh_1", buf, sizeof(float4) * params_OptiX.numberOfGaussians);
				error_CUDA = cudaMemcpy(params_OptiX.m_SH_1, buf, sizeof(float4) * params_OptiX.numberOfGaussians, cudaMemcpyHostToDevice);
				if (error_CUDA != cudaSuccess) throw 0;

				LoadFromFile(dirPath, params_OptiX.epoch, "m_sh_2", buf, sizeof(float4) * params_OptiX.numberOfGaussians);
				error_CUDA = cudaMemcpy(params_OptiX.m_SH_2, buf, sizeof(float4) * params_OptiX.numberOfGaussians, cudaMemcpyHostToDevice);
				if (error_CUDA != cudaSuccess) throw 0;

				if constexpr (SH_degree >= 2) {
					LoadFromFile(dirPath, params_OptiX.epoch, "m_sh_3", buf, sizeof(float4) * params_OptiX.numberOfGaussians);
					error_CUDA = cudaMemcpy(params_OptiX.m_SH_3, buf, sizeof(float4) * params_OptiX.numberOfGaussians, cudaMemcpyHostToDevice);
					if (error_CUDA != cudaSuccess) throw 0;

					LoadFromFile(dirPath, params_OptiX.epoch, "m_sh_4", buf, sizeof(float4) * params_OptiX.numberOfGaussians);
					error_CUDA = cudaMemcpy(params_OptiX.m_SH_4, buf, sizeof(float4) * params_OptiX.numberOfGaussians, cudaMemcpyHostToDevice);
					if (error_CUDA != cudaSuccess) throw 0;

					LoadFromFile(dirPath, params_OptiX.epoch, "m_sh_5", buf, sizeof(float4) * params_OptiX.numberOfGaussians);
					error_CUDA = cudaMemcpy(params_OptiX.m_SH_5, buf, sizeof(float4) * params_OptiX.numberOfGaussians, cudaMemcpyHostToDevice);
					if (error_CUDA != cudaSuccess) throw 0;

					LoadFromFile(dirPath, params_OptiX.epoch, "m_sh_6", buf, sizeof(float4) * params_OptiX.numberOfGaussians);
					error_CUDA = cudaMemcpy(params_OptiX.m_SH_6, buf, sizeof(float4) * params_OptiX.numberOfGaussians, cudaMemcpyHostToDevice);
					if (error_CUDA != cudaSuccess) throw 0;

					if constexpr (SH_degree >= 3) {
						LoadFromFile(dirPath, params_OptiX.epoch, "m_sh_7", buf, sizeof(float4) * params_OptiX.numberOfGaussians);
						error_CUDA = cudaMemcpy(params_OptiX.m_SH_7, buf, sizeof(float4) * params_OptiX.numberOfGaussians, cudaMemcpyHostToDevice);
						if (error_CUDA != cudaSuccess) throw 0;

						LoadFromFile(dirPath, params_OptiX.epoch, "m_sh_8", buf, sizeof(float4) * params_OptiX.numberOfGaussians);
						error_CUDA = cudaMemcpy(params_OptiX.m_SH_8, buf, sizeof(float4) * params_OptiX.numberOfGaussians, cudaMemcpyHostToDevice);
						if (error_CUDA != cudaSuccess) throw 0;

						LoadFromFile(dirPath, params_OptiX.epoch, "m_sh_9", buf, sizeof(float4) * params_OptiX.numberOfGaussians);
						error_CUDA = cudaMemcpy(params_OptiX.m_SH_9, buf, sizeof(float4) * params_OptiX.numberOfGaussians, cudaMemcpyHostToDevice);
						if (error_CUDA != cudaSuccess) throw 0;

						LoadFromFile(dirPath, params_OptiX.epoch, "m_sh_10", buf, sizeof(float4) * params_OptiX.numberOfGaussians);
						error_CUDA = cudaMemcpy(params_OptiX.m_SH_10, buf, sizeof(float4) * params_OptiX.numberOfGaussians, cudaMemcpyHostToDevice);
						if (error_CUDA != cudaSuccess) throw 0;

						LoadFromFile(dirPath, params_OptiX.epoch, "m_sh_11", buf, sizeof(float4) * params_OptiX.numberOfGaussians);
						error_CUDA = cudaMemcpy(params_OptiX.m_SH_11, buf, sizeof(float4) * params_OptiX.numberOfGaussians, cudaMemcpyHostToDevice);
						if (error_CUDA != cudaSuccess) throw 0;

						if constexpr (SH_degree >= 4) {
							LoadFromFile(dirPath, params_OptiX.epoch, "m_sh_12", buf, sizeof(float4) * params_OptiX.numberOfGaussians);
							error_CUDA = cudaMemcpy(params_OptiX.m_SH_12, buf, sizeof(float4) * params_OptiX.numberOfGaussians, cudaMemcpyHostToDevice);
							if (error_CUDA != cudaSuccess) throw 0;

							LoadFromFile(dirPath, params_OptiX.epoch, "m_sh_13", buf, sizeof(float4) * params_OptiX.numberOfGaussians);
							error_CUDA = cudaMemcpy(params_OptiX.m_SH_13, buf, sizeof(float4) * params_OptiX.numberOfGaussians, cudaMemcpyHostToDevice);
							if (error_CUDA != cudaSuccess) throw 0;

							LoadFromFile(dirPath, params_OptiX.epoch, "m_sh_14", buf, sizeof(float4) * params_OptiX.numberOfGaussians);
							error_CUDA = cudaMemcpy(params_OptiX.m_SH_14, buf, sizeof(float4) * params_OptiX.numberOfGaussians, cudaMemcpyHostToDevice);
							if (error_CUDA != cudaSuccess) throw 0;

							LoadFromFile(dirPath, params_OptiX.epoch, "m_sh_15", buf, sizeof(float4) * params_OptiX.numberOfGaussians);
							error_CUDA = cudaMemcpy(params_OptiX.m_SH_15, buf, sizeof(float4) * params_OptiX.numberOfGaussians, cudaMemcpyHostToDevice);
							if (error_CUDA != cudaSuccess) throw 0;

							LoadFromFile(dirPath, params_OptiX.epoch, "m_sh_16", buf, sizeof(float4) * params_OptiX.numberOfGaussians);
							error_CUDA = cudaMemcpy(params_OptiX.m_SH_16, buf, sizeof(float4) * params_OptiX.numberOfGaussians, cudaMemcpyHostToDevice);
							if (error_CUDA != cudaSuccess) throw 0;

							LoadFromFile(dirPath, params_OptiX.epoch, "m_sh_17", buf, sizeof(float4) * params_OptiX.numberOfGaussians);
							error_CUDA = cudaMemcpy(params_OptiX.m_SH_17, buf, sizeof(float4) * params_OptiX.numberOfGaussians, cudaMemcpyHostToDevice);
							if (error_CUDA != cudaSuccess) throw 0;

							LoadFromFile(dirPath, params_OptiX.epoch, "m_sh_18", buf, sizeof(float4) * params_OptiX.numberOfGaussians);
							error_CUDA = cudaMemcpy(params_OptiX.m_SH_18, buf, sizeof(float4) * params_OptiX.numberOfGaussians, cudaMemcpyHostToDevice);
							if (error_CUDA != cudaSuccess) throw 0;
						} else {
							// !!! !!! !!!
							LoadFromFile(dirPath, params_OptiX.epoch, "m_sh_12", buf, sizeof(float) * params_OptiX.numberOfGaussians);
							error_CUDA = cudaMemcpy(params_OptiX.m_SH_12, buf, sizeof(float) * params_OptiX.numberOfGaussians, cudaMemcpyHostToDevice);
							if (error_CUDA != cudaSuccess) throw 0;
						}
					}
				} else {
					// !!! !!! !!!
					LoadFromFile(dirPath, params_OptiX.epoch, "m_sh_3", buf, sizeof(float) * params_OptiX.numberOfGaussians);
					error_CUDA = cudaMemcpy(params_OptiX.m_SH_3, buf, sizeof(float) * params_OptiX.numberOfGaussians, cudaMemcpyHostToDevice);
					if (error_CUDA != cudaSuccess) throw 0;
				}
			}

			// *** *** *** *** ***

			LoadFromFile(dirPath, params_OptiX.epoch, "v1", buf, sizeof(float4) * params_OptiX.numberOfGaussians);
			error_CUDA = cudaMemcpy(params_OptiX.v11, buf, sizeof(float4) * params_OptiX.numberOfGaussians, cudaMemcpyHostToDevice);
			if (error_CUDA != cudaSuccess) throw 0;

			LoadFromFile(dirPath, params_OptiX.epoch, "v2", buf, sizeof(float4) * params_OptiX.numberOfGaussians);
			error_CUDA = cudaMemcpy(params_OptiX.v21, buf, sizeof(float4) * params_OptiX.numberOfGaussians, cudaMemcpyHostToDevice);
			if (error_CUDA != cudaSuccess) throw 0;

			LoadFromFile(dirPath, params_OptiX.epoch, "v3", buf, sizeof(float4) * params_OptiX.numberOfGaussians);
			error_CUDA = cudaMemcpy(params_OptiX.v31, buf, sizeof(float4) * params_OptiX.numberOfGaussians, cudaMemcpyHostToDevice);
			if (error_CUDA != cudaSuccess) throw 0;

			LoadFromFile(dirPath, params_OptiX.epoch, "v4", buf, sizeof(float2) * params_OptiX.numberOfGaussians);
			error_CUDA = cudaMemcpy(params_OptiX.v41, buf, sizeof(float2) * params_OptiX.numberOfGaussians, cudaMemcpyHostToDevice);
			if (error_CUDA != cudaSuccess) throw 0;

			// *** *** *** *** ***

			// Spherical harmonics
			if constexpr (SH_degree >= 1) {
				LoadFromFile(dirPath, params_OptiX.epoch, "v_sh_1", buf, sizeof(float4) * params_OptiX.numberOfGaussians);
				error_CUDA = cudaMemcpy(params_OptiX.v_SH_1, buf, sizeof(float4) * params_OptiX.numberOfGaussians, cudaMemcpyHostToDevice);
				if (error_CUDA != cudaSuccess) throw 0;

				LoadFromFile(dirPath, params_OptiX.epoch, "v_sh_2", buf, sizeof(float4) * params_OptiX.numberOfGaussians);
				error_CUDA = cudaMemcpy(params_OptiX.v_SH_2, buf, sizeof(float4) * params_OptiX.numberOfGaussians, cudaMemcpyHostToDevice);
				if (error_CUDA != cudaSuccess) throw 0;

				if constexpr (SH_degree >= 2) {
					LoadFromFile(dirPath, params_OptiX.epoch, "v_sh_3", buf, sizeof(float4) * params_OptiX.numberOfGaussians);
					error_CUDA = cudaMemcpy(params_OptiX.v_SH_3, buf, sizeof(float4) * params_OptiX.numberOfGaussians, cudaMemcpyHostToDevice);
					if (error_CUDA != cudaSuccess) throw 0;

					LoadFromFile(dirPath, params_OptiX.epoch, "v_sh_4", buf, sizeof(float4) * params_OptiX.numberOfGaussians);
					error_CUDA = cudaMemcpy(params_OptiX.v_SH_4, buf, sizeof(float4) * params_OptiX.numberOfGaussians, cudaMemcpyHostToDevice);
					if (error_CUDA != cudaSuccess) throw 0;

					LoadFromFile(dirPath, params_OptiX.epoch, "v_sh_5", buf, sizeof(float4) * params_OptiX.numberOfGaussians);
					error_CUDA = cudaMemcpy(params_OptiX.v_SH_5, buf, sizeof(float4) * params_OptiX.numberOfGaussians, cudaMemcpyHostToDevice);
					if (error_CUDA != cudaSuccess) throw 0;

					LoadFromFile(dirPath, params_OptiX.epoch, "v_sh_6", buf, sizeof(float4) * params_OptiX.numberOfGaussians);
					error_CUDA = cudaMemcpy(params_OptiX.v_SH_6, buf, sizeof(float4) * params_OptiX.numberOfGaussians, cudaMemcpyHostToDevice);
					if (error_CUDA != cudaSuccess) throw 0;

					if constexpr (SH_degree >= 3) {
						LoadFromFile(dirPath, params_OptiX.epoch, "v_sh_7", buf, sizeof(float4) * params_OptiX.numberOfGaussians);
						error_CUDA = cudaMemcpy(params_OptiX.v_SH_7, buf, sizeof(float4) * params_OptiX.numberOfGaussians, cudaMemcpyHostToDevice);
						if (error_CUDA != cudaSuccess) throw 0;

						LoadFromFile(dirPath, params_OptiX.epoch, "v_sh_8", buf, sizeof(float4) * params_OptiX.numberOfGaussians);
						error_CUDA = cudaMemcpy(params_OptiX.v_SH_8, buf, sizeof(float4) * params_OptiX.numberOfGaussians, cudaMemcpyHostToDevice);
						if (error_CUDA != cudaSuccess) throw 0;

						LoadFromFile(dirPath, params_OptiX.epoch, "v_sh_9", buf, sizeof(float4) * params_OptiX.numberOfGaussians);
						error_CUDA = cudaMemcpy(params_OptiX.v_SH_9, buf, sizeof(float4) * params_OptiX.numberOfGaussians, cudaMemcpyHostToDevice);
						if (error_CUDA != cudaSuccess) throw 0;

						LoadFromFile(dirPath, params_OptiX.epoch, "v_sh_10", buf, sizeof(float4) * params_OptiX.numberOfGaussians);
						error_CUDA = cudaMemcpy(params_OptiX.v_SH_10, buf, sizeof(float4) * params_OptiX.numberOfGaussians, cudaMemcpyHostToDevice);
						if (error_CUDA != cudaSuccess) throw 0;

						LoadFromFile(dirPath, params_OptiX.epoch, "v_sh_11", buf, sizeof(float4) * params_OptiX.numberOfGaussians);
						error_CUDA = cudaMemcpy(params_OptiX.v_SH_11, buf, sizeof(float4) * params_OptiX.numberOfGaussians, cudaMemcpyHostToDevice);
						if (error_CUDA != cudaSuccess) throw 0;

						if constexpr (SH_degree >= 4) {
							LoadFromFile(dirPath, params_OptiX.epoch, "v_sh_12", buf, sizeof(float4) * params_OptiX.numberOfGaussians);
							error_CUDA = cudaMemcpy(params_OptiX.v_SH_12, buf, sizeof(float4) * params_OptiX.numberOfGaussians, cudaMemcpyHostToDevice);
							if (error_CUDA != cudaSuccess) throw 0;

							LoadFromFile(dirPath, params_OptiX.epoch, "v_sh_13", buf, sizeof(float4) * params_OptiX.numberOfGaussians);
							error_CUDA = cudaMemcpy(params_OptiX.v_SH_13, buf, sizeof(float4) * params_OptiX.numberOfGaussians, cudaMemcpyHostToDevice);
							if (error_CUDA != cudaSuccess) throw 0;

							LoadFromFile(dirPath, params_OptiX.epoch, "v_sh_14", buf, sizeof(float4) * params_OptiX.numberOfGaussians);
							error_CUDA = cudaMemcpy(params_OptiX.v_SH_14, buf, sizeof(float4) * params_OptiX.numberOfGaussians, cudaMemcpyHostToDevice);
							if (error_CUDA != cudaSuccess) throw 0;

							LoadFromFile(dirPath, params_OptiX.epoch, "v_sh_15", buf, sizeof(float4) * params_OptiX.numberOfGaussians);
							error_CUDA = cudaMemcpy(params_OptiX.v_SH_15, buf, sizeof(float4) * params_OptiX.numberOfGaussians, cudaMemcpyHostToDevice);
							if (error_CUDA != cudaSuccess) throw 0;

							LoadFromFile(dirPath, params_OptiX.epoch, "v_sh_16", buf, sizeof(float4) * params_OptiX.numberOfGaussians);
							error_CUDA = cudaMemcpy(params_OptiX.v_SH_16, buf, sizeof(float4) * params_OptiX.numberOfGaussians, cudaMemcpyHostToDevice);
							if (error_CUDA != cudaSuccess) throw 0;

							LoadFromFile(dirPath, params_OptiX.epoch, "v_sh_17", buf, sizeof(float4) * params_OptiX.numberOfGaussians);
							error_CUDA = cudaMemcpy(params_OptiX.v_SH_17, buf, sizeof(float4) * params_OptiX.numberOfGaussians, cudaMemcpyHostToDevice);
							if (error_CUDA != cudaSuccess) throw 0;

							LoadFromFile(dirPath, params_OptiX.epoch, "v_sh_18", buf, sizeof(float4) * params_OptiX.numberOfGaussians);
							error_CUDA = cudaMemcpy(params_OptiX.v_SH_18, buf, sizeof(float4) * params_OptiX.numberOfGaussians, cudaMemcpyHostToDevice);
							if (error_CUDA != cudaSuccess) throw 0;
						} else {
							// !!! !!! !!!
							LoadFromFile(dirPath, params_OptiX.epoch, "v_sh_12", buf, sizeof(float) * params_OptiX.numberOfGaussians);
							error_CUDA = cudaMemcpy(params_OptiX.v_SH_12, buf, sizeof(float) * params_OptiX.numberOfGaussians, cudaMemcpyHostToDevice);
							if (error_CUDA != cudaSuccess) throw 0;
						}
					}
				} else {
					// !!! !!! !!!
					LoadFromFile(dirPath, params_OptiX.epoch, "v_sh_3", buf, sizeof(float) * params_OptiX.numberOfGaussians);
					error_CUDA = cudaMemcpy(params_OptiX.v_SH_3, buf, sizeof(float) * params_OptiX.numberOfGaussians, cudaMemcpyHostToDevice);
					if (error_CUDA != cudaSuccess) throw 0;
				}
			}

			// *** *** *** *** ***

			// !!! !!! !!!
			LoadFromFile(dirPath, params_OptiX.epoch, "counter", buf, sizeof(unsigned) * 1);
			error_CUDA = cudaMemcpy(params_OptiX.counter2, buf, sizeof(unsigned) * 1, cudaMemcpyHostToDevice);
			if (error_CUDA != cudaSuccess) throw 0;

			// *** *** *** *** ***

			free(buf);
		}

		//**********************************************************************************************
		//* SSIM                                                                                       *
		// *********************************************************************************************

		const int kernel_size = 11;
		const int kernel_radius = kernel_size >> 1;
		const REAL_G sigma = ((REAL_G)1.5);

		int arraySizeReal = (params_OptiX.width + (kernel_size - 1)) * (params_OptiX.height + (kernel_size - 1)); // !!! !!! !!!
		int arraySizeComplex = (((params_OptiX.width + (kernel_size - 1)) >> 1) + 1) * (params_OptiX.height + (kernel_size - 1)); // !!! !!! !!!

		REAL_G *buf;
		buf = (REAL_G *)malloc(sizeof(REAL_G) * arraySizeReal);
		if (buf == NULL) throw 0;

		cufftResult error_CUFFT;

		// ************************************************************************************************

		error_CUFFT = cufftPlan2d(&params_OptiX.planr2c, params_OptiX.height + (kernel_size - 1), params_OptiX.width + (kernel_size - 1), REAL_TO_COMPLEX_G);
		if (error_CUFFT != CUFFT_SUCCESS) throw 0;

		error_CUFFT = cufftPlan2d(&params_OptiX.planc2r, params_OptiX.height + (kernel_size - 1), params_OptiX.width + (kernel_size - 1), COMPLEX_TO_REAL_G);
		if (error_CUFFT != CUFFT_SUCCESS) throw 0;

		// ************************************************************************************************

		error_CUDA = cudaMalloc(&params_OptiX.F_1, sizeof(COMPLEX_G) * arraySizeComplex);
		if (error_CUDA != cudaSuccess) throw 0;

		error_CUDA = cudaMalloc(&params_OptiX.F_2, sizeof(COMPLEX_G) * arraySizeComplex);
		if (error_CUDA != cudaSuccess) throw 0;

		// ************************************************************************************************

		#ifndef SSIM_REDUCE_MEMORY_OVERHEAD
			error_CUDA = cudaMalloc(&params_OptiX.bitmap_ref_R, sizeof(REAL_G) * arraySizeReal * params.NUMBER_OF_POSES);
			if (error_CUDA != cudaSuccess) throw 0;

			error_CUDA = cudaMalloc(&params_OptiX.bitmap_ref_G, sizeof(REAL_G) * arraySizeReal * params.NUMBER_OF_POSES);
			if (error_CUDA != cudaSuccess) throw 0;

			error_CUDA = cudaMalloc(&params_OptiX.bitmap_ref_B, sizeof(REAL_G) * arraySizeReal * params.NUMBER_OF_POSES);
			if (error_CUDA != cudaSuccess) throw 0;

			error_CUDA = cudaMalloc(&params_OptiX.mu_bitmap_ref_R, sizeof(REAL_G) * arraySizeReal * params.NUMBER_OF_POSES);
			if (error_CUDA != cudaSuccess) throw 0;

			error_CUDA = cudaMalloc(&params_OptiX.mu_bitmap_ref_G, sizeof(REAL_G) * arraySizeReal * params.NUMBER_OF_POSES);
			if (error_CUDA != cudaSuccess) throw 0;

			error_CUDA = cudaMalloc(&params_OptiX.mu_bitmap_ref_B, sizeof(REAL_G) * arraySizeReal * params.NUMBER_OF_POSES);
			if (error_CUDA != cudaSuccess) throw 0;
		#else
			error_CUDA = cudaMalloc(&params_OptiX.bitmap_ref_R, sizeof(REAL_G) * arraySizeReal * 1);
			if (error_CUDA != cudaSuccess) throw 0;

			error_CUDA = cudaMalloc(&params_OptiX.bitmap_ref_G, sizeof(REAL_G) * arraySizeReal * 1);
			if (error_CUDA != cudaSuccess) throw 0;

			error_CUDA = cudaMalloc(&params_OptiX.bitmap_ref_B, sizeof(REAL_G) * arraySizeReal * 1);
			if (error_CUDA != cudaSuccess) throw 0;

			error_CUDA = cudaMalloc(&params_OptiX.mu_bitmap_ref_R, sizeof(REAL_G) * arraySizeReal * 1);
			if (error_CUDA != cudaSuccess) throw 0;

			error_CUDA = cudaMalloc(&params_OptiX.mu_bitmap_ref_G, sizeof(REAL_G) * arraySizeReal * 1);
			if (error_CUDA != cudaSuccess) throw 0;

			error_CUDA = cudaMalloc(&params_OptiX.mu_bitmap_ref_B, sizeof(REAL_G) * arraySizeReal * 1);
			if (error_CUDA != cudaSuccess) throw 0;
		#endif

		error_CUDA = cudaMalloc(&params_OptiX.mu_bitmap_out_bitmap_ref_R, sizeof(REAL_G) * arraySizeReal);
		if (error_CUDA != cudaSuccess) throw 0;

		error_CUDA = cudaMalloc(&params_OptiX.mu_bitmap_out_bitmap_ref_G, sizeof(REAL_G) * arraySizeReal);
		if (error_CUDA != cudaSuccess) throw 0;

		error_CUDA = cudaMalloc(&params_OptiX.mu_bitmap_out_bitmap_ref_B, sizeof(REAL_G) * arraySizeReal);
		if (error_CUDA != cudaSuccess) throw 0;

		#ifndef SSIM_REDUCE_MEMORY_OVERHEAD
			error_CUDA = cudaMalloc(&params_OptiX.mu_bitmap_ref_R_square, sizeof(REAL_G) * arraySizeReal * params.NUMBER_OF_POSES);
			if (error_CUDA != cudaSuccess) throw 0;

			error_CUDA = cudaMalloc(&params_OptiX.mu_bitmap_ref_G_square, sizeof(REAL_G) * arraySizeReal * params.NUMBER_OF_POSES);
			if (error_CUDA != cudaSuccess) throw 0;

			error_CUDA = cudaMalloc(&params_OptiX.mu_bitmap_ref_B_square, sizeof(REAL_G) * arraySizeReal * params.NUMBER_OF_POSES);
			if (error_CUDA != cudaSuccess) throw 0;
		#else
			error_CUDA = cudaMalloc(&params_OptiX.mu_bitmap_ref_R_square, sizeof(REAL_G) * arraySizeReal * 1);
			if (error_CUDA != cudaSuccess) throw 0;

			error_CUDA = cudaMalloc(&params_OptiX.mu_bitmap_ref_G_square, sizeof(REAL_G) * arraySizeReal * 1);
			if (error_CUDA != cudaSuccess) throw 0;

			error_CUDA = cudaMalloc(&params_OptiX.mu_bitmap_ref_B_square, sizeof(REAL_G) * arraySizeReal * 1);
			if (error_CUDA != cudaSuccess) throw 0;
		#endif

		error_CUDA = cudaMalloc(&params_OptiX.bitmap_out_R, sizeof(REAL_G) * arraySizeReal);
		if (error_CUDA != cudaSuccess) throw 0;

		error_CUDA = cudaMalloc(&params_OptiX.bitmap_out_G, sizeof(REAL_G) * arraySizeReal);
		if (error_CUDA != cudaSuccess) throw 0;

		error_CUDA = cudaMalloc(&params_OptiX.bitmap_out_B, sizeof(REAL_G) * arraySizeReal);
		if (error_CUDA != cudaSuccess) throw 0;

		error_CUDA = cudaMalloc(&params_OptiX.mu_bitmap_out_R, sizeof(REAL_G) * arraySizeReal);
		if (error_CUDA != cudaSuccess) throw 0;

		error_CUDA = cudaMalloc(&params_OptiX.mu_bitmap_out_G, sizeof(REAL_G) * arraySizeReal);
		if (error_CUDA != cudaSuccess) throw 0;

		error_CUDA = cudaMalloc(&params_OptiX.mu_bitmap_out_B, sizeof(REAL_G) * arraySizeReal);
		if (error_CUDA != cudaSuccess) throw 0;

		error_CUDA = cudaMalloc(&params_OptiX.mu_bitmap_out_R_square, sizeof(REAL_G) * arraySizeReal);
		if (error_CUDA != cudaSuccess) throw 0;

		error_CUDA = cudaMalloc(&params_OptiX.mu_bitmap_out_G_square, sizeof(REAL_G) * arraySizeReal);
		if (error_CUDA != cudaSuccess) throw 0;

		error_CUDA = cudaMalloc(&params_OptiX.mu_bitmap_out_B_square, sizeof(REAL_G) * arraySizeReal);
		if (error_CUDA != cudaSuccess) throw 0;

		// ************************************************************************************************

		// Create Gaussian kernel
		memset(buf, 0, sizeof(REAL_G) * arraySizeReal);
		REAL_G sum = ((REAL_G)0.0);
		for (int i = 0; i < kernel_size; ++i) {
			for (int j = 0; j < kernel_size; ++j) {
				REAL_G tmp = exp((-(((j - kernel_radius) * (j - kernel_radius)) + ((i - kernel_radius) * (i - kernel_radius)))) / (((REAL_G)2.0) * sigma * sigma));
				sum += tmp;
				buf[(i * (params_OptiX.width + (kernel_size - 1))) + j] = tmp;
			}
		}
		for (int i = 0; i < kernel_size; ++i) {
			for (int j = 0; j < kernel_size; ++j)
				buf[(i * (params_OptiX.width + (kernel_size - 1))) + j] /= sum;
		}

		// dev_mu_bitmap_out_bitmap_ref_R = dev_kernel
		error_CUDA = cudaMemcpy(params_OptiX.mu_bitmap_out_bitmap_ref_R, buf, sizeof(REAL_G) * arraySizeReal, cudaMemcpyHostToDevice);
		if (error_CUDA != cudaSuccess) throw 0;

		error_CUFFT = DFFT_G(params_OptiX.planr2c, params_OptiX.mu_bitmap_out_bitmap_ref_R, params_OptiX.F_1);
		if (error_CUFFT != CUFFT_SUCCESS) throw 0;

		// ************************************************************************************************

		#ifndef SSIM_REDUCE_MEMORY_OVERHEAD
			for (int pose = 0; pose < params.NUMBER_OF_POSES; ++pose) {
				// R channel
				for (int i = 0; i < params_OptiX.height; ++i) {
					for (int j = 0; j < params_OptiX.width; ++j) {
						unsigned char R = params.bitmap_ref[(pose * (params_OptiX.width * params_OptiX.height)) + ((i * params_OptiX.width) + j)] >> 16;
						buf[(i * (params_OptiX.width + (kernel_size - 1))) + j] = R / ((REAL_G)255.0);
					}
				}

				error_CUDA = cudaMemcpy(params_OptiX.bitmap_ref_R + (pose * arraySizeReal), buf, sizeof(REAL_G) * arraySizeReal, cudaMemcpyHostToDevice);
				if (error_CUDA != cudaSuccess) throw 0;

				// *** *** *** *** ***

				// G channel
				for (int i = 0; i < params_OptiX.height; ++i) {
					for (int j = 0; j < params_OptiX.width; ++j) {
						unsigned char G = (params.bitmap_ref[(pose * (params_OptiX.width * params_OptiX.height)) + ((i * params_OptiX.width) + j)] >> 8) & 255;
						buf[(i * (params_OptiX.width + (kernel_size - 1))) + j] = G / ((REAL_G)255.0);
					}
				}

				error_CUDA = cudaMemcpy(params_OptiX.bitmap_ref_G + (pose * arraySizeReal), buf, sizeof(REAL_G) * arraySizeReal, cudaMemcpyHostToDevice);
				if (error_CUDA != cudaSuccess) throw 0;

				// *** *** *** *** ***

				// B channel
				for (int i = 0; i < params_OptiX.height; ++i) {
					for (int j = 0; j < params_OptiX.width; ++j) {
						unsigned char B = params.bitmap_ref[(pose * (params_OptiX.width * params_OptiX.height)) + ((i * params_OptiX.width) + j)] & 255;
						buf[(i * (params_OptiX.width + (kernel_size - 1))) + j] = B / ((REAL_G)255.0);
					}
				}

				error_CUDA = cudaMemcpy(params_OptiX.bitmap_ref_B + (pose * arraySizeReal), buf, sizeof(REAL_G) * arraySizeReal, cudaMemcpyHostToDevice);
				if (error_CUDA != cudaSuccess) throw 0;
			}

			// ************************************************************************************************

			// Compute mu's for reference images
			for (int pose = 0; pose < params.NUMBER_OF_POSES; ++pose) {
				// R channel
				error_CUFFT = DFFT_G(params_OptiX.planr2c, params_OptiX.bitmap_ref_R + (pose * arraySizeReal), params_OptiX.F_2);
				if (error_CUFFT != CUFFT_SUCCESS) throw 0;

				MultiplyPointwiseComplex<<<(arraySizeComplex + 255) >> 8, 256>>>(params_OptiX.F_2, params_OptiX.F_1, params_OptiX.F_2, arraySizeComplex);
				error_CUDA = cudaGetLastError();
				if (error_CUDA != cudaSuccess) throw 0;

				error_CUFFT = IDFFT_G(params_OptiX.planc2r, params_OptiX.F_2, params_OptiX.mu_bitmap_ref_R + (pose * arraySizeReal));
				if (error_CUFFT != CUFFT_SUCCESS) throw 0;

				// *** *** *** *** ***

				// G channel
				error_CUFFT = DFFT_G(params_OptiX.planr2c, params_OptiX.bitmap_ref_G + (pose * arraySizeReal), params_OptiX.F_2);
				if (error_CUFFT != CUFFT_SUCCESS) throw 0;

				MultiplyPointwiseComplex<<<(arraySizeComplex + 255) >> 8, 256>>>(params_OptiX.F_2, params_OptiX.F_1, params_OptiX.F_2, arraySizeComplex);
				error_CUDA = cudaGetLastError();
				if (error_CUDA != cudaSuccess) throw 0;

				error_CUFFT = IDFFT_G(params_OptiX.planc2r, params_OptiX.F_2, params_OptiX.mu_bitmap_ref_G + (pose * arraySizeReal));
				if (error_CUFFT != CUFFT_SUCCESS) throw 0;

				// *** *** *** *** ***

				// B channel
				error_CUFFT = DFFT_G(params_OptiX.planr2c, params_OptiX.bitmap_ref_B + (pose * arraySizeReal), params_OptiX.F_2);
				if (error_CUFFT != CUFFT_SUCCESS) throw 0;

				MultiplyPointwiseComplex<<<(arraySizeComplex + 255) >> 8, 256>>>(params_OptiX.F_2, params_OptiX.F_1, params_OptiX.F_2, arraySizeComplex);
				error_CUDA = cudaGetLastError();
				if (error_CUDA != cudaSuccess) throw 0;

				error_CUFFT = IDFFT_G(params_OptiX.planc2r, params_OptiX.F_2, params_OptiX.mu_bitmap_ref_B + (pose * arraySizeReal));
				if (error_CUFFT != CUFFT_SUCCESS) throw 0;
			}

			// ************************************************************************************************

			// Compute mu's for reference images square
			for (int pose = 0; pose < params.NUMBER_OF_POSES; ++pose) {
				// R channel
				// mu_bitmap_out_bitmap_ref_R = bitmap_ref_R * bitmap_ref_R
				MultiplyPointwiseReal<<<(arraySizeReal + 255) >> 8, 256>>>(
					params_OptiX.bitmap_ref_R + (pose * arraySizeReal),
					params_OptiX.bitmap_ref_R + (pose * arraySizeReal),
					params_OptiX.mu_bitmap_out_bitmap_ref_R,
					arraySizeReal
					);
				error_CUDA = cudaGetLastError();
				if (error_CUDA != cudaSuccess) throw 0;

				error_CUFFT = DFFT_G(params_OptiX.planr2c, params_OptiX.mu_bitmap_out_bitmap_ref_R, params_OptiX.F_2);
				if (error_CUFFT != CUFFT_SUCCESS) throw 0;

				MultiplyPointwiseComplex<<<(arraySizeComplex + 255) >> 8, 256>>>(params_OptiX.F_2, params_OptiX.F_1, params_OptiX.F_2, arraySizeComplex);
				error_CUDA = cudaGetLastError();
				if (error_CUDA != cudaSuccess) throw 0;

				error_CUFFT = IDFFT_G(params_OptiX.planc2r, params_OptiX.F_2, params_OptiX.mu_bitmap_ref_R_square + (pose * arraySizeReal));
				if (error_CUFFT != CUFFT_SUCCESS) throw 0;

				// *** *** *** *** ***

				// G channel
				// mu_bitmap_out_bitmap_ref_R = bitmap_ref_G * bitmap_ref_G
				MultiplyPointwiseReal<<<(arraySizeReal + 255) >> 8, 256>>>(
					params_OptiX.bitmap_ref_G + (pose * arraySizeReal),
					params_OptiX.bitmap_ref_G + (pose * arraySizeReal),
					params_OptiX.mu_bitmap_out_bitmap_ref_R,
					arraySizeReal
					);
				error_CUDA = cudaGetLastError();
				if (error_CUDA != cudaSuccess) throw 0;

				error_CUFFT = DFFT_G(params_OptiX.planr2c, params_OptiX.mu_bitmap_out_bitmap_ref_R, params_OptiX.F_2);
				if (error_CUFFT != CUFFT_SUCCESS) throw 0;

				MultiplyPointwiseComplex<<<(arraySizeComplex + 255) >> 8, 256>>>(params_OptiX.F_2, params_OptiX.F_1, params_OptiX.F_2, arraySizeComplex);
				error_CUDA = cudaGetLastError();
				if (error_CUDA != cudaSuccess) throw 0;

				error_CUFFT = IDFFT_G(params_OptiX.planc2r, params_OptiX.F_2, params_OptiX.mu_bitmap_ref_G_square + (pose * arraySizeReal));
				if (error_CUFFT != CUFFT_SUCCESS) throw 0;

				// *** *** *** *** ***

				// B channel
				// mu_bitmap_out_bitmap_ref_R = bitmap_ref_B * bitmap_ref_B
				MultiplyPointwiseReal<<<(arraySizeReal + 255) >> 8, 256>>>(
					params_OptiX.bitmap_ref_B + (pose * arraySizeReal),
					params_OptiX.bitmap_ref_B + (pose * arraySizeReal),
					params_OptiX.mu_bitmap_out_bitmap_ref_R,
					arraySizeReal
					);
				error_CUDA = cudaGetLastError();
				if (error_CUDA != cudaSuccess) throw 0;

				error_CUFFT = DFFT_G(params_OptiX.planr2c, params_OptiX.mu_bitmap_out_bitmap_ref_R, params_OptiX.F_2);
				if (error_CUFFT != CUFFT_SUCCESS) throw 0;

				MultiplyPointwiseComplex<<<(arraySizeComplex + 255) >> 8, 256>>>(params_OptiX.F_2, params_OptiX.F_1, params_OptiX.F_2, arraySizeComplex);
				error_CUDA = cudaGetLastError();
				if (error_CUDA != cudaSuccess) throw 0;

				error_CUFFT = IDFFT_G(params_OptiX.planc2r, params_OptiX.F_2, params_OptiX.mu_bitmap_ref_B_square + (pose * arraySizeReal));
				if (error_CUFFT != CUFFT_SUCCESS) throw 0;
			}
		#endif

		// *********************************************************************************************

		return true;
	} catch (...) {
		return false;
	}
}

// *************************************************************************************************

template bool InitializeOptiXOptimizer<0>(
	SRenderParams<0> &params,
	SOptiXRenderParams<0> &params_OptiX,
	char *dirPath
);

// *************************************************************************************************

template bool InitializeOptiXOptimizer<1>(
	SRenderParams<1> &params,
	SOptiXRenderParams<1> &params_OptiX,
	char *dirPath
);

// *************************************************************************************************

template bool InitializeOptiXOptimizer<2>(
	SRenderParams<2> &params,
	SOptiXRenderParams<2> &params_OptiX,
	char *dirPath
);

// *************************************************************************************************

template bool InitializeOptiXOptimizer<3>(
	SRenderParams<3> &params,
	SOptiXRenderParams<3> &params_OptiX,
	char *dirPath
);

// *************************************************************************************************

template bool InitializeOptiXOptimizer<4>(
	SRenderParams<4> &params,
	SOptiXRenderParams<4> &params_OptiX,
	char *dirPath
);