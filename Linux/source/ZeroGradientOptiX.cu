#include "Header.cuh"

// *************************************************************************************************

template<int SH_degree>
bool ZeroGradientOptiX(SOptiXRenderParams<SH_degree> &params_OptiX) {
	cudaError_t error_CUDA;

	// *** *** *** *** ***

	try {
		error_CUDA = cudaMemset(params_OptiX.dL_dparams_1, 0, sizeof(REAL4_G) * params_OptiX.numberOfGaussians);
		if (error_CUDA != cudaSuccess) throw 0;

		error_CUDA = cudaMemset(params_OptiX.dL_dparams_2, 0, sizeof(REAL4_G) * params_OptiX.numberOfGaussians);
		if (error_CUDA != cudaSuccess) throw 0;

		error_CUDA = cudaMemset(params_OptiX.dL_dparams_3, 0, sizeof(REAL4_G) * params_OptiX.numberOfGaussians);
		if (error_CUDA != cudaSuccess) throw 0;

		// !!! !!! !!!
		error_CUDA = cudaMemset(params_OptiX.dL_dparams_4, 0, sizeof(REAL2_G) * params_OptiX.numberOfGaussians);
		if (error_CUDA != cudaSuccess) throw 0;

		// Spherical harmonics
		if constexpr (SH_degree >= 1) {
			error_CUDA = cudaMemset(params_OptiX.dL_dparams_SH_1, 0, sizeof(REAL4_G) * params_OptiX.numberOfGaussians);
			if (error_CUDA != cudaSuccess) throw 0;

			error_CUDA = cudaMemset(params_OptiX.dL_dparams_SH_2, 0, sizeof(REAL4_G) * params_OptiX.numberOfGaussians);
			if (error_CUDA != cudaSuccess) throw 0;

			if constexpr (SH_degree >= 2) {
				error_CUDA = cudaMemset(params_OptiX.dL_dparams_SH_3, 0, sizeof(REAL4_G) * params_OptiX.numberOfGaussians);
				if (error_CUDA != cudaSuccess) throw 0;

				error_CUDA = cudaMemset(params_OptiX.dL_dparams_SH_4, 0, sizeof(REAL4_G) * params_OptiX.numberOfGaussians);
				if (error_CUDA != cudaSuccess) throw 0;

				error_CUDA = cudaMemset(params_OptiX.dL_dparams_SH_5, 0, sizeof(REAL4_G) * params_OptiX.numberOfGaussians);
				if (error_CUDA != cudaSuccess) throw 0;

				error_CUDA = cudaMemset(params_OptiX.dL_dparams_SH_6, 0, sizeof(REAL4_G) * params_OptiX.numberOfGaussians);
				if (error_CUDA != cudaSuccess) throw 0;

				if constexpr (SH_degree >= 3) {
					error_CUDA = cudaMemset(params_OptiX.dL_dparams_SH_7, 0, sizeof(REAL4_G) * params_OptiX.numberOfGaussians);
					if (error_CUDA != cudaSuccess) throw 0;

					error_CUDA = cudaMemset(params_OptiX.dL_dparams_SH_8, 0, sizeof(REAL4_G) * params_OptiX.numberOfGaussians);
					if (error_CUDA != cudaSuccess) throw 0;

					error_CUDA = cudaMemset(params_OptiX.dL_dparams_SH_9, 0, sizeof(REAL4_G) * params_OptiX.numberOfGaussians);
					if (error_CUDA != cudaSuccess) throw 0;

					error_CUDA = cudaMemset(params_OptiX.dL_dparams_SH_10, 0, sizeof(REAL4_G) * params_OptiX.numberOfGaussians);
					if (error_CUDA != cudaSuccess) throw 0;

					error_CUDA = cudaMemset(params_OptiX.dL_dparams_SH_11, 0, sizeof(REAL4_G) * params_OptiX.numberOfGaussians);
					if (error_CUDA != cudaSuccess) throw 0;

					if constexpr (SH_degree >= 4) {
						error_CUDA = cudaMemset(params_OptiX.dL_dparams_SH_12, 0, sizeof(REAL4_G) * params_OptiX.numberOfGaussians);
						if (error_CUDA != cudaSuccess) throw 0;

						error_CUDA = cudaMemset(params_OptiX.dL_dparams_SH_13, 0, sizeof(REAL4_G) * params_OptiX.numberOfGaussians);
						if (error_CUDA != cudaSuccess) throw 0;

						error_CUDA = cudaMemset(params_OptiX.dL_dparams_SH_14, 0, sizeof(REAL4_G) * params_OptiX.numberOfGaussians);
						if (error_CUDA != cudaSuccess) throw 0;

						error_CUDA = cudaMemset(params_OptiX.dL_dparams_SH_15, 0, sizeof(REAL4_G) * params_OptiX.numberOfGaussians);
						if (error_CUDA != cudaSuccess) throw 0;

						error_CUDA = cudaMemset(params_OptiX.dL_dparams_SH_16, 0, sizeof(REAL4_G) * params_OptiX.numberOfGaussians);
						if (error_CUDA != cudaSuccess) throw 0;

						error_CUDA = cudaMemset(params_OptiX.dL_dparams_SH_17, 0, sizeof(REAL4_G) * params_OptiX.numberOfGaussians);
						if (error_CUDA != cudaSuccess) throw 0;

						error_CUDA = cudaMemset(params_OptiX.dL_dparams_SH_18, 0, sizeof(REAL4_G) * params_OptiX.numberOfGaussians);
						if (error_CUDA != cudaSuccess) throw 0;
					} else {
						// !!! !!! !!!
						error_CUDA = cudaMemset(params_OptiX.dL_dparams_SH_12, 0, sizeof(REAL_G) * params_OptiX.numberOfGaussians);
						if (error_CUDA != cudaSuccess) throw 0;
					}
				}
			} else {
				// !!! !!! !!!
				error_CUDA = cudaMemset(params_OptiX.dL_dparams_SH_3, 0, sizeof(REAL_G) * params_OptiX.numberOfGaussians);
				if (error_CUDA != cudaSuccess) throw 0;
			}
		}

		// *** *** *** *** ***

		error_CUDA = cudaMemset(params_OptiX.loss_device, 0, sizeof(double) * 1);
		if (error_CUDA != cudaSuccess) throw 0;

		//**********************************************************************************************

		return true;
	} catch (...) {
		return false;
	}
}

// *************************************************************************************************

template bool ZeroGradientOptiX<0>(SOptiXRenderParams<0> &params_OptiX);

// *************************************************************************************************

template bool ZeroGradientOptiX<1>(SOptiXRenderParams<1> &params_OptiX);

// *************************************************************************************************

template bool ZeroGradientOptiX<2>(SOptiXRenderParams<2> &params_OptiX);

// *************************************************************************************************

template bool ZeroGradientOptiX<3>(SOptiXRenderParams<3> &params_OptiX);

// *************************************************************************************************

template bool ZeroGradientOptiX<4>(SOptiXRenderParams<4> &params_OptiX);