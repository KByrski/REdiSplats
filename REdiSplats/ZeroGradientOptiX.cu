#include "Header.cuh"

// *************************************************************************************************

template<int SH_degree>
bool ZeroGradientOptiX(SOptiXRenderParams<SH_degree> &params_OptiX) {
	cudaError_t error_CUDA;

	// *** *** *** *** ***

	error_CUDA = cudaMemset(params_OptiX.dL_dparams_1, 0, sizeof(REAL4_G) * params_OptiX.numberOfGaussians);
	if (error_CUDA != cudaSuccess) goto Error;

	error_CUDA = cudaMemset(params_OptiX.dL_dparams_2, 0, sizeof(REAL4_G) * params_OptiX.numberOfGaussians);
	if (error_CUDA != cudaSuccess) goto Error;

	error_CUDA = cudaMemset(params_OptiX.dL_dparams_3, 0, sizeof(REAL4_G) * params_OptiX.numberOfGaussians);
	if (error_CUDA != cudaSuccess) goto Error;

	// !!! !!! !!!
	error_CUDA = cudaMemset(params_OptiX.dL_dparams_4, 0, sizeof(REAL2_G) * params_OptiX.numberOfGaussians);
	if (error_CUDA != cudaSuccess) goto Error;

	// Spherical harmonics
	if constexpr (SH_degree >= 1) {
		error_CUDA = cudaMemset(params_OptiX.dL_dparams_SH_1, 0, sizeof(REAL4_G) * params_OptiX.numberOfGaussians);
		if (error_CUDA != cudaSuccess) goto Error;

		error_CUDA = cudaMemset(params_OptiX.dL_dparams_SH_2, 0, sizeof(REAL4_G) * params_OptiX.numberOfGaussians);
		if (error_CUDA != cudaSuccess) goto Error;

		if constexpr (SH_degree >= 2) {
			error_CUDA = cudaMemset(params_OptiX.dL_dparams_SH_3, 0, sizeof(REAL4_G) * params_OptiX.numberOfGaussians);
			if (error_CUDA != cudaSuccess) goto Error;

			error_CUDA = cudaMemset(params_OptiX.dL_dparams_SH_4, 0, sizeof(REAL4_G) * params_OptiX.numberOfGaussians);
			if (error_CUDA != cudaSuccess) goto Error;

			error_CUDA = cudaMemset(params_OptiX.dL_dparams_SH_5, 0, sizeof(REAL4_G) * params_OptiX.numberOfGaussians);
			if (error_CUDA != cudaSuccess) goto Error;

			error_CUDA = cudaMemset(params_OptiX.dL_dparams_SH_6, 0, sizeof(REAL4_G) * params_OptiX.numberOfGaussians);
			if (error_CUDA != cudaSuccess) goto Error;

			if constexpr (SH_degree >= 3) {
				error_CUDA = cudaMemset(params_OptiX.dL_dparams_SH_7, 0, sizeof(REAL4_G) * params_OptiX.numberOfGaussians);
				if (error_CUDA != cudaSuccess) goto Error;

				error_CUDA = cudaMemset(params_OptiX.dL_dparams_SH_8, 0, sizeof(REAL4_G) * params_OptiX.numberOfGaussians);
				if (error_CUDA != cudaSuccess) goto Error;

				error_CUDA = cudaMemset(params_OptiX.dL_dparams_SH_9, 0, sizeof(REAL4_G) * params_OptiX.numberOfGaussians);
				if (error_CUDA != cudaSuccess) goto Error;

				error_CUDA = cudaMemset(params_OptiX.dL_dparams_SH_10, 0, sizeof(REAL4_G) * params_OptiX.numberOfGaussians);
				if (error_CUDA != cudaSuccess) goto Error;

				error_CUDA = cudaMemset(params_OptiX.dL_dparams_SH_11, 0, sizeof(REAL4_G) * params_OptiX.numberOfGaussians);
				if (error_CUDA != cudaSuccess) goto Error;

				if constexpr (SH_degree >= 4) {
					error_CUDA = cudaMemset(params_OptiX.dL_dparams_SH_12, 0, sizeof(REAL4_G) * params_OptiX.numberOfGaussians);
					if (error_CUDA != cudaSuccess) goto Error;

					error_CUDA = cudaMemset(params_OptiX.dL_dparams_SH_13, 0, sizeof(REAL4_G) * params_OptiX.numberOfGaussians);
					if (error_CUDA != cudaSuccess) goto Error;

					error_CUDA = cudaMemset(params_OptiX.dL_dparams_SH_14, 0, sizeof(REAL4_G) * params_OptiX.numberOfGaussians);
					if (error_CUDA != cudaSuccess) goto Error;

					error_CUDA = cudaMemset(params_OptiX.dL_dparams_SH_15, 0, sizeof(REAL4_G) * params_OptiX.numberOfGaussians);
					if (error_CUDA != cudaSuccess) goto Error;

					error_CUDA = cudaMemset(params_OptiX.dL_dparams_SH_16, 0, sizeof(REAL4_G) * params_OptiX.numberOfGaussians);
					if (error_CUDA != cudaSuccess) goto Error;

					error_CUDA = cudaMemset(params_OptiX.dL_dparams_SH_17, 0, sizeof(REAL4_G) * params_OptiX.numberOfGaussians);
					if (error_CUDA != cudaSuccess) goto Error;

					error_CUDA = cudaMemset(params_OptiX.dL_dparams_SH_18, 0, sizeof(REAL4_G) * params_OptiX.numberOfGaussians);
					if (error_CUDA != cudaSuccess) goto Error;
				} else {
					// !!! !!! !!!
					error_CUDA = cudaMemset(params_OptiX.dL_dparams_SH_12, 0, sizeof(REAL_G) * params_OptiX.numberOfGaussians);
					if (error_CUDA != cudaSuccess) goto Error;
				}
			}
		} else {
			// !!! !!! !!!
			error_CUDA = cudaMemset(params_OptiX.dL_dparams_SH_3, 0, sizeof(REAL_G) * params_OptiX.numberOfGaussians);
			if (error_CUDA != cudaSuccess) goto Error;
		}
	}

	// *** *** *** *** ***

	error_CUDA = cudaMemset(params_OptiX.loss_device, 0, sizeof(double) * 1);
	if (error_CUDA != cudaSuccess) goto Error;

	//**********************************************************************************************

	return true;
Error:
	return false;
}

// *************************************************************************************************

bool ZeroGradientOptiXSH0(SOptiXRenderParams<0> &params_OptiX) {
	return ZeroGradientOptiX<0>(params_OptiX);
}

// *************************************************************************************************

bool ZeroGradientOptiXSH1(SOptiXRenderParams<1> &params_OptiX) {
	return ZeroGradientOptiX<1>(params_OptiX);
}

// *************************************************************************************************

bool ZeroGradientOptiXSH2(SOptiXRenderParams<2> &params_OptiX) {
	return ZeroGradientOptiX<2>(params_OptiX);
}

// *************************************************************************************************

bool ZeroGradientOptiXSH3(SOptiXRenderParams<3> &params_OptiX) {
	return ZeroGradientOptiX<3>(params_OptiX);
}

// *************************************************************************************************

bool ZeroGradientOptiXSH4(SOptiXRenderParams<4> &params_OptiX) {
	return ZeroGradientOptiX<4>(params_OptiX);
}