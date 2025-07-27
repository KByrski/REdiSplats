#include "Header.cuh"

// *************************************************************************************************

template<int SH_degree>
bool RenderOptiX(SOptiXRenderParams<SH_degree>& params_OptiX, bool inference = true) {
	cudaError_t error_CUDA;
	OptixResult error_OptiX;

	// *********************************************************************************************

	LaunchParams<SH_degree> launchParams;
	launchParams.bitmap = params_OptiX.bitmap_out_device;
	launchParams.width = params_OptiX.width;
	launchParams.height = params_OptiX.height;
	launchParams.O = params_OptiX.O;
	launchParams.R = params_OptiX.R;
	launchParams.D = params_OptiX.D;
	launchParams.F = params_OptiX.F;
	launchParams.double_tan_half_fov_x = params_OptiX.double_tan_half_fov_x;
	launchParams.double_tan_half_fov_y = params_OptiX.double_tan_half_fov_y;
	launchParams.traversable = params_OptiX.IAS;
	launchParams.GC_part_1 = params_OptiX.GC_part_1_1;
	launchParams.GC_part_2 = params_OptiX.GC_part_2_1;
	launchParams.GC_part_3 = params_OptiX.GC_part_3_1;
	launchParams.GC_part_4 = params_OptiX.GC_part_4_1;

	launchParams.inference = inference;

	// !!! !!! !!!
	launchParams.max_RSH = params_OptiX.max_RSH;
	launchParams.max_GSH = params_OptiX.max_GSH;
	launchParams.max_BSH = params_OptiX.max_BSH;
	// !!! !!! !!!

	// *** *** *** *** ***

	// Spherical harmonics
	if constexpr (SH_degree >= 1) {
		launchParams.GC_SH_1 = params_OptiX.GC_SH_1;
		launchParams.GC_SH_2 = params_OptiX.GC_SH_2;
		launchParams.GC_SH_3 = params_OptiX.GC_SH_3;

		if constexpr (SH_degree >= 2) {
			launchParams.GC_SH_4 = params_OptiX.GC_SH_4;
			launchParams.GC_SH_5 = params_OptiX.GC_SH_5;
			launchParams.GC_SH_6 = params_OptiX.GC_SH_6;

			if constexpr (SH_degree >= 3) {
				launchParams.GC_SH_7 = params_OptiX.GC_SH_7;
				launchParams.GC_SH_8 = params_OptiX.GC_SH_8;
				launchParams.GC_SH_9 = params_OptiX.GC_SH_9;
				launchParams.GC_SH_10 = params_OptiX.GC_SH_10;
				launchParams.GC_SH_11 = params_OptiX.GC_SH_11;
				launchParams.GC_SH_12 = params_OptiX.GC_SH_12;

				if constexpr (SH_degree >= 4) {
					launchParams.GC_SH_13 = params_OptiX.GC_SH_13;
					launchParams.GC_SH_14 = params_OptiX.GC_SH_14;
					launchParams.GC_SH_15 = params_OptiX.GC_SH_15;
					launchParams.GC_SH_16 = params_OptiX.GC_SH_16;
					launchParams.GC_SH_17 = params_OptiX.GC_SH_17;
					launchParams.GC_SH_18 = params_OptiX.GC_SH_18;
				}
			}
		}
	}

	// *** *** *** *** ***

	launchParams.Gaussians_indices = params_OptiX.Gaussians_indices;
	launchParams.bitmap_out_R = params_OptiX.bitmap_out_R;
	launchParams.bitmap_out_G = params_OptiX.bitmap_out_G;
	launchParams.bitmap_out_B = params_OptiX.bitmap_out_B;
	launchParams.ray_termination_T_threshold = ray_termination_T_threshold_host; // !!! !!! !!!
	launchParams.chi_square_squared_radius = chi_square_squared_radius_host; // !!! !!! !!!
	launchParams.max_Gaussians_per_ray = max_Gaussians_per_ray_host; // !!! !!! !!!
	launchParams.bg_color_R = bg_color_R_host; // !!! !!! !!!
	launchParams.bg_color_G = bg_color_G_host; // !!! !!! !!!
	launchParams.bg_color_B = bg_color_B_host; // !!! !!! !!!

	error_CUDA = cudaMemcpy(params_OptiX.launchParamsBuffer, &launchParams, sizeof(LaunchParams<SH_degree>) * 1, cudaMemcpyHostToDevice);
	if (error_CUDA != cudaSuccess) goto Error;

	error_OptiX = optixLaunch(
		params_OptiX.pipeline,
		0,
		(CUdeviceptr)params_OptiX.launchParamsBuffer,
		sizeof(LaunchParams<SH_degree>) * 1,
		params_OptiX.sbt,
		params_OptiX.width,
		params_OptiX.height,
		1
	);
	if (error_OptiX != OPTIX_SUCCESS) goto Error;

	cudaDeviceSynchronize();
	error_CUDA = cudaGetLastError();
	if (error_CUDA != cudaSuccess) goto Error;

	// *********************************************************************************************

	if (params_OptiX.copyBitmapToHostMemory) {
		error_CUDA = cudaMemcpy(
			params_OptiX.bitmap_out_host,
			params_OptiX.bitmap_out_device,
			sizeof(unsigned) * params_OptiX.width * params_OptiX.height,
			cudaMemcpyDeviceToHost
		);
		if (error_CUDA != cudaSuccess) goto Error;
	}

	// *********************************************************************************************

	return true;
Error:
	return false;
}

// *************************************************************************************************

bool RenderOptiXSH0(SOptiXRenderParams<0>& params_OptiX, bool inference = true) {
	return RenderOptiX<0>(params_OptiX, inference);
}

// *************************************************************************************************

bool RenderOptiXSH1(SOptiXRenderParams<1>& params_OptiX, bool inference = true) {
	return RenderOptiX<1>(params_OptiX, inference);
}

// *************************************************************************************************

bool RenderOptiXSH2(SOptiXRenderParams<2>& params_OptiX, bool inference = true) {
	return RenderOptiX<2>(params_OptiX, inference);
}

// *************************************************************************************************

bool RenderOptiXSH3(SOptiXRenderParams<3>& params_OptiX, bool inference = true) {
	return RenderOptiX<3>(params_OptiX, inference);
}

// *************************************************************************************************

bool RenderOptiXSH4(SOptiXRenderParams<4>& params_OptiX, bool inference = true) {
	return RenderOptiX<4>(params_OptiX, inference);
}