#define _USE_MATH_DEFINES

#include "cuda_runtime.h"
#include "cufft.h"
#include "device_launch_parameters.h"

#include "optix.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// *** *** *** *** ***

// !!! !!! !!!
const int NUMBER_OF_SIDES = 8;

#define SSIM_REDUCE_MEMORY_OVERHEAD
//#define RENDERER_OPTIX_USE_DOUBLE_PRECISION
//#define GRADIENT_OPTIX_USE_DOUBLE_PRECISION
//#define OPTIMIZER_OPTIX_USE_DOUBLE_PRECISION // to be implemented

#ifndef RENDERER_OPTIX_USE_DOUBLE_PRECISION
	typedef cufftReal REAL_R;
	typedef float3 REAL3_R;
	
	#define EXP_R(x) expf(x)
	#define MAD_R(x, y, z) __fmaf_rn(x, y, z)
	#define make_REAL3_R(x, y, z) make_float3(x, y, z)
	#define MIN_R(x, y) fminf(x, y)
	#define MAX_R(x, y) fmaxf(x, y)
	#define RINT_R(_X) rintf(_X)
	#define SQRT_R(x) sqrtf(x)
	#define TAN_R(_X) tanf(_X)
	#define RCP_R(x) __frcp_rn(x)
	#define COPYSIGN_R(x, y) copysignf(x, y)
#else
	typedef cufftDoubleReal REAL_R;
	typedef double3 REAL3_R;
	
	#define EXP_R(x) exp(x)
	#define MAD_R(x, y, z) __fma_rn(x, y, z)
	#define make_REAL3_R(x, y, z) make_double3(x, y, z)
	#define MIN_R(x, y) fmin(x, y)
	#define MAX_R(x, y) fmax(x, y)
	#define RINT_R(_X) rint(_X)
	#define SQRT_R(x) sqrt(x)
	#define TAN_R(_X) tan(_X)
	#define RCP_R(x) __drcp_rn(x)
	#define COPYSIGN_R(x, y) copysign(x, y)
#endif

#ifndef GRADIENT_OPTIX_USE_DOUBLE_PRECISION
	typedef cufftComplex COMPLEX_G;
	typedef cufftReal REAL_G;
	typedef float2 REAL2_G;
	typedef float3 REAL3_G;
	typedef float4 REAL4_G;

	#define REAL_TO_COMPLEX_G CUFFT_R2C
	#define COMPLEX_TO_REAL_G CUFFT_C2R

	#define DFFT_G(plan, idata, odata) cufftExecR2C(plan, idata, odata)
	#define IDFFT_G(plan, idata, odata) cufftExecC2R(plan, idata, odata)

	#define make_REAL3_G(x, y, z) make_float3(x, y, z)
	#define TAN_G(_X) tanf(_X)
	#define ABS_G(x) fabsf(x)
	#define EXP_G(x) expf(x)
	#define MIN_G(x, y) fminf(x, y)
	#define MAX_G(x, y) fmaxf(x, y)
	#define MAD_G(x, y, z) __fmaf_rn(x, y, z)
#else
	typedef cufftDoubleComplex COMPLEX_G;
	typedef cufftDoubleReal REAL_G;
	typedef double2 REAL2_G;
	typedef double3 REAL3_G;
	typedef double4 REAL4_G;

	#define REAL_TO_COMPLEX_G CUFFT_D2Z
	#define COMPLEX_TO_REAL_G CUFFT_Z2D

	#define DFFT_G(plan, idata, odata) cufftExecD2Z(plan, idata, odata)
	#define IDFFT_G(plan, idata, odata) cufftExecZ2D(plan, idata, odata)

	#define make_REAL3_G(x, y, z) make_double3(x, y, z)
	#define TAN_G(_X) tan(_X)
	#define ABS_G(x) fabs(x)
	#define EXP_G(x) exp(x)
	#define MIN_G(x, y) fmin(x, y)
	#define MAX_G(x, y) fmax(x, y)
	#define MAD_G(x, y, z) __fma_rn(x, y, z)
#endif
// !!! !!! !!!

// *** *** *** *** ***

struct SCamera {
	float Ox; float Oy; float Oz;
	float Rx; float Ry; float Rz;
	float Dx; float Dy; float Dz;
	float Fx; float Fy; float Fz;
};

// *** *** *** *** ***

struct SGaussianComponent {
	float mX, mY, mZ;

	float qr, qi, qj, qk;
	float sX, sY, sZ;

	float R, G, B;
	float alpha;
};

// *** *** *** *** ***

struct SRenderParams {
	float Ox; float Oy; float Oz;
	float Rx; float Ry; float Rz;
	float Dx; float Dy; float Dz;
	float Fx; float Fy; float Fz;
	float double_tan_half_fov_x;
	float double_tan_half_fov_y;
	void* bitmap;
	int w; int h;
	SGaussianComponent* GC;
	int numberOfGaussians;

	// !!! !!! !!!
	SCamera *poses;
	unsigned *bitmap_ref;
	int poseNum;
	int epoch;
	int NUMBER_OF_POSES;
	double loss;
	// !!! !!! !!!
};

// *************************************************************************************************

struct SOptiXRenderParams {
	// RENDERER
	OptixDeviceContext optixContext;
	OptixPipeline pipeline;

	OptixShaderBindingTable *sbt;
	void *raygenRecordsBuffer;
	void *missRecordsBuffer;
	void *hitgroupRecordsBuffer;

	int numberOfGaussians;
	int maxNumberOfGaussians;

	void *aabbBuffer;
	void *compactedSizeBuffer;
	void *tempBuffer;
	unsigned long long tempBufferSize;
	void *outputBuffer;
	unsigned long long outputBufferSize;
	void *asBuffer;
	unsigned long long asBufferSize;

	OptixTraversableHandle asHandle;

	unsigned *bitmap_out_device;
	unsigned *bitmap_out_host;
	unsigned width;
	unsigned height;

	float3 O;
	float3 R, D, F;
	float double_tan_half_fov_x;
	float double_tan_half_fov_y;

	float4 *GC_part_1_1, *GC_part_1_2;
	float4 *GC_part_2_1, *GC_part_2_2;
	float4 *GC_part_3_1, *GC_part_3_2;
	float2 *GC_part_4_1, *GC_part_4_2;

	float4 *m11, *m12;
	float4 *m21, *m22;
	float4 *m31, *m32;
	float2 *m41, *m42;

	float4 *v11, *v12;
	float4 *v21, *v22;
	float4 *v31, *v32;
	float2 *v41, *v42;

	// *** *** *** *** ***

	// OPTIMIZER
	unsigned *bitmap_ref;
	void *dL_dparams_1;
	void *dL_dparams_2;
	void *dL_dparams_3;
	void *dL_dparams_4;
	double *loss_device;
	double loss_host;
	int *Gaussians_indices;

	// !!! !!! !!!
	cufftHandle planr2c;
	cufftHandle planc2r;

	COMPLEX_G *F_1;
	COMPLEX_G *F_2;

	REAL_G *bitmap_ref_R;
	REAL_G *bitmap_ref_G;
	REAL_G *bitmap_ref_B;
	REAL_G *mu_bitmap_ref_R;
	REAL_G *mu_bitmap_ref_G;
	REAL_G *mu_bitmap_ref_B;
	REAL_G *mu_bitmap_out_bitmap_ref_R;
	REAL_G *mu_bitmap_out_bitmap_ref_G;
	REAL_G *mu_bitmap_out_bitmap_ref_B;
	REAL_G *mu_bitmap_ref_R_square;
	REAL_G *mu_bitmap_ref_G_square;
	REAL_G *mu_bitmap_ref_B_square;

	REAL_G *bitmap_out_R;
	REAL_G *bitmap_out_G;
	REAL_G *bitmap_out_B;
	REAL_G *mu_bitmap_out_R;
	REAL_G *mu_bitmap_out_G;
	REAL_G *mu_bitmap_out_B;
	REAL_G *mu_bitmap_out_R_square;
	REAL_G *mu_bitmap_out_G_square;
	REAL_G *mu_bitmap_out_B_square;
	// !!! !!! !!!

	// *** *** *** *** ***

	bool copyBitmapToHostMemory;
	unsigned poseNum;
	unsigned epoch;
	void *counter1; // !!! !!! !!!
	void *counter2; // !!! !!! !!!

	// *** *** *** *** ***

	float2 *Gaussian_as_polygon_vertices;
	int3 *Gaussian_as_polygon_indices;
	float3 *Gaussians_as_polygon_vertices;
	int3 *Gaussians_as_polygon_indices;
};

// *************************************************************************************************

struct SOptiXRenderConfig {
	char *learning_phase;
	char *data_path;
	char *pretrained_model_path;
	char *data_format;

	int start_epoch;
	int end_epoch;

	float lr_RGB;
	float lr_RGB_exponential_decay_coefficient;
	float lr_RGB_final;

	float lr_alpha;
	float lr_alpha_exponential_decay_coefficient;
	float lr_alpha_final;

	float lr_m;
	float lr_m_exponential_decay_coefficient;
	float lr_m_final;
	
	float lr_s;
	float lr_s_exponential_decay_coefficient;
	float lr_s_final;

	float lr_q;
	float lr_q_exponential_decay_coefficient;
	float lr_q_final;

	int densification_frequency;
	int densification_start_epoch;
	int densification_end_epoch;
	float alpha_threshold_for_Gauss_removal;
	float min_s_coefficients_clipping_threshold;
	float max_s_coefficients_clipping_threshold;
	float min_s_norm_threshold_for_Gauss_removal;
	float max_s_norm_threshold_for_Gauss_removal;
	float mu_grad_norm_threshold_for_densification;
	float s_norm_threshold_for_split_strategy;
	float split_ratio;
	float lambda;
	float ray_termination_T_threshold;
	float last_significant_Gauss_alpha_gradient_precision;
	float chi_square_squared_radius;
	int max_Gaussians_per_ray;
	int saving_frequency;
	int evaluation_frequency;
	int evaluation_epoch;
	int max_Gaussians_per_model;
};