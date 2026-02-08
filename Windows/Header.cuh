#define _USE_MATH_DEFINES

#include "cuda.h"
#include "cuda_runtime.h"
#include "cufft.h"
#include "device_launch_parameters.h"

#include "optix.h"
#include "optix_host.h"
#include "optix_stack_size.h"
#include "optix_stubs.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <thrust/device_vector.h>
#include <thrust/scan.h>
#include <thrust/gather.h>
#include <thrust/reduce.h>

// *** *** *** *** ***

// !!! !!! !!!
#define MAX_RAY_LENGTH 1 
//#define USE_DOUBLE_PRECISION 

//#define DEBUG_GRADIENT
#define SSIM_REDUCE_MEMORY_OVERHEAD
//#define RENDERER_OPTIX_USE_DOUBLE_PRECISION
//#define GRADIENT_OPTIX_USE_DOUBLE_PRECISION

#ifndef USE_DOUBLE_PRECISION
typedef cufftReal REAL;
typedef cufftComplex COMPLEX;

#define REAL_TO_COMPLEX CUFFT_R2C
#define COMPLEX_TO_REAL CUFFT_C2R

#define DFFT(plan, idata, odata) cufftExecR2C(plan, idata, odata);
#define IDFFT(plan, idata, odata) cufftExecC2R(plan, idata, odata);
#else
typedef cufftDoubleReal REAL;
typedef cufftDoubleComplex COMPLEX;

#define REAL_TO_COMPLEX CUFFT_D2Z
#define COMPLEX_TO_REAL CUFFT_Z2D

#define DFFT(plan, idata, odata) cufftExecD2Z(plan, idata, odata);
#define IDFFT(plan, idata, odata) cufftExecZ2D(plan, idata, odata);
#endif

#ifndef RENDERER_OPTIX_USE_DOUBLE_PRECISION
typedef cufftReal REAL_R;
typedef float3 REAL3_R;

#define DIVIDE_R(x, y) __fdividef(x, y)
#define EXP_R(x) __expf(x)
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

#define DIVIDE_R(x, y) (x / y);
#define EXP_R(x) exp(x)
#define MAD_R(x, y, z) __fma_rn(x, y, z)
#define make_REAL3_R(x, y, z) make_double3(x, y, z)
#define MIN_R(x, y) fmin(x, y)
#define MAX_R(x, y) fmax(x, y)
#define RINT_R(_X) rint(_X)
#define SQRT_R(x) __dsqrt_rn(x)
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
#define SQRT_G(x) sqrtf(x)
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
#define SQRT_G(x) __dsqrt_rn(x)
#endif
// !!! !!! !!!

// !!! !!! !!! EXPERIMENTAL !!! !!! !!!
struct AABB { float a; float b; float c; float d; float e; float f; };
extern int *needsToBeRemoved_host;
extern __device__ int *needsToBeRemoved;
extern int *Gaussians_indices_after_removal_host;
extern int *scatterBuffer;
// !!! !!! !!! EXPERIMENTAL !!! !!! !!!

// *** *** *** *** ***

struct SReductionOperator_OptixAabb {
	__device__ OptixAabb operator()(const OptixAabb &a, const OptixAabb &b) const;
};

extern __constant__ float scene_extent;

// *** *** *** *** ***

extern float bg_color_R_host;
extern float bg_color_G_host;
extern float bg_color_B_host;
extern int number_of_sides_host;
extern int densification_frequency_host;
extern int densification_start_epoch_host;
extern int densification_end_epoch_host;
extern float min_s_coefficients_clipping_threshold_host;
extern float max_s_coefficients_clipping_threshold_host;
extern float ray_termination_T_threshold_host;
extern float last_significant_Gauss_alpha_gradient_precision_host;
extern float chi_square_squared_radius_host; 
extern int max_Gaussians_per_ray_host;
extern int max_Gaussians_per_model_host;
extern double tmp_arrays_growth_factor_host;

extern __constant__ float bg_color_R;
extern __constant__ float bg_color_G;
extern __constant__ float bg_color_B;

extern __constant__ float lr_SH0;
extern __constant__ float lr_SH0_exponential_decay_coefficient;
extern __constant__ float lr_SH0_final;

extern __constant__ int   SH1_activation_iter;
extern __constant__ float lr_SH1;
extern __constant__ float lr_SH1_exponential_decay_coefficient;
extern __constant__ float lr_SH1_final;

extern __constant__ int   SH2_activation_iter;
extern __constant__ float lr_SH2;
extern __constant__ float lr_SH2_exponential_decay_coefficient;
extern __constant__ float lr_SH2_final;

extern __constant__ int   SH3_activation_iter;
extern __constant__ float lr_SH3;
extern __constant__ float lr_SH3_exponential_decay_coefficient;
extern __constant__ float lr_SH3_final;

extern __constant__ int   SH4_activation_iter;
extern __constant__ float lr_SH4;
extern __constant__ float lr_SH4_exponential_decay_coefficient;
extern __constant__ float lr_SH4_final;

extern __constant__ float lr_alpha;
extern __constant__ float lr_alpha_exponential_decay_coefficient;
extern __constant__ float lr_alpha_final;

extern __constant__ float lr_m;
extern __constant__ float lr_m_exponential_decay_coefficient;
extern __constant__ float lr_m_final;

extern __constant__ float lr_s;
extern __constant__ float lr_s_exponential_decay_coefficient;
extern __constant__ float lr_s_final;

extern __constant__ float lr_q;
extern __constant__ float lr_q_exponential_decay_coefficient;
extern __constant__ float lr_q_final;

extern __constant__ int densification_frequency;
extern __constant__ int densification_start_epoch;
extern __constant__ int densification_end_epoch;
extern __constant__ float alpha_threshold_for_Gauss_removal;
extern __constant__ float min_s_coefficients_clipping_threshold;
extern __constant__ float max_s_coefficients_clipping_threshold;
extern __constant__ float min_s_norm_threshold_for_Gauss_removal;
extern __constant__ float max_s_norm_threshold_for_Gauss_removal;
extern __constant__ float mu_grad_norm_threshold_for_densification;
extern __constant__ float s_norm_threshold_for_split_strategy;
extern __constant__ float split_ratio;
extern __constant__ float lambda;
extern __constant__ float ray_termination_T_threshold;
extern __constant__ float chi_square_squared_radius; 
extern __constant__ int max_Gaussians_per_ray;
extern __constant__ int max_Gaussians_per_model;

// *** *** *** *** ***

struct SbtRecord {
	__align__(OPTIX_SBT_RECORD_ALIGNMENT) char header[OPTIX_SBT_RECORD_HEADER_SIZE];
};

// *** *** *** *** ***

void LoadFromFile(const char *dirPath, int epochNum, const char *fName, void *buf, int size);

__global__ void GenerateInstances(
	float4 *GC_part_1, float4 *GC_part_2, float4 *GC_part_3, float2 *GC_part_4,
	int numberOfGaussians,
	OptixTraversableHandle GAS,
	float *instances
);

__global__ void ComputeAABBs(
	float4 *GC_part_1, float4 *GC_part_2, float4 *GC_part_3, float2 *GC_part_4,
	int numberOfGaussians,
	float *AABBs
);

__global__ void MultiplyPointwiseReal(REAL_G *arr1_in, REAL_G *arr2_in, REAL_G *arr_out, int size);
__global__ void MultiplyPointwiseComplex(COMPLEX_G *arr1_in, COMPLEX_G *arr2_in, COMPLEX_G *arr_out, int size);

__device__ float RandomFloat(unsigned n);
__device__ void RandomNormalFloat(unsigned n, float& Z1, float& Z2);

__device__ void ComputeRotationMatrix(
	float4 q,

	float &R11, float &R12, float &R13,
	float &R21, float &R22, float &R23,
	float &R31, float &R32, float &R33
);

__device__ float3 RandomMultinormalFloat(
	float3 m,
	float3 scale,

	float R11, float R12, float R13,
	float R21, float R22, float R23,
	float R31, float R32, float R33,

	float Z1, float Z2, float Z3
);

// *** *** *** *** ***

template<int SH_degree>
struct SGaussianComponent {
	float mX, mY, mZ;

	float qr, qi, qj, qk;
	float sX, sY, sZ;

	float A11, A12, A13;
	float A21, A22, A23;
	float A31, A32, A33;

	float R, G, B;
	float alpha;

	float RGB_SH_higher_order[(((SH_degree + 1) * (SH_degree + 1)) * 3) - 3];
};

template<>
struct SGaussianComponent<0> {
	float mX, mY, mZ;

	float qr, qi, qj, qk;
	float sX, sY, sZ;

	float A11, A12, A13;
	float A21, A22, A23;
	float A31, A32, A33;

	float R, G, B;
	float alpha;
};
// *** *** *** *** ***

template<int SH_degree>
struct SRenderParams {
	float double_tan_half_fov_x;
	float double_tan_half_fov_y;
	void *bitmap;
	int w; int h;
	SGaussianComponent<SH_degree> *GC;
	int numberOfGaussians;
	unsigned *bitmap_ref;
	int NUMBER_OF_POSES;
};

// *************************************************************************************************

template<int SH_degree>
struct SOptiXRenderParams {
	// RENDERER
	OptixDeviceContext optixContext;
	OptixPipeline pipeline;

	OptixShaderBindingTable *sbt;
	void *raygenRecordsBuffer;
	void *missRecordsBuffer;
	void *hitgroupRecordsBuffer;

	int numberOfGaussians;
	int scatterBufferSize; // !!! !!! !!!
	int tmpArraysGroup1Size;
	int tmpArraysGroup2Size; // !!! !!! !!!

	float3 *Gaussian_as_polygon_vertices;
	int3 *Gaussian_as_polygon_indices;

	void *compactedSizeBuffer;
	void *tempBuffer;
	unsigned long long tempBufferSize;
	void *outputBuffer;
	unsigned long long outputBufferSize;
	OptixTraversableHandle GAS;
	void *GASBuffer;
	unsigned long long GASBufferSize;
	void *instancesBuffer;
	OptixTraversableHandle IAS;
	void *IASBuffer;
	unsigned long long IASBufferSize;

	unsigned *bitmap_out_device;
	unsigned *bitmap_out_host;
	unsigned width;
	unsigned height;

	float3 O;
	float3 R, D, F;
	float double_tan_half_fov_x;
	float double_tan_half_fov_y;

	void *launchParamsBuffer;

	// *** *** *** *** ***

	float4 *GC_part_1_1, *m11, *v11;
	float4 *GC_part_2_1, *m21, *v21;
	float4 *GC_part_3_1, *m31, *v31;
	float2 *GC_part_4_1, *m41, *v41; // !!! !!! !!!

	REAL4_G *dL_dparams_1;
	REAL4_G *dL_dparams_2;
	REAL4_G *dL_dparams_3;
	REAL2_G *dL_dparams_4; // !!! !!! !!!

	// *** *** *** *** ***

	// OPTIMIZER
	unsigned *bitmap_ref;
	double *loss_device;
	double loss_host;

	float *max_RSH;
	float *max_GSH;
	float *max_BSH;

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
};

// *** *** *** *** ***

template<>
struct SOptiXRenderParams<1> {
	// RENDERER
	OptixDeviceContext optixContext;
	OptixPipeline pipeline;

	OptixShaderBindingTable *sbt;
	void *raygenRecordsBuffer;
	void *missRecordsBuffer;
	void *hitgroupRecordsBuffer;

	int numberOfGaussians;
	int scatterBufferSize; // !!! !!! !!!
	int tmpArraysGroup1Size;
	int tmpArraysGroup2Size; // !!! !!! !!!

	float3 *Gaussian_as_polygon_vertices;
	int3 *Gaussian_as_polygon_indices;

	void *compactedSizeBuffer;
	void *tempBuffer;
	unsigned long long tempBufferSize;
	void *outputBuffer;
	unsigned long long outputBufferSize;
	OptixTraversableHandle GAS;
	void *GASBuffer;
	unsigned long long GASBufferSize;
	void *instancesBuffer;
	OptixTraversableHandle IAS;
	void *IASBuffer;
	unsigned long long IASBufferSize;

	unsigned *bitmap_out_device;
	unsigned *bitmap_out_host;
	unsigned width;
	unsigned height;

	float3 O;
	float3 R, D, F;
	float double_tan_half_fov_x;
	float double_tan_half_fov_y;

	void *launchParamsBuffer;

	// *** *** *** *** ***

	float4 *GC_part_1_1, *m11, *v11;
	float4 *GC_part_2_1, *m21, *v21;
	float4 *GC_part_3_1, *m31, *v31;
	float2 *GC_part_4_1, *m41, *v41; // !!! !!! !!!

	REAL4_G *dL_dparams_1;
	REAL4_G *dL_dparams_2;
	REAL4_G *dL_dparams_3;
	REAL2_G *dL_dparams_4; // !!! !!! !!!

	float4 *GC_SH_1, *m_SH_1, *v_SH_1;
	float4 *GC_SH_2, *m_SH_2, *v_SH_2;
	float *GC_SH_3, *m_SH_3, *v_SH_3; // !!! !!! !!!
	
	REAL4_G *dL_dparams_SH_1;
	REAL4_G *dL_dparams_SH_2;
	REAL_G *dL_dparams_SH_3; // !!! !!! !!!

	// *** *** *** *** ***

	// OPTIMIZER
	unsigned *bitmap_ref;
	double *loss_device;
	double loss_host;

	float *max_RSH;
	float *max_GSH;
	float *max_BSH;

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
};

// *** *** *** *** ***

template<>
struct SOptiXRenderParams<2> {
	// RENDERER
	OptixDeviceContext optixContext;
	OptixPipeline pipeline;

	OptixShaderBindingTable *sbt;
	void *raygenRecordsBuffer;
	void *missRecordsBuffer;
	void *hitgroupRecordsBuffer;

	int numberOfGaussians;
	int scatterBufferSize; // !!! !!! !!!
	int tmpArraysGroup1Size;
	int tmpArraysGroup2Size; // !!! !!! !!!

	float3 *Gaussian_as_polygon_vertices;
	int3 *Gaussian_as_polygon_indices;

	void *compactedSizeBuffer;
	void *tempBuffer;
	unsigned long long tempBufferSize;
	void *outputBuffer;
	unsigned long long outputBufferSize;
	OptixTraversableHandle GAS;
	void *GASBuffer;
	unsigned long long GASBufferSize;
	void *instancesBuffer;
	OptixTraversableHandle IAS;
	void *IASBuffer;
	unsigned long long IASBufferSize;

	unsigned *bitmap_out_device;
	unsigned *bitmap_out_host;
	unsigned width;
	unsigned height;

	float3 O;
	float3 R, D, F;
	float double_tan_half_fov_x;
	float double_tan_half_fov_y;

	void *launchParamsBuffer;

	// *** *** *** *** ***

	float4 *GC_part_1_1, *m11, *v11;
	float4 *GC_part_2_1, *m21, *v21;
	float4 *GC_part_3_1, *m31, *v31;
	float2 *GC_part_4_1, *m41, *v41; // !!! !!! !!!

	REAL4_G *dL_dparams_1;
	REAL4_G *dL_dparams_2;
	REAL4_G *dL_dparams_3;
	REAL2_G *dL_dparams_4; // !!! !!! !!!

	float4 *GC_SH_1, *m_SH_1, *v_SH_1;
	float4 *GC_SH_2, *m_SH_2, *v_SH_2;
	float4 *GC_SH_3, *m_SH_3, *v_SH_3;
	float4 *GC_SH_4, *m_SH_4, *v_SH_4;
	float4 *GC_SH_5, *m_SH_5, *v_SH_5;
	float4 *GC_SH_6, *m_SH_6, *v_SH_6;
	
	REAL4_G *dL_dparams_SH_1;
	REAL4_G *dL_dparams_SH_2;
	REAL4_G *dL_dparams_SH_3;
	REAL4_G *dL_dparams_SH_4;
	REAL4_G *dL_dparams_SH_5;
	REAL4_G *dL_dparams_SH_6;

	// *** *** *** *** ***
	
	// OPTIMIZER
	unsigned *bitmap_ref;
	double *loss_device;
	double loss_host;

	float *max_RSH;
	float *max_GSH;
	float *max_BSH;

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
};

// *** *** *** *** ***

template<>
struct SOptiXRenderParams<3> {
	// RENDERER
	OptixDeviceContext optixContext;
	OptixPipeline pipeline;

	OptixShaderBindingTable *sbt;
	void *raygenRecordsBuffer;
	void *missRecordsBuffer;
	void *hitgroupRecordsBuffer;

	int numberOfGaussians;
	int scatterBufferSize; // !!! !!! !!!
	int tmpArraysGroup1Size;
	int tmpArraysGroup2Size; // !!! !!! !!!

	float3 *Gaussian_as_polygon_vertices;
	int3 *Gaussian_as_polygon_indices;

	void *compactedSizeBuffer;
	void *tempBuffer;
	unsigned long long tempBufferSize;
	void *outputBuffer;
	unsigned long long outputBufferSize;
	OptixTraversableHandle GAS;
	void *GASBuffer;
	unsigned long long GASBufferSize;
	void *instancesBuffer;
	OptixTraversableHandle IAS;
	void *IASBuffer;
	unsigned long long IASBufferSize;

	unsigned *bitmap_out_device;
	unsigned *bitmap_out_host;
	unsigned width;
	unsigned height;

	float3 O;
	float3 R, D, F;
	float double_tan_half_fov_x;
	float double_tan_half_fov_y;

	void *launchParamsBuffer;

	// *** *** *** *** ***

	float4 *GC_part_1_1, *m11, *v11;
	float4 *GC_part_2_1, *m21, *v21;
	float4 *GC_part_3_1, *m31, *v31;
	float2 *GC_part_4_1, *m41, *v41; // !!! !!! !!!

	REAL4_G *dL_dparams_1;
	REAL4_G *dL_dparams_2;
	REAL4_G *dL_dparams_3;
	REAL2_G *dL_dparams_4; // !!! !!! !!!

	float4 *GC_SH_1, *m_SH_1, *v_SH_1;
	float4 *GC_SH_2, *m_SH_2, *v_SH_2;
	float4 *GC_SH_3, *m_SH_3, *v_SH_3;
	float4 *GC_SH_4, *m_SH_4, *v_SH_4;
	float4 *GC_SH_5, *m_SH_5, *v_SH_5;
	float4 *GC_SH_6, *m_SH_6, *v_SH_6;
	float4 *GC_SH_7, *m_SH_7, *v_SH_7;
	float4 *GC_SH_8, *m_SH_8, *v_SH_8;
	float4 *GC_SH_9, *m_SH_9, *v_SH_9;
	float4 *GC_SH_10, *m_SH_10, *v_SH_10;
	float4 *GC_SH_11, *m_SH_11, *v_SH_11;
	float *GC_SH_12, *m_SH_12, *v_SH_12; // !!! !!! !!!
	
	REAL4_G *dL_dparams_SH_1;
	REAL4_G *dL_dparams_SH_2;
	REAL4_G *dL_dparams_SH_3;
	REAL4_G *dL_dparams_SH_4;
	REAL4_G *dL_dparams_SH_5;
	REAL4_G *dL_dparams_SH_6;
	REAL4_G *dL_dparams_SH_7;
	REAL4_G *dL_dparams_SH_8;
	REAL4_G *dL_dparams_SH_9;
	REAL4_G *dL_dparams_SH_10;
	REAL4_G *dL_dparams_SH_11;
	REAL_G *dL_dparams_SH_12; // !!! !!! !!!
	
	// *** *** *** *** ***

	// OPTIMIZER
	unsigned *bitmap_ref;
	double *loss_device;
	double loss_host;

	float *max_RSH;
	float *max_GSH;
	float *max_BSH;

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
};

// *** *** *** *** ***

template<>
struct SOptiXRenderParams<4> {
	// RENDERER
	OptixDeviceContext optixContext;
	OptixPipeline pipeline;

	OptixShaderBindingTable *sbt;
	void *raygenRecordsBuffer;
	void *missRecordsBuffer;
	void *hitgroupRecordsBuffer;

	int numberOfGaussians;
	int scatterBufferSize; // !!! !!! !!!
	int tmpArraysGroup1Size;
	int tmpArraysGroup2Size; // !!! !!! !!!

	float3 *Gaussian_as_polygon_vertices;
	int3 *Gaussian_as_polygon_indices;

	void *compactedSizeBuffer;
	void *tempBuffer;
	unsigned long long tempBufferSize;
	void *outputBuffer;
	unsigned long long outputBufferSize;
	OptixTraversableHandle GAS;
	void *GASBuffer;
	unsigned long long GASBufferSize;
	void *instancesBuffer;
	OptixTraversableHandle IAS;
	void *IASBuffer;
	unsigned long long IASBufferSize;

	unsigned *bitmap_out_device;
	unsigned *bitmap_out_host;
	unsigned width;
	unsigned height;

	float3 O;
	float3 R, D, F;
	float double_tan_half_fov_x;
	float double_tan_half_fov_y;

	void *launchParamsBuffer;

	// *** *** *** *** ***

	float4 *GC_part_1_1, *m11, *v11;
	float4 *GC_part_2_1, *m21, *v21;
	float4 *GC_part_3_1, *m31, *v31;
	float2 *GC_part_4_1, *m41, *v41; // !!! !!! !!!

	REAL4_G *dL_dparams_1;
	REAL4_G *dL_dparams_2;
	REAL4_G *dL_dparams_3;
	REAL2_G *dL_dparams_4; // !!! !!! !!!

	float4 *GC_SH_1, *m_SH_1, *v_SH_1;
	float4 *GC_SH_2, *m_SH_2, *v_SH_2;
	float4 *GC_SH_3, *m_SH_3, *v_SH_3;
	float4 *GC_SH_4, *m_SH_4, *v_SH_4;
	float4 *GC_SH_5, *m_SH_5, *v_SH_5;
	float4 *GC_SH_6, *m_SH_6, *v_SH_6;
	float4 *GC_SH_7, *m_SH_7, *v_SH_7;
	float4 *GC_SH_8, *m_SH_8, *v_SH_8;
	float4 *GC_SH_9, *m_SH_9, *v_SH_9;
	float4 *GC_SH_10, *m_SH_10, *v_SH_10;
	float4 *GC_SH_11, *m_SH_11, *v_SH_11;
	float4 *GC_SH_12, *m_SH_12, *v_SH_12;
	float4 *GC_SH_13, *m_SH_13, *v_SH_13;
	float4 *GC_SH_14, *m_SH_14, *v_SH_14;
	float4 *GC_SH_15, *m_SH_15, *v_SH_15;
	float4 *GC_SH_16, *m_SH_16, *v_SH_16;
	float4 *GC_SH_17, *m_SH_17, *v_SH_17;
	float4 *GC_SH_18, *m_SH_18, *v_SH_18;

	REAL4_G *dL_dparams_SH_1;
	REAL4_G *dL_dparams_SH_2;
	REAL4_G *dL_dparams_SH_3;
	REAL4_G *dL_dparams_SH_4;
	REAL4_G *dL_dparams_SH_5;
	REAL4_G *dL_dparams_SH_6;
	REAL4_G *dL_dparams_SH_7;
	REAL4_G *dL_dparams_SH_8;
	REAL4_G *dL_dparams_SH_9;
	REAL4_G *dL_dparams_SH_10;
	REAL4_G *dL_dparams_SH_11;
	REAL4_G *dL_dparams_SH_12;
	REAL4_G *dL_dparams_SH_13;
	REAL4_G *dL_dparams_SH_14;
	REAL4_G *dL_dparams_SH_15;
	REAL4_G *dL_dparams_SH_16;
	REAL4_G *dL_dparams_SH_17;
	REAL4_G *dL_dparams_SH_18;

	// *** *** *** *** ***

	// OPTIMIZER
	unsigned *bitmap_ref;
	double *loss_device;
	double loss_host;

	float *max_RSH;
	float *max_GSH;
	float *max_BSH;

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
};

// *************************************************************************************************

struct SOptiXRenderConfig {
	char *learning_phase;
	char *data_path;
	char *pretrained_model_path;
	char *data_format;

	int start_epoch;
	int end_epoch;

	int SH_degree;

	float bg_color_R;
	float bg_color_G;
	float bg_color_B;

	int number_of_sides;
	float epsilon;

	float lr_SH0;
	float lr_SH0_exponential_decay_coefficient;
	float lr_SH0_final;

	int   SH1_activation_iter;
	float lr_SH1;
	float lr_SH1_exponential_decay_coefficient;
	float lr_SH1_final;

	int   SH2_activation_iter;
	float lr_SH2;
	float lr_SH2_exponential_decay_coefficient;
	float lr_SH2_final;

	int   SH3_activation_iter;
	float lr_SH3;
	float lr_SH3_exponential_decay_coefficient;
	float lr_SH3_final;

	int   SH4_activation_iter;
	float lr_SH4;
	float lr_SH4_exponential_decay_coefficient;
	float lr_SH4_final;

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
	float ray_termination_T_threshold_inference;
	float last_significant_Gauss_alpha_gradient_precision;
	float chi_square_squared_radius;
	int max_Gaussians_per_ray;
	
	int saving_frequency;
	int saving_iter;

	int saving_frequency_PLY;
	int saving_iter_PLY;

	bool evaluation_on_startup_train;
	int evaluation_frequency_train;
	int evaluation_iter_train;
	bool evaluation_on_finish_train;

	bool evaluation_on_startup_test;
	int evaluation_frequency_test;
	int evaluation_iter_test;
	bool evaluation_on_finish_test;

	bool visualization_on_startup_train;
	int visualization_frequency_train;
	int visualization_iter_train;
	bool visualization_on_finish_train;

	bool visualization_on_startup_test;
	int visualization_frequency_test;
	int visualization_iter_test;
	bool visualization_on_finish_test;

	int max_Gaussians_per_model;

	double tmp_arrays_growth_factor;
};

// *************************************************************************************************
// SHADERS                                                                                         *
// *************************************************************************************************

template<int n>
struct LaunchParams {
	unsigned *bitmap;
	unsigned width;
	unsigned height;

	float3 O;
	float3 R, D, F;
	float double_tan_half_fov_x;
	float double_tan_half_fov_y;

	OptixTraversableHandle traversable;

	float4 *GC_part_1;
	float4 *GC_part_2;
	float4 *GC_part_3;
	float2 *GC_part_4;

	int *Gaussians_indices;
	REAL_G *bitmap_out_R, *bitmap_out_G, *bitmap_out_B;

	float ray_termination_T_threshold;
	float chi_square_squared_radius;
	int max_Gaussians_per_ray;

	float bg_color_R;
	float bg_color_G;
	float bg_color_B;

	bool inference;

	float *max_RSH;
	float *max_GSH;
	float *max_BSH;
};

// *** *** *** *** ***

template<>
struct LaunchParams<1> {
	unsigned *bitmap;
	unsigned width;
	unsigned height;

	float3 O;
	float3 R, D, F;
	float double_tan_half_fov_x;
	float double_tan_half_fov_y;

	OptixTraversableHandle traversable;

	float4 *GC_part_1;
	float4 *GC_part_2;
	float4 *GC_part_3;
	float2 *GC_part_4;

	float4 *GC_SH_1;
	float4 *GC_SH_2;
	float *GC_SH_3;

	int *Gaussians_indices;
	REAL_G *bitmap_out_R, *bitmap_out_G, *bitmap_out_B;

	float ray_termination_T_threshold;
	float chi_square_squared_radius;
	int max_Gaussians_per_ray;

	float bg_color_R;
	float bg_color_G;
	float bg_color_B;

	bool inference;

	float *max_RSH;
	float *max_GSH;
	float *max_BSH;
};

// *** *** *** *** ***

template<>
struct LaunchParams<2> {
	unsigned *bitmap;
	unsigned width;
	unsigned height;

	float3 O;
	float3 R, D, F;
	float double_tan_half_fov_x;
	float double_tan_half_fov_y;

	OptixTraversableHandle traversable;

	float4 *GC_part_1;
	float4 *GC_part_2;
	float4 *GC_part_3;
	float2 *GC_part_4;

	float4 *GC_SH_1;
	float4 *GC_SH_2;
	float4 *GC_SH_3;
	float4 *GC_SH_4;
	float4 *GC_SH_5;
	float4 *GC_SH_6;

	int *Gaussians_indices;
	REAL_G *bitmap_out_R, *bitmap_out_G, *bitmap_out_B;

	float ray_termination_T_threshold;
	float chi_square_squared_radius;
	int max_Gaussians_per_ray;

	float bg_color_R;
	float bg_color_G;
	float bg_color_B;

	bool inference;

	float *max_RSH;
	float *max_GSH;
	float *max_BSH;
};

// *** *** *** *** ***

template<>
struct LaunchParams<3> {
	unsigned *bitmap;
	unsigned width;
	unsigned height;

	float3 O;
	float3 R, D, F;
	float double_tan_half_fov_x;
	float double_tan_half_fov_y;

	OptixTraversableHandle traversable;

	float4 *GC_part_1;
	float4 *GC_part_2;
	float4 *GC_part_3;
	float2 *GC_part_4;

	float4 *GC_SH_1;
	float4 *GC_SH_2;
	float4 *GC_SH_3;
	float4 *GC_SH_4;
	float4 *GC_SH_5;
	float4 *GC_SH_6;
	float4 *GC_SH_7;
	float4 *GC_SH_8;
	float4 *GC_SH_9;
	float4 *GC_SH_10;
	float4 *GC_SH_11;
	float *GC_SH_12;

	int *Gaussians_indices;
	REAL_G *bitmap_out_R, *bitmap_out_G, *bitmap_out_B;

	float ray_termination_T_threshold;
	float chi_square_squared_radius;
	int max_Gaussians_per_ray;

	float bg_color_R;
	float bg_color_G;
	float bg_color_B;

	bool inference;

	float *max_RSH;
	float *max_GSH;
	float *max_BSH;
};

// *** *** *** *** ***

template<>
struct LaunchParams<4> {
	unsigned *bitmap;
	unsigned width;
	unsigned height;

	float3 O;
	float3 R, D, F;
	float double_tan_half_fov_x;
	float double_tan_half_fov_y;

	OptixTraversableHandle traversable;

	float4 *GC_part_1;
	float4 *GC_part_2;
	float4 *GC_part_3;
	float2 *GC_part_4;

	float4 *GC_SH_1;
	float4 *GC_SH_2;
	float4 *GC_SH_3;
	float4 *GC_SH_4;
	float4 *GC_SH_5;
	float4 *GC_SH_6;
	float4 *GC_SH_7;
	float4 *GC_SH_8;
	float4 *GC_SH_9;
	float4 *GC_SH_10;
	float4 *GC_SH_11;
	float4 *GC_SH_12;
	float4 *GC_SH_13;
	float4 *GC_SH_14;
	float4 *GC_SH_15;
	float4 *GC_SH_16;
	float4 *GC_SH_17;
	float4 *GC_SH_18;

	int *Gaussians_indices;
	REAL_G *bitmap_out_R, *bitmap_out_G, *bitmap_out_B;

	float ray_termination_T_threshold;
	float chi_square_squared_radius;
	int max_Gaussians_per_ray;

	float bg_color_R;
	float bg_color_G;
	float bg_color_B;

	bool inference;

	float *max_RSH;
	float *max_GSH;
	float *max_BSH;
};