#pragma once

#include <Windows.h>

// *** *** *** *** ***

// !!! !!! !!!
extern unsigned seed_dword;
extern unsigned long long seed_qword;
// !!! !!! !!!

// *** *** *** *** ***

struct SCamera {
	float Ox; float Oy; float Oz;
	float Rx; float Ry; float Rz;
	float Dx; float Dy; float Dz;
	float Fx; float Fy; float Fz;
};

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

float RandomFloat();
unsigned RandomInteger();
double RandomDouble();

// *************************************************************************************************

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

typedef struct {
	float x;
	float y;
	float z;
} float3;

template<int SH_degree>
struct SOptiXRenderParams {
	// RENDERER
	void *optixContext;
	void *pipeline;

	void *sbt;
	void *raygenRecordsBuffer;
	void *missRecordsBuffer;
	void *hitgroupRecordsBuffer;

	int numberOfGaussians;
	int scatterBufferSize; // !!! !!! !!!
	int tmpArraysGroup1Size;
	int tmpArraysGroup2Size; // !!! !!! !!!

	void *Gaussian_as_polygon_vertices;
	void *Gaussian_as_polygon_indices;

	void *compactedSizeBuffer;
	void *tempBuffer;
	unsigned long long tempBufferSize;
	void *outputBuffer;
	unsigned long long outputBufferSize;
	unsigned long long GAS;
	void *GASBuffer;
	unsigned long long GASBufferSize;
	void *instancesBuffer;
	unsigned long long IAS;
	void *IASBuffer;
	unsigned long long IASBufferSize;

	unsigned *bitmap_device;
	unsigned *bitmap_host;
	unsigned width;
	unsigned height;

	float3 O;
	float3 R, D, F;
	float double_tan_half_fov_x;
	float double_tan_half_fov_y;

	void *launchParamsBuffer;

	// *** *** *** *** ***

	void *GC_part_1_1, *m11, *v11;
	void *GC_part_2_1, *m21, *v21;
	void *GC_part_3_1, *m31, *v31;
	void *GC_part_4_1, *m41, *v41; // !!! !!! !!!

	void *dL_dparams_1;
	void *dL_dparams_2;
	void *dL_dparams_3;
	void *dL_dparams_4; // !!! !!! !!!

	// *** *** *** *** ***

	// OPTIMIZER
	unsigned *bitmap_ref;
	double *loss_device;
	double loss_host;

	float *max_RSH;
	float *max_GSH;
	float *max_BSH;

	int *Gaussians_indices;
	
	typedef int cufftHandle;
	typedef void cufftComplex;
	typedef void cufftReal;

	// !!! !!! !!!
	cufftHandle planr2c;
	cufftHandle planc2r;

	cufftComplex *F_1;
	cufftComplex *F_2;

	cufftReal *bitmap_ref_R;
	cufftReal *bitmap_ref_G;
	cufftReal *bitmap_ref_B;
	cufftReal *mu_bitmap_ref_R;
	cufftReal *mu_bitmap_ref_G;
	cufftReal *mu_bitmap_ref_B;
	cufftReal *mu_bitmap_out_bitmap_ref_R;
	cufftReal *mu_bitmap_out_bitmap_ref_G;
	cufftReal *mu_bitmap_out_bitmap_ref_B;
	cufftReal *mu_bitmap_ref_R_square;
	cufftReal *mu_bitmap_ref_G_square;
	cufftReal *mu_bitmap_ref_B_square;

	cufftReal *bitmap_out_R;
	cufftReal *bitmap_out_G;
	cufftReal *bitmap_out_B;
	cufftReal *mu_bitmap_out_R;
	cufftReal *mu_bitmap_out_G;
	cufftReal *mu_bitmap_out_B;
	cufftReal *mu_bitmap_out_R_square;
	cufftReal *mu_bitmap_out_G_square;
	cufftReal *mu_bitmap_out_B_square;
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
	void *optixContext;
	void *pipeline;

	void *sbt;
	void *raygenRecordsBuffer;
	void *missRecordsBuffer;
	void *hitgroupRecordsBuffer;

	int numberOfGaussians;
	int scatterBufferSize; // !!! !!! !!!
	int tmpArraysGroup1Size;
	int tmpArraysGroup2Size; // !!! !!! !!!

	void *Gaussian_as_polygon_vertices;
	void *Gaussian_as_polygon_indices;

	void *compactedSizeBuffer;
	void *tempBuffer;
	unsigned long long tempBufferSize;
	void *outputBuffer;
	unsigned long long outputBufferSize;
	unsigned long long GAS;
	void *GASBuffer;
	unsigned long long GASBufferSize;
	void *instancesBuffer;
	unsigned long long IAS;
	void *IASBuffer;
	unsigned long long IASBufferSize;

	unsigned *bitmap_device;
	unsigned *bitmap_host;
	unsigned width;
	unsigned height;

	float3 O;
	float3 R, D, F;
	float double_tan_half_fov_x;
	float double_tan_half_fov_y;

	void *launchParamsBuffer;

	// *** *** *** *** ***

	void *GC_part_1_1, *m11, *v11;
	void *GC_part_2_1, *m21, *v21;
	void *GC_part_3_1, *m31, *v31;
	void *GC_part_4_1, *m41, *v41; // !!! !!! !!!

	void *dL_dparams_1;
	void *dL_dparams_2;
	void *dL_dparams_3;
	void *dL_dparams_4; // !!! !!! !!!

	void *GC_SH_1, *m_SH_1, *v_SH_1;
	void *GC_SH_2, *m_SH_2, *v_SH_2;
	void *GC_SH_3, *m_SH_3, *v_SH_3; // !!! !!! !!!

	void *dL_dparams_SH_1;
	void *dL_dparams_SH_2;
	void *dL_dparams_SH_3; // !!! !!! !!!

	// *** *** *** *** ***

	// OPTIMIZER
	unsigned *bitmap_ref;
	double *loss_device;
	double loss_host;

	float *max_RSH;
	float *max_GSH;
	float *max_BSH;

	int *Gaussians_indices;

	typedef int cufftHandle;
	typedef void cufftComplex;
	typedef void cufftReal;

	// !!! !!! !!!
	cufftHandle planr2c;
	cufftHandle planc2r;

	cufftComplex *F_1;
	cufftComplex *F_2;

	cufftReal *bitmap_ref_R;
	cufftReal *bitmap_ref_G;
	cufftReal *bitmap_ref_B;
	cufftReal *mu_bitmap_ref_R;
	cufftReal *mu_bitmap_ref_G;
	cufftReal *mu_bitmap_ref_B;
	cufftReal *mu_bitmap_out_bitmap_ref_R;
	cufftReal *mu_bitmap_out_bitmap_ref_G;
	cufftReal *mu_bitmap_out_bitmap_ref_B;
	cufftReal *mu_bitmap_ref_R_square;
	cufftReal *mu_bitmap_ref_G_square;
	cufftReal *mu_bitmap_ref_B_square;

	cufftReal *bitmap_out_R;
	cufftReal *bitmap_out_G;
	cufftReal *bitmap_out_B;
	cufftReal *mu_bitmap_out_R;
	cufftReal *mu_bitmap_out_G;
	cufftReal *mu_bitmap_out_B;
	cufftReal *mu_bitmap_out_R_square;
	cufftReal *mu_bitmap_out_G_square;
	cufftReal *mu_bitmap_out_B_square;
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
	void *optixContext;
	void *pipeline;

	void *sbt;
	void *raygenRecordsBuffer;
	void *missRecordsBuffer;
	void *hitgroupRecordsBuffer;

	int numberOfGaussians;
	int scatterBufferSize; // !!! !!! !!!
	int tmpArraysGroup1Size;
	int tmpArraysGroup2Size; // !!! !!! !!!

	void *Gaussian_as_polygon_vertices;
	void *Gaussian_as_polygon_indices;

	void *compactedSizeBuffer;
	void *tempBuffer;
	unsigned long long tempBufferSize;
	void *outputBuffer;
	unsigned long long outputBufferSize;
	unsigned long long GAS;
	void *GASBuffer;
	unsigned long long GASBufferSize;
	void *instancesBuffer;
	unsigned long long IAS;
	void *IASBuffer;
	unsigned long long IASBufferSize;

	unsigned *bitmap_device;
	unsigned *bitmap_host;
	unsigned width;
	unsigned height;

	float3 O;
	float3 R, D, F;
	float double_tan_half_fov_x;
	float double_tan_half_fov_y;

	void *launchParamsBuffer;

	// *** *** *** *** ***

	void *GC_part_1_1, *m11, *v11;
	void *GC_part_2_1, *m21, *v21;
	void *GC_part_3_1, *m31, *v31;
	void *GC_part_4_1, *m41, *v41; // !!! !!! !!!

	void *dL_dparams_1;
	void *dL_dparams_2;
	void *dL_dparams_3;
	void *dL_dparams_4; // !!! !!! !!!

	void *GC_SH_1, *m_SH_1, *v_SH_1;
	void *GC_SH_2, *m_SH_2, *v_SH_2;
	void *GC_SH_3, *m_SH_3, *v_SH_3;
	void *GC_SH_4, *m_SH_4, *v_SH_4;
	void *GC_SH_5, *m_SH_5, *v_SH_5;
	void *GC_SH_6, *m_SH_6, *v_SH_6;

	void *dL_dparams_SH_1;
	void *dL_dparams_SH_2;
	void *dL_dparams_SH_3;
	void *dL_dparams_SH_4;
	void *dL_dparams_SH_5;
	void *dL_dparams_SH_6;

	// *** *** *** *** ***

	// OPTIMIZER
	unsigned *bitmap_ref;
	double *loss_device;
	double loss_host;

	float *max_RSH;
	float *max_GSH;
	float *max_BSH;

	int *Gaussians_indices;

	typedef int cufftHandle;
	typedef void cufftComplex;
	typedef void cufftReal;

	// !!! !!! !!!
	cufftHandle planr2c;
	cufftHandle planc2r;

	cufftComplex *F_1;
	cufftComplex *F_2;

	cufftReal *bitmap_ref_R;
	cufftReal *bitmap_ref_G;
	cufftReal *bitmap_ref_B;
	cufftReal *mu_bitmap_ref_R;
	cufftReal *mu_bitmap_ref_G;
	cufftReal *mu_bitmap_ref_B;
	cufftReal *mu_bitmap_out_bitmap_ref_R;
	cufftReal *mu_bitmap_out_bitmap_ref_G;
	cufftReal *mu_bitmap_out_bitmap_ref_B;
	cufftReal *mu_bitmap_ref_R_square;
	cufftReal *mu_bitmap_ref_G_square;
	cufftReal *mu_bitmap_ref_B_square;

	cufftReal *bitmap_out_R;
	cufftReal *bitmap_out_G;
	cufftReal *bitmap_out_B;
	cufftReal *mu_bitmap_out_R;
	cufftReal *mu_bitmap_out_G;
	cufftReal *mu_bitmap_out_B;
	cufftReal *mu_bitmap_out_R_square;
	cufftReal *mu_bitmap_out_G_square;
	cufftReal *mu_bitmap_out_B_square;
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
	void *optixContext;
	void *pipeline;

	void *sbt;
	void *raygenRecordsBuffer;
	void *missRecordsBuffer;
	void *hitgroupRecordsBuffer;

	int numberOfGaussians;
	int scatterBufferSize; // !!! !!! !!!
	int tmpArraysGroup1Size;
	int tmpArraysGroup2Size; // !!! !!! !!!

	void *Gaussian_as_polygon_vertices;
	void *Gaussian_as_polygon_indices;

	void *compactedSizeBuffer;
	void *tempBuffer;
	unsigned long long tempBufferSize;
	void *outputBuffer;
	unsigned long long outputBufferSize;
	unsigned long long GAS;
	void *GASBuffer;
	unsigned long long GASBufferSize;
	void *instancesBuffer;
	unsigned long long IAS;
	void *IASBuffer;
	unsigned long long IASBufferSize;

	unsigned *bitmap_device;
	unsigned *bitmap_host;
	unsigned width;
	unsigned height;

	float3 O;
	float3 R, D, F;
	float double_tan_half_fov_x;
	float double_tan_half_fov_y;

	void *launchParamsBuffer;

	// *** *** *** *** ***

	void *GC_part_1_1, *m11, *v11;
	void *GC_part_2_1, *m21, *v21;
	void *GC_part_3_1, *m31, *v31;
	void *GC_part_4_1, *m41, *v41; // !!! !!! !!!

	void *dL_dparams_1;
	void *dL_dparams_2;
	void *dL_dparams_3;
	void *dL_dparams_4; // !!! !!! !!!

	void *GC_SH_1, *m_SH_1, *v_SH_1;
	void *GC_SH_2, *m_SH_2, *v_SH_2;
	void *GC_SH_3, *m_SH_3, *v_SH_3;
	void *GC_SH_4, *m_SH_4, *v_SH_4;
	void *GC_SH_5, *m_SH_5, *v_SH_5;
	void *GC_SH_6, *m_SH_6, *v_SH_6;
	void *GC_SH_7, *m_SH_7, *v_SH_7;
	void *GC_SH_8, *m_SH_8, *v_SH_8;
	void *GC_SH_9, *m_SH_9, *v_SH_9;
	void *GC_SH_10, *m_SH_10, *v_SH_10;
	void *GC_SH_11, *m_SH_11, *v_SH_11;
	void *GC_SH_12, *m_SH_12, *v_SH_12; // !!! !!! !!!

	void *dL_dparams_SH_1;
	void *dL_dparams_SH_2;
	void *dL_dparams_SH_3;
	void *dL_dparams_SH_4;
	void *dL_dparams_SH_5;
	void *dL_dparams_SH_6;
	void *dL_dparams_SH_7;
	void *dL_dparams_SH_8;
	void *dL_dparams_SH_9;
	void *dL_dparams_SH_10;
	void *dL_dparams_SH_11;
	void *dL_dparams_SH_12; // !!! !!! !!!

	// *** *** *** *** ***

	// OPTIMIZER
	unsigned *bitmap_ref;
	double *loss_device;
	double loss_host;

	float *max_RSH;
	float *max_GSH;
	float *max_BSH;

	int *Gaussians_indices;

	typedef int cufftHandle;
	typedef void cufftComplex;
	typedef void cufftReal;

	// !!! !!! !!!
	cufftHandle planr2c;
	cufftHandle planc2r;

	cufftComplex *F_1;
	cufftComplex *F_2;

	cufftReal *bitmap_ref_R;
	cufftReal *bitmap_ref_G;
	cufftReal *bitmap_ref_B;
	cufftReal *mu_bitmap_ref_R;
	cufftReal *mu_bitmap_ref_G;
	cufftReal *mu_bitmap_ref_B;
	cufftReal *mu_bitmap_out_bitmap_ref_R;
	cufftReal *mu_bitmap_out_bitmap_ref_G;
	cufftReal *mu_bitmap_out_bitmap_ref_B;
	cufftReal *mu_bitmap_ref_R_square;
	cufftReal *mu_bitmap_ref_G_square;
	cufftReal *mu_bitmap_ref_B_square;

	cufftReal *bitmap_out_R;
	cufftReal *bitmap_out_G;
	cufftReal *bitmap_out_B;
	cufftReal *mu_bitmap_out_R;
	cufftReal *mu_bitmap_out_G;
	cufftReal *mu_bitmap_out_B;
	cufftReal *mu_bitmap_out_R_square;
	cufftReal *mu_bitmap_out_G_square;
	cufftReal *mu_bitmap_out_B_square;
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
	void *optixContext;
	void *pipeline;

	void *sbt;
	void *raygenRecordsBuffer;
	void *missRecordsBuffer;
	void *hitgroupRecordsBuffer;

	int numberOfGaussians;
	int scatterBufferSize; // !!! !!! !!!
	int tmpArraysGroup1Size;
	int tmpArraysGroup2Size; // !!! !!! !!!

	void *Gaussian_as_polygon_vertices;
	void *Gaussian_as_polygon_indices;

	void *compactedSizeBuffer;
	void *tempBuffer;
	unsigned long long tempBufferSize;
	void *outputBuffer;
	unsigned long long outputBufferSize;
	unsigned long long GAS;
	void *GASBuffer;
	unsigned long long GASBufferSize;
	void *instancesBuffer;
	unsigned long long IAS;
	void *IASBuffer;
	unsigned long long IASBufferSize;

	unsigned *bitmap_device;
	unsigned *bitmap_host;
	unsigned width;
	unsigned height;

	float3 O;
	float3 R, D, F;
	float double_tan_half_fov_x;
	float double_tan_half_fov_y;

	void *launchParamsBuffer;

	// *** *** *** *** ***

	void *GC_part_1_1, *m11, *v11;
	void *GC_part_2_1, *m21, *v21;
	void *GC_part_3_1, *m31, *v31;
	void *GC_part_4_1, *m41, *v41; // !!! !!! !!!

	void *dL_dparams_1;
	void *dL_dparams_2;
	void *dL_dparams_3;
	void *dL_dparams_4; // !!! !!! !!!

	void *GC_SH_1, *m_SH_1, *v_SH_1;
	void *GC_SH_2, *m_SH_2, *v_SH_2;
	void *GC_SH_3, *m_SH_3, *v_SH_3;
	void *GC_SH_4, *m_SH_4, *v_SH_4;
	void *GC_SH_5, *m_SH_5, *v_SH_5;
	void *GC_SH_6, *m_SH_6, *v_SH_6;
	void *GC_SH_7, *m_SH_7, *v_SH_7;
	void *GC_SH_8, *m_SH_8, *v_SH_8;
	void *GC_SH_9, *m_SH_9, *v_SH_9;
	void *GC_SH_10, *m_SH_10, *v_SH_10;
	void *GC_SH_11, *m_SH_11, *v_SH_11;
	void *GC_SH_12, *m_SH_12, *v_SH_12;
	void *GC_SH_13, *m_SH_13, *v_SH_13;
	void *GC_SH_14, *m_SH_14, *v_SH_14;
	void *GC_SH_15, *m_SH_15, *v_SH_15;
	void *GC_SH_16, *m_SH_16, *v_SH_16;
	void *GC_SH_17, *m_SH_17, *v_SH_17;
	void *GC_SH_18, *m_SH_18, *v_SH_18;

	void *dL_dparams_SH_1;
	void *dL_dparams_SH_2;
	void *dL_dparams_SH_3;
	void *dL_dparams_SH_4;
	void *dL_dparams_SH_5;
	void *dL_dparams_SH_6;
	void *dL_dparams_SH_7;
	void *dL_dparams_SH_8;
	void *dL_dparams_SH_9;
	void *dL_dparams_SH_10;
	void *dL_dparams_SH_11;
	void *dL_dparams_SH_12;
	void *dL_dparams_SH_13;
	void *dL_dparams_SH_14;
	void *dL_dparams_SH_15;
	void *dL_dparams_SH_16;
	void *dL_dparams_SH_17;
	void *dL_dparams_SH_18;

	// *** *** *** *** ***

	// OPTIMIZER
	unsigned *bitmap_ref;
	double *loss_device;
	double loss_host;

	float *max_RSH;
	float *max_GSH;
	float *max_BSH;

	int *Gaussians_indices;

	typedef int cufftHandle;
	typedef void cufftComplex;
	typedef void cufftReal;

	// !!! !!! !!!
	cufftHandle planr2c;
	cufftHandle planc2r;

	cufftComplex *F_1;
	cufftComplex *F_2;

	cufftReal *bitmap_ref_R;
	cufftReal *bitmap_ref_G;
	cufftReal *bitmap_ref_B;
	cufftReal *mu_bitmap_ref_R;
	cufftReal *mu_bitmap_ref_G;
	cufftReal *mu_bitmap_ref_B;
	cufftReal *mu_bitmap_out_bitmap_ref_R;
	cufftReal *mu_bitmap_out_bitmap_ref_G;
	cufftReal *mu_bitmap_out_bitmap_ref_B;
	cufftReal *mu_bitmap_ref_R_square;
	cufftReal *mu_bitmap_ref_G_square;
	cufftReal *mu_bitmap_ref_B_square;

	cufftReal *bitmap_out_R;
	cufftReal *bitmap_out_G;
	cufftReal *bitmap_out_B;
	cufftReal *mu_bitmap_out_R;
	cufftReal *mu_bitmap_out_G;
	cufftReal *mu_bitmap_out_B;
	cufftReal *mu_bitmap_out_R_square;
	cufftReal *mu_bitmap_out_G_square;
	cufftReal *mu_bitmap_out_B_square;
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

extern bool SetConfigurationOptiX(SOptiXRenderConfig &config_OptiX);
extern bool GetSceneExtentOptiX(float &scene_extent_host);

// *************************************************************************************************

extern bool InitializeOptiXRendererSH0(
	SRenderParams<0> &params,
	SOptiXRenderParams<0> &params_OptiX,
	char *dirPath = NULL
);
extern bool InitializeOptiXRendererSH1(
	SRenderParams<1> &params,
	SOptiXRenderParams<1> &params_OptiX,
	char *dirPath = NULL
);
extern bool InitializeOptiXRendererSH2(
	SRenderParams<2> &params,
	SOptiXRenderParams<2> &params_OptiX,
	char *dirPath = NULL
);
extern bool InitializeOptiXRendererSH3(
	SRenderParams<3> &params,
	SOptiXRenderParams<3> &params_OptiX,
	char *dirPath = NULL
);
extern bool InitializeOptiXRendererSH4(
	SRenderParams<4> &params,
	SOptiXRenderParams<4> &params_OptiX,
	char *dirPath = NULL
);

// *************************************************************************************************

extern bool InitializeOptiXOptimizerSH0(
	SRenderParams<0> &params,
	SOptiXRenderParams<0> &params_OptiX,
	char *dirPath = NULL
);
extern bool InitializeOptiXOptimizerSH1(
	SRenderParams<1> &params,
	SOptiXRenderParams<1> &params_OptiX,
	char *dirPath = NULL
);
extern bool InitializeOptiXOptimizerSH2(
	SRenderParams<2> &params,
	SOptiXRenderParams<2> &params_OptiX,
	char *dirPath = NULL
);
extern bool InitializeOptiXOptimizerSH3(
	SRenderParams<3> &params,
	SOptiXRenderParams<3> &params_OptiX,
	char *dirPath = NULL
);
extern bool InitializeOptiXOptimizerSH4(
	SRenderParams<4> &params,
	SOptiXRenderParams<4> &params_OptiX,
	char *dirPath = NULL
);

// *************************************************************************************************

extern bool ZeroGradientOptiXSH0(SOptiXRenderParams<0> &params_OptiX);
extern bool ZeroGradientOptiXSH1(SOptiXRenderParams<1> &params_OptiX);
extern bool ZeroGradientOptiXSH2(SOptiXRenderParams<2> &params_OptiX);
extern bool ZeroGradientOptiXSH3(SOptiXRenderParams<3> &params_OptiX);
extern bool ZeroGradientOptiXSH4(SOptiXRenderParams<4> &params_OptiX);

// *************************************************************************************************

extern bool RenderOptiXSH0(SOptiXRenderParams<0>& params_OptiX, bool inference = true);
extern bool RenderOptiXSH1(SOptiXRenderParams<1>& params_OptiX, bool inference = true);
extern bool RenderOptiXSH2(SOptiXRenderParams<2>& params_OptiX, bool inference = true);
extern bool RenderOptiXSH3(SOptiXRenderParams<3>& params_OptiX, bool inference = true);
extern bool RenderOptiXSH4(SOptiXRenderParams<4>& params_OptiX, bool inference = true);

// *************************************************************************************************

extern bool UpdateGradientOptiXSH0(SOptiXRenderParams<0>& params_OptiX);
extern bool UpdateGradientOptiXSH1(SOptiXRenderParams<1>& params_OptiX);
extern bool UpdateGradientOptiXSH2(SOptiXRenderParams<2>& params_OptiX);
extern bool UpdateGradientOptiXSH3(SOptiXRenderParams<3>& params_OptiX);
extern bool UpdateGradientOptiXSH4(SOptiXRenderParams<4>& params_OptiX);

// *************************************************************************************************

extern bool DumpParametersOptiXSH0(SOptiXRenderParams<0> &params_OptiX, char *dirPath);
extern bool DumpParametersOptiXSH1(SOptiXRenderParams<1> &params_OptiX, char *dirPath);
extern bool DumpParametersOptiXSH2(SOptiXRenderParams<2> &params_OptiX, char *dirPath);
extern bool DumpParametersOptiXSH3(SOptiXRenderParams<3> &params_OptiX, char *dirPath);
extern bool DumpParametersOptiXSH4(SOptiXRenderParams<4> &params_OptiX, char *dirPath);

// *************************************************************************************************

extern bool DumpParametersToPLYFileOptiXSH0(SOptiXRenderParams<0> &params_OptiX, char *dirPath);
extern bool DumpParametersToPLYFileOptiXSH1(SOptiXRenderParams<1> &params_OptiX, char *dirPath);
extern bool DumpParametersToPLYFileOptiXSH2(SOptiXRenderParams<2> &params_OptiX, char *dirPath);
extern bool DumpParametersToPLYFileOptiXSH3(SOptiXRenderParams<3> &params_OptiX, char *dirPath);
extern bool DumpParametersToPLYFileOptiXSH4(SOptiXRenderParams<4> &params_OptiX, char *dirPath);