#include "Header.cuh"

// *************************************************************************************************

template<int SH_degree>
struct S_GC_SH {
};

// *** *** *** *** ***

template<>
struct S_GC_SH<1> {
	float4 *GC_SH_1;
	float4 *GC_SH_2;
	float  *GC_SH_3;
};

// *** *** *** *** ***

template<>
struct S_GC_SH<2> {
	float4 *GC_SH_1;
	float4 *GC_SH_2;
	float4 *GC_SH_3;
	float4 *GC_SH_4;
	float4 *GC_SH_5;
	float4 *GC_SH_6;
};

// *** *** *** *** ***

template<>
struct S_GC_SH<3> {
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
	float  *GC_SH_12;
};

// *** *** *** *** ***

template<>
struct S_GC_SH<4> {
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
};

// *************************************************************************************************

template<int SH_degree>
struct SPLYFileStruct {
	float x;
	float y;
	float z;
	float nx;
	float ny;
	float nz;
	float f_dc_0;
	float f_dc_1;
	float f_dc_2;
	float f_rest_array[(((SH_degree + 1) * (SH_degree + 1)) * 3) - 3];
	float opacity;
	float scale_0;
	float scale_1;
	float scale_2;
	float rot_0;
	float rot_1;
	float rot_2;
	float rot_3;
};

// *** *** *** *** ***

template<>
struct SPLYFileStruct<0> {
	float x;
	float y;
	float z;
	float nx;
	float ny;
	float nz;
	float f_dc_0;
	float f_dc_1;
	float f_dc_2;
	float opacity;
	float scale_0;
	float scale_1;
	float scale_2;
	float rot_0;
	float rot_1;
	float rot_2;
	float rot_3;
};

// *************************************************************************************************

template<int SH_degree>
bool DumpParametersToPLYFileOptiX(SOptiXRenderParams<SH_degree>& params_OptiX, char *dirPath) {
	constexpr int numberOfPropertiesSH = (((SH_degree + 1) * (SH_degree + 1)) * 3) - 3;

	// *** *** *** *** ***

	cudaError_t error_CUDA;

	float4 *GC1 = (float4 *)malloc(sizeof(float4) * params_OptiX.numberOfGaussians);
	if (GC1 == NULL) goto Error;

	float4 *GC2 = (float4 *)malloc(sizeof(float4) * params_OptiX.numberOfGaussians);
	if (GC2 == NULL) goto Error;

	float4 *GC3 = (float4 *)malloc(sizeof(float4) * params_OptiX.numberOfGaussians);
	if (GC3 == NULL) goto Error;

	float2 *GC4 = (float2 *)malloc(sizeof(float2) * params_OptiX.numberOfGaussians);
	if (GC4 == NULL) goto Error;

	// *** *** *** *** ***

	S_GC_SH<SH_degree> GC_SH;

	// Spherical harmonics
	if constexpr (SH_degree >= 1) {
		GC_SH.GC_SH_1 = (float4 *)malloc(sizeof(float4) * params_OptiX.numberOfGaussians);
		if (GC_SH.GC_SH_1 == NULL) goto Error;

		GC_SH.GC_SH_2 = (float4 *)malloc(sizeof(float4) * params_OptiX.numberOfGaussians);
		if (GC_SH.GC_SH_2 == NULL) goto Error;

		if constexpr (SH_degree >= 2) {
			GC_SH.GC_SH_3 = (float4 *)malloc(sizeof(float4) * params_OptiX.numberOfGaussians);
			if (GC_SH.GC_SH_3 == NULL) goto Error;

			GC_SH.GC_SH_4 = (float4 *)malloc(sizeof(float4) * params_OptiX.numberOfGaussians);
			if (GC_SH.GC_SH_4 == NULL) goto Error;

			GC_SH.GC_SH_5 = (float4 *)malloc(sizeof(float4) * params_OptiX.numberOfGaussians);
			if (GC_SH.GC_SH_5 == NULL) goto Error;

			GC_SH.GC_SH_6 = (float4 *)malloc(sizeof(float4) * params_OptiX.numberOfGaussians);
			if (GC_SH.GC_SH_6 == NULL) goto Error;

			if constexpr (SH_degree >= 3) {
				GC_SH.GC_SH_7 = (float4 *)malloc(sizeof(float4) * params_OptiX.numberOfGaussians);
				if (GC_SH.GC_SH_7 == NULL) goto Error;

				GC_SH.GC_SH_8 = (float4 *)malloc(sizeof(float4) * params_OptiX.numberOfGaussians);
				if (GC_SH.GC_SH_8 == NULL) goto Error;

				GC_SH.GC_SH_9 = (float4 *)malloc(sizeof(float4) * params_OptiX.numberOfGaussians);
				if (GC_SH.GC_SH_9 == NULL) goto Error;

				GC_SH.GC_SH_10 = (float4 *)malloc(sizeof(float4) * params_OptiX.numberOfGaussians);
				if (GC_SH.GC_SH_10 == NULL) goto Error;

				GC_SH.GC_SH_11 = (float4 *)malloc(sizeof(float4) * params_OptiX.numberOfGaussians);
				if (GC_SH.GC_SH_11 == NULL) goto Error;

				if constexpr (SH_degree >= 4) {
					GC_SH.GC_SH_12 = (float4 *)malloc(sizeof(float4) * params_OptiX.numberOfGaussians);
					if (GC_SH.GC_SH_12 == NULL) goto Error;

					GC_SH.GC_SH_13 = (float4 *)malloc(sizeof(float4) * params_OptiX.numberOfGaussians);
					if (GC_SH.GC_SH_13 == NULL) goto Error;

					GC_SH.GC_SH_14 = (float4 *)malloc(sizeof(float4) * params_OptiX.numberOfGaussians);
					if (GC_SH.GC_SH_14 == NULL) goto Error;

					GC_SH.GC_SH_15 = (float4 *)malloc(sizeof(float4) * params_OptiX.numberOfGaussians);
					if (GC_SH.GC_SH_15 == NULL) goto Error;

					GC_SH.GC_SH_16 = (float4 *)malloc(sizeof(float4) * params_OptiX.numberOfGaussians);
					if (GC_SH.GC_SH_16 == NULL) goto Error;

					GC_SH.GC_SH_17 = (float4 *)malloc(sizeof(float4) * params_OptiX.numberOfGaussians);
					if (GC_SH.GC_SH_17 == NULL) goto Error;

					GC_SH.GC_SH_18 = (float4 *)malloc(sizeof(float4) * params_OptiX.numberOfGaussians);
					if (GC_SH.GC_SH_18 == NULL) goto Error;
				} else {
					// !!! !!! !!!
					GC_SH.GC_SH_12 = (float *)malloc(sizeof(float) * params_OptiX.numberOfGaussians);
					if (GC_SH.GC_SH_12 == NULL) goto Error;
				}
			}
		} else {
			// !!! !!! !!!
			GC_SH.GC_SH_3 = (float *)malloc(sizeof(float) * params_OptiX.numberOfGaussians);
			if (GC_SH.GC_SH_3 == NULL) goto Error;
		}
	}

	// *** *** *** *** ***

	SPLYFileStruct<SH_degree> *GC = (SPLYFileStruct<SH_degree> *)malloc(sizeof(SPLYFileStruct<SH_degree>) * params_OptiX.numberOfGaussians);
	if (GC == NULL) goto Error;

	// *** *** *** *** ***

	error_CUDA = cudaMemcpy(GC1, params_OptiX.GC_part_1_1, sizeof(float4) * params_OptiX.numberOfGaussians, cudaMemcpyDeviceToHost);
	if (error_CUDA != cudaSuccess) goto Error;

	error_CUDA = cudaMemcpy(GC2, params_OptiX.GC_part_2_1, sizeof(float4) * params_OptiX.numberOfGaussians, cudaMemcpyDeviceToHost);
	if (error_CUDA != cudaSuccess) goto Error;

	error_CUDA = cudaMemcpy(GC3, params_OptiX.GC_part_3_1, sizeof(float4) * params_OptiX.numberOfGaussians, cudaMemcpyDeviceToHost);
	if (error_CUDA != cudaSuccess) goto Error;

	error_CUDA = cudaMemcpy(GC4, params_OptiX.GC_part_4_1, sizeof(float2) * params_OptiX.numberOfGaussians, cudaMemcpyDeviceToHost);
	if (error_CUDA != cudaSuccess) goto Error;

	// *** *** *** *** ***

	// Spherical harmonics
	if constexpr (SH_degree >= 1) {
		error_CUDA = cudaMemcpy(GC_SH.GC_SH_1, params_OptiX.GC_SH_1, sizeof(float4) * params_OptiX.numberOfGaussians, cudaMemcpyDeviceToHost);
		if (error_CUDA != cudaSuccess) goto Error;

		error_CUDA = cudaMemcpy(GC_SH.GC_SH_2, params_OptiX.GC_SH_2, sizeof(float4) * params_OptiX.numberOfGaussians, cudaMemcpyDeviceToHost);
		if (error_CUDA != cudaSuccess) goto Error;

		if constexpr (SH_degree >= 2) {
			error_CUDA = cudaMemcpy(GC_SH.GC_SH_3, params_OptiX.GC_SH_3, sizeof(float4) * params_OptiX.numberOfGaussians, cudaMemcpyDeviceToHost);
			if (error_CUDA != cudaSuccess) goto Error;

			error_CUDA = cudaMemcpy(GC_SH.GC_SH_4, params_OptiX.GC_SH_4, sizeof(float4) * params_OptiX.numberOfGaussians, cudaMemcpyDeviceToHost);
			if (error_CUDA != cudaSuccess) goto Error;

			error_CUDA = cudaMemcpy(GC_SH.GC_SH_5, params_OptiX.GC_SH_5, sizeof(float4) * params_OptiX.numberOfGaussians, cudaMemcpyDeviceToHost);
			if (error_CUDA != cudaSuccess) goto Error;

			error_CUDA = cudaMemcpy(GC_SH.GC_SH_6, params_OptiX.GC_SH_6, sizeof(float4) * params_OptiX.numberOfGaussians, cudaMemcpyDeviceToHost);
			if (error_CUDA != cudaSuccess) goto Error;

			if constexpr (SH_degree >= 3) {
				error_CUDA = cudaMemcpy(GC_SH.GC_SH_7, params_OptiX.GC_SH_7, sizeof(float4) * params_OptiX.numberOfGaussians, cudaMemcpyDeviceToHost);
				if (error_CUDA != cudaSuccess) goto Error;

				error_CUDA = cudaMemcpy(GC_SH.GC_SH_8, params_OptiX.GC_SH_8, sizeof(float4) * params_OptiX.numberOfGaussians, cudaMemcpyDeviceToHost);
				if (error_CUDA != cudaSuccess) goto Error;

				error_CUDA = cudaMemcpy(GC_SH.GC_SH_9, params_OptiX.GC_SH_9, sizeof(float4) * params_OptiX.numberOfGaussians, cudaMemcpyDeviceToHost);
				if (error_CUDA != cudaSuccess) goto Error;

				error_CUDA = cudaMemcpy(GC_SH.GC_SH_10, params_OptiX.GC_SH_10, sizeof(float4) * params_OptiX.numberOfGaussians, cudaMemcpyDeviceToHost);
				if (error_CUDA != cudaSuccess) goto Error;

				error_CUDA = cudaMemcpy(GC_SH.GC_SH_11, params_OptiX.GC_SH_11, sizeof(float4) * params_OptiX.numberOfGaussians, cudaMemcpyDeviceToHost);
				if (error_CUDA != cudaSuccess) goto Error;

				if constexpr (SH_degree >= 4) {
					error_CUDA = cudaMemcpy(GC_SH.GC_SH_12, params_OptiX.GC_SH_12, sizeof(float4) * params_OptiX.numberOfGaussians, cudaMemcpyDeviceToHost);
					if (error_CUDA != cudaSuccess) goto Error;

					error_CUDA = cudaMemcpy(GC_SH.GC_SH_13, params_OptiX.GC_SH_13, sizeof(float4) * params_OptiX.numberOfGaussians, cudaMemcpyDeviceToHost);
					if (error_CUDA != cudaSuccess) goto Error;

					error_CUDA = cudaMemcpy(GC_SH.GC_SH_14, params_OptiX.GC_SH_14, sizeof(float4) * params_OptiX.numberOfGaussians, cudaMemcpyDeviceToHost);
					if (error_CUDA != cudaSuccess) goto Error;

					error_CUDA = cudaMemcpy(GC_SH.GC_SH_15, params_OptiX.GC_SH_15, sizeof(float4) * params_OptiX.numberOfGaussians, cudaMemcpyDeviceToHost);
					if (error_CUDA != cudaSuccess) goto Error;

					error_CUDA = cudaMemcpy(GC_SH.GC_SH_16, params_OptiX.GC_SH_16, sizeof(float4) * params_OptiX.numberOfGaussians, cudaMemcpyDeviceToHost);
					if (error_CUDA != cudaSuccess) goto Error;

					error_CUDA = cudaMemcpy(GC_SH.GC_SH_17, params_OptiX.GC_SH_17, sizeof(float4) * params_OptiX.numberOfGaussians, cudaMemcpyDeviceToHost);
					if (error_CUDA != cudaSuccess) goto Error;

					error_CUDA = cudaMemcpy(GC_SH.GC_SH_18, params_OptiX.GC_SH_18, sizeof(float4) * params_OptiX.numberOfGaussians, cudaMemcpyDeviceToHost);
					if (error_CUDA != cudaSuccess) goto Error;
				} else {
					// !!! !!! !!!
					error_CUDA = cudaMemcpy(GC_SH.GC_SH_12, params_OptiX.GC_SH_12, sizeof(float) * params_OptiX.numberOfGaussians, cudaMemcpyDeviceToHost);
					if (error_CUDA != cudaSuccess) goto Error;
				}
			}
		} else {
			// !!! !!! !!!
			error_CUDA = cudaMemcpy(GC_SH.GC_SH_3, params_OptiX.GC_SH_3, sizeof(float) * params_OptiX.numberOfGaussians, cudaMemcpyDeviceToHost);
			if (error_CUDA != cudaSuccess) goto Error;
		}
	}

	// *** *** *** *** ***

	FILE *f;
	char fPath[256];

	sprintf_s(fPath, "%s\\iter_%d.ply", dirPath, params_OptiX.epoch);
	fopen_s(&f, fPath, "wb");

	fprintf(f, "ply\n");
	fprintf(f, "format binary_little_endian 1.0\n");
	fprintf(f, "element vertex %d\n", params_OptiX.numberOfGaussians);
	fprintf(f, "property float x\n");
	fprintf(f, "property float y\n");
	fprintf(f, "property float z\n");
	fprintf(f, "property float nx\n");
	fprintf(f, "property float ny\n");
	fprintf(f, "property float nz\n");
	fprintf(f, "property float f_dc_0\n");
	fprintf(f, "property float f_dc_1\n");
	fprintf(f, "property float f_dc_2\n");

	for (int i = 0; i < numberOfPropertiesSH; ++i)
		fprintf(f, "property float f_rest_%d\n", i);

	fprintf(f, "property float opacity\n");
	fprintf(f, "property float scale_0\n");
	fprintf(f, "property float scale_1\n");
	fprintf(f, "property float scale_2\n");
	fprintf(f, "property float rot_0\n");
	fprintf(f, "property float rot_1\n");
	fprintf(f, "property float rot_2\n");
	fprintf(f, "property float rot_3\n");
	fprintf(f, "end_header\n");

	fclose(f);

	// *** *** *** *** ***

	for (int i = 0; i < params_OptiX.numberOfGaussians; ++i) {
		GC[i].x = GC2[i].x;
		GC[i].y = GC2[i].y;
		GC[i].z = GC2[i].z;

		// *** *** *** *** ***

		GC[i].nx = 0.0f;
		GC[i].ny = 0.0f;
		GC[i].nz = 0.0f;

		// *** *** *** *** ***

		GC[i].f_dc_0 = GC1[i].x;
		GC[i].f_dc_1 = GC1[i].y;
		GC[i].f_dc_2 = GC1[i].z;

		// *** *** *** *** ***

		int ind = 0;

		// *** *** *** *** ***

		// Spherical harmonics: R channel
		if constexpr (SH_degree >= 1) {
			GC[i].f_rest_array[ind++] = GC_SH.GC_SH_1[i].x; // R_1_(-1)
			GC[i].f_rest_array[ind++] = GC_SH.GC_SH_1[i].w; // R_1_0
			GC[i].f_rest_array[ind++] = GC_SH.GC_SH_2[i].z; // R_1_1

			if constexpr (SH_degree >= 2) {
				GC[i].f_rest_array[ind++] = GC_SH.GC_SH_3[i].y; // R_2_(-2)
				GC[i].f_rest_array[ind++] = GC_SH.GC_SH_4[i].x; // R_2_(-1)
				GC[i].f_rest_array[ind++] = GC_SH.GC_SH_4[i].w; // R_2_0
				GC[i].f_rest_array[ind++] = GC_SH.GC_SH_5[i].z; // R_2_1
				GC[i].f_rest_array[ind++] = GC_SH.GC_SH_6[i].y; // R_2_2

				if constexpr (SH_degree >= 3) {
					GC[i].f_rest_array[ind++] = GC_SH.GC_SH_7[i].x; // R_3_(-3)
					GC[i].f_rest_array[ind++] = GC_SH.GC_SH_7[i].w; // R_3_(-2)
					GC[i].f_rest_array[ind++] = GC_SH.GC_SH_8[i].z; // R_3_(-1)
					GC[i].f_rest_array[ind++] = GC_SH.GC_SH_9[i].y; // R_3_0
					GC[i].f_rest_array[ind++] = GC_SH.GC_SH_10[i].x; // R_3_1
					GC[i].f_rest_array[ind++] = GC_SH.GC_SH_10[i].w; // R_3_2
					GC[i].f_rest_array[ind++] = GC_SH.GC_SH_11[i].z; // R_3_3

					if constexpr (SH_degree >= 4) {
						GC[i].f_rest_array[ind++] = GC_SH.GC_SH_12[i].y; // R_4_(-4)
						GC[i].f_rest_array[ind++] = GC_SH.GC_SH_13[i].x; // R_4_(-3)
						GC[i].f_rest_array[ind++] = GC_SH.GC_SH_13[i].w; // R_4_(-2)
						GC[i].f_rest_array[ind++] = GC_SH.GC_SH_14[i].z; // R_4_(-1)
						GC[i].f_rest_array[ind++] = GC_SH.GC_SH_15[i].y; // R_4_0
						GC[i].f_rest_array[ind++] = GC_SH.GC_SH_16[i].x; // R_4_1
						GC[i].f_rest_array[ind++] = GC_SH.GC_SH_16[i].w; // R_4_2
						GC[i].f_rest_array[ind++] = GC_SH.GC_SH_17[i].z; // R_4_3
						GC[i].f_rest_array[ind++] = GC_SH.GC_SH_18[i].y; // R_4_4
					}
				}
			}
		}

		// *** *** *** *** ***

		// Spherical harmonics: G channel
		if constexpr (SH_degree >= 1) {
			GC[i].f_rest_array[ind++] = GC_SH.GC_SH_1[i].y; // G_1_(-1)
			GC[i].f_rest_array[ind++] = GC_SH.GC_SH_2[i].x; // G_1_0
			GC[i].f_rest_array[ind++] = GC_SH.GC_SH_2[i].w; // G_1_1

			if constexpr (SH_degree >= 2) {
				GC[i].f_rest_array[ind++] = GC_SH.GC_SH_3[i].z; // G_2_(-2)
				GC[i].f_rest_array[ind++] = GC_SH.GC_SH_4[i].y; // G_2_(-1)
				GC[i].f_rest_array[ind++] = GC_SH.GC_SH_5[i].x; // G_2_0
				GC[i].f_rest_array[ind++] = GC_SH.GC_SH_5[i].w; // G_2_1
				GC[i].f_rest_array[ind++] = GC_SH.GC_SH_6[i].z; // G_2_2

				if constexpr (SH_degree >= 3) {
					GC[i].f_rest_array[ind++] = GC_SH.GC_SH_7[i].y; // G_3_(-3)
					GC[i].f_rest_array[ind++] = GC_SH.GC_SH_8[i].x; // G_3_(-2)
					GC[i].f_rest_array[ind++] = GC_SH.GC_SH_8[i].w; // G_3_(-1)
					GC[i].f_rest_array[ind++] = GC_SH.GC_SH_9[i].z; // G_3_0
					GC[i].f_rest_array[ind++] = GC_SH.GC_SH_10[i].y; // G_3_1
					GC[i].f_rest_array[ind++] = GC_SH.GC_SH_11[i].x; // G_3_2
					GC[i].f_rest_array[ind++] = GC_SH.GC_SH_11[i].w; // G_3_3

					if constexpr (SH_degree >= 4) {
						GC[i].f_rest_array[ind++] = GC_SH.GC_SH_12[i].z; // G_4_(-4)
						GC[i].f_rest_array[ind++] = GC_SH.GC_SH_13[i].y; // G_4_(-3)
						GC[i].f_rest_array[ind++] = GC_SH.GC_SH_14[i].x; // G_4_(-2)
						GC[i].f_rest_array[ind++] = GC_SH.GC_SH_14[i].w; // G_4_(-1)
						GC[i].f_rest_array[ind++] = GC_SH.GC_SH_15[i].z; // G_4_0
						GC[i].f_rest_array[ind++] = GC_SH.GC_SH_16[i].y; // G_4_1
						GC[i].f_rest_array[ind++] = GC_SH.GC_SH_17[i].x; // G_4_2
						GC[i].f_rest_array[ind++] = GC_SH.GC_SH_17[i].w; // G_4_3
						GC[i].f_rest_array[ind++] = GC_SH.GC_SH_18[i].z; // G_4_4
					}
				}
			}
		}

		// *** *** *** *** ***

		// Spherical harmonics: B channel
		if constexpr (SH_degree >= 1) {
			GC[i].f_rest_array[ind++] = GC_SH.GC_SH_1[i].z; // B_1_(-1)
			GC[i].f_rest_array[ind++] = GC_SH.GC_SH_2[i].y; // B_1_0

			if constexpr (SH_degree >= 2) {
				GC[i].f_rest_array[ind++] = GC_SH.GC_SH_3[i].x; // B_1_1

				GC[i].f_rest_array[ind++] = GC_SH.GC_SH_3[i].w; // B_2_(-2)
				GC[i].f_rest_array[ind++] = GC_SH.GC_SH_4[i].z; // B_2_(-1)
				GC[i].f_rest_array[ind++] = GC_SH.GC_SH_5[i].y; // B_2_0
				GC[i].f_rest_array[ind++] = GC_SH.GC_SH_6[i].x; // B_2_1
				GC[i].f_rest_array[ind++] = GC_SH.GC_SH_6[i].w; // B_2_2

				if constexpr (SH_degree >= 3) {
					GC[i].f_rest_array[ind++] = GC_SH.GC_SH_7[i].z; // B_3_(-3)
					GC[i].f_rest_array[ind++] = GC_SH.GC_SH_8[i].y; // B_3_(-2)
					GC[i].f_rest_array[ind++] = GC_SH.GC_SH_9[i].x; // B_3_(-1)
					GC[i].f_rest_array[ind++] = GC_SH.GC_SH_9[i].w; // B_3_0
					GC[i].f_rest_array[ind++] = GC_SH.GC_SH_10[i].z; // B_3_1
					GC[i].f_rest_array[ind++] = GC_SH.GC_SH_11[i].y; // B_3_2

					if constexpr (SH_degree >= 4) {
						GC[i].f_rest_array[ind++] = GC_SH.GC_SH_12[i].x; // B_3_3

						GC[i].f_rest_array[ind++] = GC_SH.GC_SH_12[i].w; // B_4_(-4)
						GC[i].f_rest_array[ind++] = GC_SH.GC_SH_13[i].z; // B_4_(-3)
						GC[i].f_rest_array[ind++] = GC_SH.GC_SH_14[i].y; // B_4_(-2)
						GC[i].f_rest_array[ind++] = GC_SH.GC_SH_15[i].x; // B_4_(-1)
						GC[i].f_rest_array[ind++] = GC_SH.GC_SH_15[i].w; // B_4_0
						GC[i].f_rest_array[ind++] = GC_SH.GC_SH_16[i].z; // B_4_1
						GC[i].f_rest_array[ind++] = GC_SH.GC_SH_17[i].y; // B_4_2
						GC[i].f_rest_array[ind++] = GC_SH.GC_SH_18[i].x; // B_4_3
						GC[i].f_rest_array[ind++] = GC_SH.GC_SH_18[i].w; // B_4_4
					} else
						// !!! !!! !!!
						GC[i].f_rest_array[ind++] = GC_SH.GC_SH_12[i]; // B_3_3
				}
			} else
				// !!! !!! !!!
				GC[i].f_rest_array[ind++] = GC_SH.GC_SH_3[i]; // B_1_1
		}

		// *** *** *** *** ***

		GC[i].opacity = GC1[i].w;

		// *** *** *** *** ***

		GC[i].scale_0 = GC2[i].w;
		GC[i].scale_1 = GC3[i].x;
		GC[i].scale_2 = GC3[i].y;

		// *** *** *** *** ***

		float qr = GC3[i].z;
		float qi = GC3[i].w;
		float qj = GC4[i].x;
		float qk = GC4[i].y;

		float qNorm = sqrtf((qr * qr) + (qi * qi) + (qj * qj) + (qk * qk));

		qr /= qNorm;
		qi /= qNorm;
		qj /= qNorm;
		qk /= qNorm;

		GC[i].rot_0 = qr;
		GC[i].rot_1 = qi;
		GC[i].rot_2 = qj;
		GC[i].rot_3 = qk;
	}

	// *** *** *** *** ***

	fopen_s(&f, fPath, "ab");
	fwrite(GC, sizeof(SPLYFileStruct<SH_degree>) * params_OptiX.numberOfGaussians, 1, f);
	fclose(f);

	// *** *** *** *** ***

	free(GC1);
	free(GC2);
	free(GC3);
	free(GC4);

	// *** *** *** *** ***

	// Spherical harmonics
	if constexpr (SH_degree >= 1) {
		free(GC_SH.GC_SH_1);
		free(GC_SH.GC_SH_2);
		free(GC_SH.GC_SH_3);

		if constexpr (SH_degree >= 2) {
			free(GC_SH.GC_SH_4);
			free(GC_SH.GC_SH_5);
			free(GC_SH.GC_SH_6);

			if constexpr (SH_degree >= 3) {
				free(GC_SH.GC_SH_7);
				free(GC_SH.GC_SH_8);
				free(GC_SH.GC_SH_9);
				free(GC_SH.GC_SH_10);
				free(GC_SH.GC_SH_11);
				free(GC_SH.GC_SH_12);

				if constexpr (SH_degree >= 4) {
					free(GC_SH.GC_SH_13);
					free(GC_SH.GC_SH_14);
					free(GC_SH.GC_SH_15);
					free(GC_SH.GC_SH_16);
					free(GC_SH.GC_SH_17);
					free(GC_SH.GC_SH_18);
				}
			}
		}
	}

	// *** *** *** *** ***

	free(GC);

	// *** *** *** *** ***

	return true;
Error:
	return false;
}

// *************************************************************************************************

bool DumpParametersToPLYFileOptiXSH0(SOptiXRenderParams<0>& params_OptiX, char *dirPath) {
	return DumpParametersToPLYFileOptiX<0>(params_OptiX, dirPath);
}

// *************************************************************************************************

bool DumpParametersToPLYFileOptiXSH1(SOptiXRenderParams<1>& params_OptiX, char *dirPath) {
	return DumpParametersToPLYFileOptiX<1>(params_OptiX, dirPath);
}

// *************************************************************************************************

bool DumpParametersToPLYFileOptiXSH2(SOptiXRenderParams<2>& params_OptiX, char *dirPath) {
	return DumpParametersToPLYFileOptiX<2>(params_OptiX, dirPath);
}

// *************************************************************************************************

bool DumpParametersToPLYFileOptiXSH3(SOptiXRenderParams<3>& params_OptiX, char *dirPath) {
	return DumpParametersToPLYFileOptiX<3>(params_OptiX, dirPath);
}

// *************************************************************************************************

bool DumpParametersToPLYFileOptiXSH4(SOptiXRenderParams<4>& params_OptiX, char *dirPath) {
	return DumpParametersToPLYFileOptiX<4>(params_OptiX, dirPath);
}