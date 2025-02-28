#include "Header.cuh"

// *************************************************************************************************

static void DumpToFile(const char *fPath, int epochNum, const char *fExtension, void *buf, int size) {
	FILE *f;

	char fName[256];
	sprintf_s(fName, "%s/%d.%s", fPath, epochNum, fExtension);

	fopen_s(&f, fName, "wb");
	fwrite(buf, size, 1, f);
	fclose(f);
}

// *************************************************************************************************

bool DumpParameters(SOptiXRenderParams& params_OptiX) {
	cudaError_t error_CUDA;

	void *buf = malloc(sizeof(REAL4_G) * params_OptiX.numberOfGaussians);
	if (buf == NULL) goto Error;

	error_CUDA = cudaMemcpy(buf, params_OptiX.GC_part_1_1, sizeof(float4) * params_OptiX.numberOfGaussians, cudaMemcpyDeviceToHost);
	if (error_CUDA != cudaSuccess) goto Error;
	DumpToFile("dump/save", params_OptiX.epoch, "GC1", buf, sizeof(float4) * params_OptiX.numberOfGaussians);

	error_CUDA = cudaMemcpy(buf, params_OptiX.GC_part_2_1, sizeof(float4) * params_OptiX.numberOfGaussians, cudaMemcpyDeviceToHost);
	if (error_CUDA != cudaSuccess) goto Error;
	DumpToFile("dump/save", params_OptiX.epoch, "GC2", buf, sizeof(float4) * params_OptiX.numberOfGaussians);

	error_CUDA = cudaMemcpy(buf, params_OptiX.GC_part_3_1, sizeof(float4) * params_OptiX.numberOfGaussians, cudaMemcpyDeviceToHost);
	if (error_CUDA != cudaSuccess) goto Error;
	DumpToFile("dump/save", params_OptiX.epoch, "GC3", buf, sizeof(float4) * params_OptiX.numberOfGaussians);

	error_CUDA = cudaMemcpy(buf, params_OptiX.GC_part_4_1, sizeof(float2) * params_OptiX.numberOfGaussians, cudaMemcpyDeviceToHost);
	if (error_CUDA != cudaSuccess) goto Error;
	DumpToFile("dump/save", params_OptiX.epoch, "GC4", buf, sizeof(float2) * params_OptiX.numberOfGaussians);

	error_CUDA = cudaMemcpy(buf, params_OptiX.m11, sizeof(float4) * params_OptiX.numberOfGaussians, cudaMemcpyDeviceToHost);
	if (error_CUDA != cudaSuccess) goto Error;
	DumpToFile("dump/save", params_OptiX.epoch, "m1", buf, sizeof(float4) * params_OptiX.numberOfGaussians);

	error_CUDA = cudaMemcpy(buf, params_OptiX.m21, sizeof(float4) * params_OptiX.numberOfGaussians, cudaMemcpyDeviceToHost);
	if (error_CUDA != cudaSuccess) goto Error;
	DumpToFile("dump/save", params_OptiX.epoch, "m2", buf, sizeof(float4) * params_OptiX.numberOfGaussians);

	error_CUDA = cudaMemcpy(buf, params_OptiX.m31, sizeof(float4) * params_OptiX.numberOfGaussians, cudaMemcpyDeviceToHost);
	if (error_CUDA != cudaSuccess) goto Error;
	DumpToFile("dump/save", params_OptiX.epoch, "m3", buf, sizeof(float4) * params_OptiX.numberOfGaussians);

	error_CUDA = cudaMemcpy(buf, params_OptiX.m41, sizeof(float2) * params_OptiX.numberOfGaussians, cudaMemcpyDeviceToHost);
	if (error_CUDA != cudaSuccess) goto Error;
	DumpToFile("dump/save", params_OptiX.epoch, "m4", buf, sizeof(float2) * params_OptiX.numberOfGaussians);

	error_CUDA = cudaMemcpy(buf, params_OptiX.v11, sizeof(float4) * params_OptiX.numberOfGaussians, cudaMemcpyDeviceToHost);
	if (error_CUDA != cudaSuccess) goto Error;
	DumpToFile("dump/save", params_OptiX.epoch, "v1", buf, sizeof(float4) * params_OptiX.numberOfGaussians);

	error_CUDA = cudaMemcpy(buf, params_OptiX.v21, sizeof(float4) * params_OptiX.numberOfGaussians, cudaMemcpyDeviceToHost);
	if (error_CUDA != cudaSuccess) goto Error;
	DumpToFile("dump/save", params_OptiX.epoch, "v2", buf, sizeof(float4) * params_OptiX.numberOfGaussians);

	error_CUDA = cudaMemcpy(buf, params_OptiX.v31, sizeof(float4) * params_OptiX.numberOfGaussians, cudaMemcpyDeviceToHost);
	if (error_CUDA != cudaSuccess) goto Error;
	DumpToFile("dump/save", params_OptiX.epoch, "v3", buf, sizeof(float4) * params_OptiX.numberOfGaussians);

	error_CUDA = cudaMemcpy(buf, params_OptiX.v41, sizeof(float2) * params_OptiX.numberOfGaussians, cudaMemcpyDeviceToHost);
	if (error_CUDA != cudaSuccess) goto Error;
	DumpToFile("dump/save", params_OptiX.epoch, "v4", buf, sizeof(float2) * params_OptiX.numberOfGaussians);

	// *** *** *** *** ***

	// DEBUG GRADIENT
	/*error_CUDA = cudaMemcpy(buf, params_OptiX.dL_dparams_1, sizeof(REAL4_G) * params_OptiX.numberOfGaussians, cudaMemcpyDeviceToHost);
	if (error_CUDA != cudaSuccess) goto Error;
	DumpToFile("dump/save", params_OptiX.epoch, "params1", buf, sizeof(REAL4_G) * params_OptiX.numberOfGaussians);

	error_CUDA = cudaMemcpy(buf, params_OptiX.dL_dparams_2, sizeof(REAL4_G) * params_OptiX.numberOfGaussians, cudaMemcpyDeviceToHost);
	if (error_CUDA != cudaSuccess) goto Error;
	DumpToFile("dump/save", params_OptiX.epoch, "params2", buf, sizeof(REAL4_G) * params_OptiX.numberOfGaussians);

	error_CUDA = cudaMemcpy(buf, params_OptiX.dL_dparams_3, sizeof(REAL4_G) * params_OptiX.numberOfGaussians, cudaMemcpyDeviceToHost);
	if (error_CUDA != cudaSuccess) goto Error;
	DumpToFile("dump/save", params_OptiX.epoch, "params3", buf, sizeof(REAL4_G) * params_OptiX.numberOfGaussians);

	error_CUDA = cudaMemcpy(buf, params_OptiX.dL_dparams_4, sizeof(REAL2_G) * params_OptiX.numberOfGaussians, cudaMemcpyDeviceToHost);
	if (error_CUDA != cudaSuccess) goto Error;
	DumpToFile("dump/save", params_OptiX.epoch, "params4", buf, sizeof(REAL2_G) * params_OptiX.numberOfGaussians);

	int *foo = (int *)malloc(sizeof(int) * params_OptiX.width * params_OptiX.height * ray length);
	error_CUDA = cudaMemcpy(foo, params_OptiX.Gaussians_indices, sizeof(int) * params_OptiX.width * params_OptiX.height * ray length, cudaMemcpyDeviceToHost);
	if (error_CUDA != cudaSuccess) goto Error;
	for (int i = 0; i < params_OptiX.height; ++i) {
		for (int j = 0; j < params_OptiX.width; ++j) {
			int k = 0;
			int ind;
			do {
				ind = foo[(k * (params_OptiX.width * params_OptiX.height)) + (i * params_OptiX.width) + j];
				++k;
			} while ((ind != -1) && (k < MAX_RAY_LENGTH));
			while (k < MAX_RAY_LENGTH) {
				foo[(k * (params_OptiX.width * params_OptiX.height)) + (i * params_OptiX.width) + j] = -1;
				++k;
			}
		}
	}
	DumpToFile("dump/save", params_OptiX.epoch, "indices", foo, sizeof(int) * params_OptiX.width * params_OptiX.height * ray length);
	free(foo);*/

	// *** *** *** *** ***

	free(buf);

	return true;
Error:
	return false;
}

// *************************************************************************************************

bool DumpParametersToPLYFile(SOptiXRenderParams& params_OptiX) {
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

	cudaError_t error_CUDA;

	float4 *GC1 = (float4 *)malloc(sizeof(float4) * params_OptiX.numberOfGaussians);
	if (GC1 == NULL) goto Error;

	float4 *GC2 = (float4 *)malloc(sizeof(float4) * params_OptiX.numberOfGaussians);
	if (GC2 == NULL) goto Error;

	float4 *GC3 = (float4 *)malloc(sizeof(float4) * params_OptiX.numberOfGaussians);
	if (GC3 == NULL) goto Error;

	float2 *GC4 = (float2 *)malloc(sizeof(float2) * params_OptiX.numberOfGaussians);
	if (GC4 == NULL) goto Error;

	SPLYFileStruct *GC = (SPLYFileStruct *)malloc(sizeof(SPLYFileStruct) * params_OptiX.numberOfGaussians);
	if (GC == NULL) goto Error;

	error_CUDA = cudaMemcpy(GC1, params_OptiX.GC_part_1_1, sizeof(float4) * params_OptiX.numberOfGaussians, cudaMemcpyDeviceToHost);
	if (error_CUDA != cudaSuccess) goto Error;

	error_CUDA = cudaMemcpy(GC2, params_OptiX.GC_part_2_1, sizeof(float4) * params_OptiX.numberOfGaussians, cudaMemcpyDeviceToHost);
	if (error_CUDA != cudaSuccess) goto Error;

	error_CUDA = cudaMemcpy(GC3, params_OptiX.GC_part_3_1, sizeof(float4) * params_OptiX.numberOfGaussians, cudaMemcpyDeviceToHost);
	if (error_CUDA != cudaSuccess) goto Error;

	error_CUDA = cudaMemcpy(GC4, params_OptiX.GC_part_4_1, sizeof(float2) * params_OptiX.numberOfGaussians, cudaMemcpyDeviceToHost);
	if (error_CUDA != cudaSuccess) goto Error;

	// *** *** *** *** ***

	FILE *f;
	char fName[256];

	sprintf_s(fName, "dump/save/%d.ply", params_OptiX.epoch);
	fopen_s(&f, fName, "wb");

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

		GC[i].nx = 0.0f;
		GC[i].ny = 0.0f;
		GC[i].nz = 0.0f;

		GC[i].f_dc_0 = (GC1[i].x - 0.5f) / 0.28209479177387814f;
		GC[i].f_dc_1 = (GC1[i].y - 0.5f) / 0.28209479177387814f;
		GC[i].f_dc_2 = (GC1[i].z - 0.5f) / 0.28209479177387814f;
		GC[i].opacity = GC1[i].w;

		// OLD INVERSE SIGMOID ACTIVATION FUNCTION FOR SCALE PARAMETERS
		/*GC[i].scale_0 = logf(1.0f / (1.0f + expf(-GC2[i].w)));
		GC[i].scale_1 = logf(1.0f / (1.0f + expf(-GC3[i].x)));
		GC[i].scale_2 = logf(1.0f / (1.0f + expf(-GC3[i].y)));*/

		// NEW EXPONENTIAL ACTIVATION FUNCTION FOR SCALE PARAMETERS
		GC[i].scale_0 = GC2[i].w;
		GC[i].scale_1 = GC3[i].x;
		GC[i].scale_2 = GC3[i].y;

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

	fopen_s(&f, fName, "ab");
	fwrite(GC, sizeof(SPLYFileStruct) * params_OptiX.numberOfGaussians, 1, f);
	fclose(f);

	// *** *** *** *** ***

	free(GC1);
	free(GC2);
	free(GC3);
	free(GC4);
	free(GC);

	return true;
Error:
	return false;
}