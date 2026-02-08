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
bool InitializeOptiXRenderer(
	SRenderParams<SH_degree> &params,
	SOptiXRenderParams<SH_degree> &params_OptiX,
	char *dirPath
) {
	cudaError_t error_CUDA;
	OptixResult error_OptiX;
	CUresult error_CUDA_Driver_API;

	error_CUDA = cudaFree(0);
	if (error_CUDA != cudaSuccess) goto Error;

	error_OptiX = optixInit();
	if (error_OptiX != OPTIX_SUCCESS) goto Error;

	CUcontext cudaContext;
	error_CUDA_Driver_API = cuCtxGetCurrent(&cudaContext);
	if(error_CUDA_Driver_API != CUDA_SUCCESS) goto Error;

	error_OptiX = optixDeviceContextCreate(cudaContext, 0, &params_OptiX.optixContext);
	if (error_OptiX != OPTIX_SUCCESS) goto Error;

	// *********************************************************************************************

	FILE *f;

	if      constexpr (SH_degree == 0) f = fopen("C:/Users/pc/source/repos/REdiSplatsSH/REdiSplatsSH/x64/Release/shaders_SH0.cu.ptx", "rb");
	else if constexpr (SH_degree == 1) f = fopen("C:/Users/pc/source/repos/REdiSplatsSH/REdiSplatsSH/x64/Release/shaders_SH1.cu.ptx", "rb");
	else if constexpr (SH_degree == 2) f = fopen("C:/Users/pc/source/repos/REdiSplatsSH/REdiSplatsSH/x64/Release/shaders_SH2.cu.ptx", "rb");
	else if constexpr (SH_degree == 3) f = fopen("C:/Users/pc/source/repos/REdiSplatsSH/REdiSplatsSH/x64/Release/shaders_SH3.cu.ptx", "rb");
	else if constexpr (SH_degree == 4) f = fopen("C:/Users/pc/source/repos/REdiSplatsSH/REdiSplatsSH/x64/Release/shaders_SH4.cu.ptx", "rb");

	fseek(f, 0, SEEK_END);
	int ptxCodeSize = ftell(f);
	fclose(f);

	char *ptxCode = (char *)malloc(sizeof(char) * (ptxCodeSize + 1));
	char *buffer = (char *)malloc(sizeof(char) * (ptxCodeSize + 1));
	ptxCode[0] = 0; // !!! !!! !!!

	if      constexpr (SH_degree == 0) f = fopen("C:/Users/pc/source/repos/REdiSplatsSH/REdiSplatsSH/x64/Release/shaders_SH0.cu.ptx", "rt");
	else if constexpr (SH_degree == 1) f = fopen("C:/Users/pc/source/repos/REdiSplatsSH/REdiSplatsSH/x64/Release/shaders_SH1.cu.ptx", "rt");
	else if constexpr (SH_degree == 2) f = fopen("C:/Users/pc/source/repos/REdiSplatsSH/REdiSplatsSH/x64/Release/shaders_SH2.cu.ptx", "rt");
	else if constexpr (SH_degree == 3) f = fopen("C:/Users/pc/source/repos/REdiSplatsSH/REdiSplatsSH/x64/Release/shaders_SH3.cu.ptx", "rt");
	else if constexpr (SH_degree == 4) f = fopen("C:/Users/pc/source/repos/REdiSplatsSH/REdiSplatsSH/x64/Release/shaders_SH4.cu.ptx", "rt");

	fgets(buffer, ptxCodeSize + 1, f);
	while (!feof(f)) {
		ptxCode = strcat(ptxCode, buffer);
		fgets(buffer, ptxCodeSize + 1, f);
	}
	fclose(f);

	free(buffer);

	// *********************************************************************************************

	OptixModuleCompileOptions moduleCompileOptions = {};
	OptixPipelineCompileOptions pipelineCompileOptions = {};

	moduleCompileOptions.maxRegisterCount = 40; // 50
	moduleCompileOptions.optLevel = OPTIX_COMPILE_OPTIMIZATION_DEFAULT;
	moduleCompileOptions.debugLevel = OPTIX_COMPILE_DEBUG_LEVEL_NONE;

	pipelineCompileOptions.traversableGraphFlags = OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_LEVEL_INSTANCING;
	pipelineCompileOptions.usesMotionBlur = false;
	#ifndef RENDERER_OPTIX_USE_DOUBLE_PRECISION
		pipelineCompileOptions.numPayloadValues = 2;
		pipelineCompileOptions.numAttributeValues = 2;
	#else
		pipelineCompileOptions.numPayloadValues = 2;
		pipelineCompileOptions.numAttributeValues = 4;
	#endif
	pipelineCompileOptions.exceptionFlags = OPTIX_EXCEPTION_FLAG_NONE;
	pipelineCompileOptions.pipelineLaunchParamsVariableName = "optixLaunchParams";
	pipelineCompileOptions.usesPrimitiveTypeFlags = OPTIX_PRIMITIVE_TYPE_FLAGS_TRIANGLE;

	OptixModule module;
	error_OptiX = optixModuleCreate(
		params_OptiX.optixContext,
		&moduleCompileOptions,
		&pipelineCompileOptions,
		ptxCode,
		strlen(ptxCode),
		NULL, NULL,
		&module
	);
	if (error_OptiX != OPTIX_SUCCESS) goto Error;

	free(ptxCode);

	// *********************************************************************************************

	OptixStackSizes oss;
	oss.cssRG = 0;
	oss.cssMS = 0;
	oss.cssCH = 0;
	oss.cssAH = 0;
	oss.cssIS = 0;
	oss.cssCC = 0;
	oss.dssDC = 0;

	// *********************************************************************************************

	OptixProgramGroupOptions pgOptions = {};

	// *********************************************************************************************

	OptixProgramGroupDesc pgDesc_raygen = {};
	pgDesc_raygen.kind = OPTIX_PROGRAM_GROUP_KIND_RAYGEN;

	pgDesc_raygen.raygen.module = module;
	if      constexpr (SH_degree == 0) pgDesc_raygen.raygen.entryFunctionName = "__raygen__SH0"; // !!! !!! !!!
	else if constexpr (SH_degree == 1) pgDesc_raygen.raygen.entryFunctionName = "__raygen__SH1"; // !!! !!! !!!
	else if constexpr (SH_degree == 2) pgDesc_raygen.raygen.entryFunctionName = "__raygen__SH2"; // !!! !!! !!!
	else if constexpr (SH_degree == 3) pgDesc_raygen.raygen.entryFunctionName = "__raygen__SH3"; // !!! !!! !!!
	else if constexpr (SH_degree == 4) pgDesc_raygen.raygen.entryFunctionName = "__raygen__SH4"; // !!! !!! !!!

	OptixProgramGroup raygenPG;
	error_OptiX = optixProgramGroupCreate(
		params_OptiX.optixContext,
		&pgDesc_raygen,
		1,
		&pgOptions,
		NULL, NULL,
		&raygenPG
	);
	if (error_OptiX != OPTIX_SUCCESS) goto Error;

	error_OptiX = optixUtilAccumulateStackSizes(raygenPG, &oss, NULL);
	if (error_OptiX != OPTIX_SUCCESS) goto Error;

	// *********************************************************************************************

	OptixProgramGroupDesc pgDesc_miss = {};
	pgDesc_miss.kind = OPTIX_PROGRAM_GROUP_KIND_MISS;

	OptixProgramGroup missPG;
	error_OptiX = optixProgramGroupCreate(
		params_OptiX.optixContext,
		&pgDesc_miss,
		1, 
		&pgOptions,
		NULL, NULL,
		&missPG
	);
	if (error_OptiX != OPTIX_SUCCESS) goto Error;

	error_OptiX = optixUtilAccumulateStackSizes(missPG, &oss, NULL);
	if (error_OptiX != OPTIX_SUCCESS) goto Error;

	// *********************************************************************************************

	OptixProgramGroupDesc pgDesc_hitgroup = {};
	pgDesc_hitgroup.kind = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;

	pgDesc_hitgroup.hitgroup.moduleAH            = module; // !!! !!! !!!
	if      constexpr (SH_degree == 0) pgDesc_hitgroup.hitgroup.entryFunctionNameAH = "__anyhit__SH0"; // !!! !!! !!!
	else if constexpr (SH_degree == 1) pgDesc_hitgroup.hitgroup.entryFunctionNameAH = "__anyhit__SH1"; // !!! !!! !!!
	else if constexpr (SH_degree == 2) pgDesc_hitgroup.hitgroup.entryFunctionNameAH = "__anyhit__SH2"; // !!! !!! !!!
	else if constexpr (SH_degree == 3) pgDesc_hitgroup.hitgroup.entryFunctionNameAH = "__anyhit__SH3"; // !!! !!! !!!
	else if constexpr (SH_degree == 4) pgDesc_hitgroup.hitgroup.entryFunctionNameAH = "__anyhit__SH4"; // !!! !!! !!!

	OptixProgramGroup hitgroupPG;
	error_OptiX = optixProgramGroupCreate(
		params_OptiX.optixContext,
		&pgDesc_hitgroup,
		1, 
		&pgOptions,
		NULL, NULL,
		&hitgroupPG
	);
	if (error_OptiX != OPTIX_SUCCESS) goto Error;

	error_OptiX = optixUtilAccumulateStackSizes(hitgroupPG, &oss, NULL);
	if (error_OptiX != OPTIX_SUCCESS) goto Error;

	// *********************************************************************************************

	OptixPipelineLinkOptions pipelineLinkOptions = {};
	pipelineLinkOptions.maxTraceDepth = 1;

	OptixProgramGroup program_groups[] = { raygenPG, missPG, hitgroupPG };

	error_OptiX = optixPipelineCreate(
		params_OptiX.optixContext,
		&pipelineCompileOptions,
		&pipelineLinkOptions,
		program_groups,
		3,
		NULL, NULL,
		&params_OptiX.pipeline
	);
	if (error_OptiX != OPTIX_SUCCESS) goto Error;

	// *********************************************************************************************

	unsigned int directCallableStackSizeFromTraversal;
	unsigned int directCallableStackSizeFromState;
	unsigned int continuationStackSize;

	error_OptiX = optixUtilComputeStackSizes(
		&oss,
		1,
		0,
		0,
		&directCallableStackSizeFromTraversal,
		&directCallableStackSizeFromState,
		&continuationStackSize 
	);
	if (error_OptiX != OPTIX_SUCCESS) goto Error;

	error_OptiX = optixPipelineSetStackSize(
		params_OptiX.pipeline, 
		directCallableStackSizeFromTraversal,
		directCallableStackSizeFromState,
		continuationStackSize,
		2
	);
	if (error_OptiX != OPTIX_SUCCESS) goto Error;

	// *********************************************************************************************

	params_OptiX.sbt = new OptixShaderBindingTable();

	// *********************************************************************************************

	SbtRecord rec_raygen;
	error_OptiX = optixSbtRecordPackHeader(raygenPG, &rec_raygen);
	if (error_OptiX != OPTIX_SUCCESS) goto Error;

	error_CUDA = cudaMalloc(&params_OptiX.raygenRecordsBuffer, sizeof(SbtRecord) * 1);
	if (error_CUDA != cudaSuccess) goto Error;

	error_CUDA = cudaMemcpy(params_OptiX.raygenRecordsBuffer, &rec_raygen, sizeof(SbtRecord) * 1, cudaMemcpyHostToDevice);
	if (error_CUDA != cudaSuccess) goto Error;

	params_OptiX.sbt->raygenRecord = (CUdeviceptr)params_OptiX.raygenRecordsBuffer;

	// *********************************************************************************************

	SbtRecord rec_miss;
	error_OptiX = optixSbtRecordPackHeader(missPG, &rec_miss);
	if (error_OptiX != OPTIX_SUCCESS) goto Error;

	error_CUDA = cudaMalloc(&params_OptiX.missRecordsBuffer, sizeof(SbtRecord) * 1);
	if (error_CUDA != cudaSuccess) goto Error;

	error_CUDA = cudaMemcpy(params_OptiX.missRecordsBuffer, &rec_miss, sizeof(SbtRecord) * 1, cudaMemcpyHostToDevice);
	if (error_CUDA != cudaSuccess) goto Error;

	params_OptiX.sbt->missRecordBase = (CUdeviceptr)params_OptiX.missRecordsBuffer;
	params_OptiX.sbt->missRecordStrideInBytes = sizeof(SbtRecord);
	params_OptiX.sbt->missRecordCount = 1;

	// *********************************************************************************************

	SbtRecord rec_hitgroup;
	error_OptiX = optixSbtRecordPackHeader(hitgroupPG, &rec_hitgroup);
	if (error_OptiX != OPTIX_SUCCESS) goto Error;

	error_CUDA = cudaMalloc(&params_OptiX.hitgroupRecordsBuffer, sizeof(SbtRecord) * 1);
	if (error_CUDA != cudaSuccess) goto Error;

	error_CUDA = cudaMemcpy(params_OptiX.hitgroupRecordsBuffer, &rec_hitgroup, sizeof(SbtRecord) * 1, cudaMemcpyHostToDevice);
	if (error_CUDA != cudaSuccess) goto Error;

	params_OptiX.sbt->hitgroupRecordBase          = (CUdeviceptr)params_OptiX.hitgroupRecordsBuffer;
	params_OptiX.sbt->hitgroupRecordStrideInBytes = sizeof(SbtRecord);
	params_OptiX.sbt->hitgroupRecordCount         = 1;

	// *********************************************************************************************

	bool densification = (params_OptiX.epoch + 1 <= densification_end_epoch_host);

	// *********************************************************************************************

	if (dirPath == NULL)
		params_OptiX.numberOfGaussians = params.numberOfGaussians; // !!! !!! !!!
	else {
		FILE *f;

		char fPath[256];
		sprintf_s(fPath, "%s\\gc1_iter_%d.checkpoint", dirPath, params_OptiX.epoch);

		fopen_s(&f, fPath, "rb");
		fseek(f, 0, SEEK_END);
		params_OptiX.numberOfGaussians = ftell(f) / sizeof(float4); // !!! !!! !!!
		fclose(f);
	}

	// !!! !!! !!!
	params_OptiX.scatterBufferSize = params_OptiX.numberOfGaussians;

	if (densification && (params_OptiX.numberOfGaussians <= max_Gaussians_per_model_host))
		params_OptiX.tmpArraysGroup1Size = params_OptiX.numberOfGaussians * tmp_arrays_growth_factor_host;
	else
		params_OptiX.tmpArraysGroup1Size = params_OptiX.numberOfGaussians;

	params_OptiX.tmpArraysGroup2Size = params_OptiX.numberOfGaussians;
	// !!! !!! !!!

	// *********************************************************************************************

	float4 *GC_part_1 = (float4 *)malloc(sizeof(float4) * params_OptiX.numberOfGaussians);
	float4 *GC_part_2 = (float4 *)malloc(sizeof(float4) * params_OptiX.numberOfGaussians);
	float4 *GC_part_3 = (float4 *)malloc(sizeof(float4) * params_OptiX.numberOfGaussians);
	float2 *GC_part_4 = (float2 *)malloc(sizeof(float2) * params_OptiX.numberOfGaussians);

	// *** *** *** *** ***

	// Spherical harmonics
	S_GC_SH<SH_degree> GC_SH;

	if constexpr (SH_degree >= 1) {
		GC_SH.GC_SH_1 = (float4 *)malloc(sizeof(float4) * params_OptiX.numberOfGaussians);
		GC_SH.GC_SH_2 = (float4 *)malloc(sizeof(float4) * params_OptiX.numberOfGaussians);

		if constexpr (SH_degree >= 2) {
			GC_SH.GC_SH_3 = (float4 *)malloc(sizeof(float4) * params_OptiX.numberOfGaussians);
			GC_SH.GC_SH_4 = (float4 *)malloc(sizeof(float4) * params_OptiX.numberOfGaussians);
			GC_SH.GC_SH_5 = (float4 *)malloc(sizeof(float4) * params_OptiX.numberOfGaussians);
			GC_SH.GC_SH_6 = (float4 *)malloc(sizeof(float4) * params_OptiX.numberOfGaussians);

			if constexpr (SH_degree >= 3) {
				GC_SH.GC_SH_7 = (float4 *)malloc(sizeof(float4) * params_OptiX.numberOfGaussians);
				GC_SH.GC_SH_8 = (float4 *)malloc(sizeof(float4) * params_OptiX.numberOfGaussians);
				GC_SH.GC_SH_9 = (float4 *)malloc(sizeof(float4) * params_OptiX.numberOfGaussians);
				GC_SH.GC_SH_10 = (float4 *)malloc(sizeof(float4) * params_OptiX.numberOfGaussians);
				GC_SH.GC_SH_11 = (float4 *)malloc(sizeof(float4) * params_OptiX.numberOfGaussians);

				if constexpr (SH_degree >= 4) {
					GC_SH.GC_SH_12 = (float4 *)malloc(sizeof(float4) * params_OptiX.numberOfGaussians);
					GC_SH.GC_SH_13 = (float4 *)malloc(sizeof(float4) * params_OptiX.numberOfGaussians);
					GC_SH.GC_SH_14 = (float4 *)malloc(sizeof(float4) * params_OptiX.numberOfGaussians);
					GC_SH.GC_SH_15 = (float4 *)malloc(sizeof(float4) * params_OptiX.numberOfGaussians);
					GC_SH.GC_SH_16 = (float4 *)malloc(sizeof(float4) * params_OptiX.numberOfGaussians);
					GC_SH.GC_SH_17 = (float4 *)malloc(sizeof(float4) * params_OptiX.numberOfGaussians);
					GC_SH.GC_SH_18 = (float4 *)malloc(sizeof(float4) * params_OptiX.numberOfGaussians);
				} else
					GC_SH.GC_SH_12 = (float *)malloc(sizeof(float) * params_OptiX.numberOfGaussians);
			}
		} else
			GC_SH.GC_SH_3 = (float *)malloc(sizeof(float) * params_OptiX.numberOfGaussians);
	}

	// *** *** *** *** ***

	if (dirPath == NULL) {
		for (int i = 0; i < params_OptiX.numberOfGaussians; ++i) {
			GC_part_1[i].x = params.GC[i].R;
			GC_part_1[i].y = params.GC[i].G;
			GC_part_1[i].z = params.GC[i].B;
			GC_part_1[i].w = params.GC[i].alpha;

			GC_part_2[i].x = params.GC[i].mX;
			GC_part_2[i].y = params.GC[i].mY;
			GC_part_2[i].z = params.GC[i].mZ;
			GC_part_2[i].w = params.GC[i].sX;

			GC_part_3[i].x = params.GC[i].sY;
			GC_part_3[i].y = params.GC[i].sZ;
			GC_part_3[i].z = params.GC[i].qr;
			GC_part_3[i].w = params.GC[i].qi;

			GC_part_4[i].x = params.GC[i].qj;
			GC_part_4[i].y = params.GC[i].qk;

			// *** *** *** *** ***

			// Spherical harmonics
			if constexpr (SH_degree >= 1) {
				GC_SH.GC_SH_1[i].x = params.GC[i].RGB_SH_higher_order[0];
				GC_SH.GC_SH_1[i].y = params.GC[i].RGB_SH_higher_order[1];
				GC_SH.GC_SH_1[i].z = params.GC[i].RGB_SH_higher_order[2];
				GC_SH.GC_SH_1[i].w = params.GC[i].RGB_SH_higher_order[3];

				GC_SH.GC_SH_2[i].x = params.GC[i].RGB_SH_higher_order[4];
				GC_SH.GC_SH_2[i].y = params.GC[i].RGB_SH_higher_order[5];
				GC_SH.GC_SH_2[i].z = params.GC[i].RGB_SH_higher_order[6];
				GC_SH.GC_SH_2[i].w = params.GC[i].RGB_SH_higher_order[7];

				if constexpr (SH_degree >= 2) {
					GC_SH.GC_SH_3[i].x = params.GC[i].RGB_SH_higher_order[8];
					GC_SH.GC_SH_3[i].y = params.GC[i].RGB_SH_higher_order[9];
					GC_SH.GC_SH_3[i].z = params.GC[i].RGB_SH_higher_order[10];
					GC_SH.GC_SH_3[i].w = params.GC[i].RGB_SH_higher_order[11];

					GC_SH.GC_SH_4[i].x = params.GC[i].RGB_SH_higher_order[12];
					GC_SH.GC_SH_4[i].y = params.GC[i].RGB_SH_higher_order[13];
					GC_SH.GC_SH_4[i].z = params.GC[i].RGB_SH_higher_order[14];
					GC_SH.GC_SH_4[i].w = params.GC[i].RGB_SH_higher_order[15];

					GC_SH.GC_SH_5[i].x = params.GC[i].RGB_SH_higher_order[16];
					GC_SH.GC_SH_5[i].y = params.GC[i].RGB_SH_higher_order[17];
					GC_SH.GC_SH_5[i].z = params.GC[i].RGB_SH_higher_order[18];
					GC_SH.GC_SH_5[i].w = params.GC[i].RGB_SH_higher_order[19];

					GC_SH.GC_SH_6[i].x = params.GC[i].RGB_SH_higher_order[20];
					GC_SH.GC_SH_6[i].y = params.GC[i].RGB_SH_higher_order[21];
					GC_SH.GC_SH_6[i].z = params.GC[i].RGB_SH_higher_order[22];
					GC_SH.GC_SH_6[i].w = params.GC[i].RGB_SH_higher_order[23];

					if constexpr (SH_degree >= 3) {
						GC_SH.GC_SH_7[i].x = params.GC[i].RGB_SH_higher_order[24];
						GC_SH.GC_SH_7[i].y = params.GC[i].RGB_SH_higher_order[25];
						GC_SH.GC_SH_7[i].z = params.GC[i].RGB_SH_higher_order[26];
						GC_SH.GC_SH_7[i].w = params.GC[i].RGB_SH_higher_order[27];

						GC_SH.GC_SH_8[i].x = params.GC[i].RGB_SH_higher_order[28];
						GC_SH.GC_SH_8[i].y = params.GC[i].RGB_SH_higher_order[29];
						GC_SH.GC_SH_8[i].z = params.GC[i].RGB_SH_higher_order[30];
						GC_SH.GC_SH_8[i].w = params.GC[i].RGB_SH_higher_order[31];

						GC_SH.GC_SH_9[i].x = params.GC[i].RGB_SH_higher_order[32];
						GC_SH.GC_SH_9[i].y = params.GC[i].RGB_SH_higher_order[33];
						GC_SH.GC_SH_9[i].z = params.GC[i].RGB_SH_higher_order[34];
						GC_SH.GC_SH_9[i].w = params.GC[i].RGB_SH_higher_order[35];

						GC_SH.GC_SH_10[i].x = params.GC[i].RGB_SH_higher_order[36];
						GC_SH.GC_SH_10[i].y = params.GC[i].RGB_SH_higher_order[37];
						GC_SH.GC_SH_10[i].z = params.GC[i].RGB_SH_higher_order[38];
						GC_SH.GC_SH_10[i].w = params.GC[i].RGB_SH_higher_order[39];

						GC_SH.GC_SH_11[i].x = params.GC[i].RGB_SH_higher_order[40];
						GC_SH.GC_SH_11[i].y = params.GC[i].RGB_SH_higher_order[41];
						GC_SH.GC_SH_11[i].z = params.GC[i].RGB_SH_higher_order[42];
						GC_SH.GC_SH_11[i].w = params.GC[i].RGB_SH_higher_order[43];

						if constexpr (SH_degree >= 4) {
							GC_SH.GC_SH_12[i].x = params.GC[i].RGB_SH_higher_order[44];
							GC_SH.GC_SH_12[i].y = params.GC[i].RGB_SH_higher_order[45];
							GC_SH.GC_SH_12[i].z = params.GC[i].RGB_SH_higher_order[46];
							GC_SH.GC_SH_12[i].w = params.GC[i].RGB_SH_higher_order[47];

							GC_SH.GC_SH_13[i].x = params.GC[i].RGB_SH_higher_order[48];
							GC_SH.GC_SH_13[i].y = params.GC[i].RGB_SH_higher_order[49];
							GC_SH.GC_SH_13[i].z = params.GC[i].RGB_SH_higher_order[50];
							GC_SH.GC_SH_13[i].w = params.GC[i].RGB_SH_higher_order[51];

							GC_SH.GC_SH_14[i].x = params.GC[i].RGB_SH_higher_order[52];
							GC_SH.GC_SH_14[i].y = params.GC[i].RGB_SH_higher_order[53];
							GC_SH.GC_SH_14[i].z = params.GC[i].RGB_SH_higher_order[54];
							GC_SH.GC_SH_14[i].w = params.GC[i].RGB_SH_higher_order[55];

							GC_SH.GC_SH_15[i].x = params.GC[i].RGB_SH_higher_order[56];
							GC_SH.GC_SH_15[i].y = params.GC[i].RGB_SH_higher_order[57];
							GC_SH.GC_SH_15[i].z = params.GC[i].RGB_SH_higher_order[58];
							GC_SH.GC_SH_15[i].w = params.GC[i].RGB_SH_higher_order[59];

							GC_SH.GC_SH_16[i].x = params.GC[i].RGB_SH_higher_order[60];
							GC_SH.GC_SH_16[i].y = params.GC[i].RGB_SH_higher_order[61];
							GC_SH.GC_SH_16[i].z = params.GC[i].RGB_SH_higher_order[62];
							GC_SH.GC_SH_16[i].w = params.GC[i].RGB_SH_higher_order[63];

							GC_SH.GC_SH_17[i].x = params.GC[i].RGB_SH_higher_order[64];
							GC_SH.GC_SH_17[i].y = params.GC[i].RGB_SH_higher_order[65];
							GC_SH.GC_SH_17[i].z = params.GC[i].RGB_SH_higher_order[66];
							GC_SH.GC_SH_17[i].w = params.GC[i].RGB_SH_higher_order[67];

							GC_SH.GC_SH_18[i].x = params.GC[i].RGB_SH_higher_order[68];
							GC_SH.GC_SH_18[i].y = params.GC[i].RGB_SH_higher_order[69];
							GC_SH.GC_SH_18[i].z = params.GC[i].RGB_SH_higher_order[70];
							GC_SH.GC_SH_18[i].w = params.GC[i].RGB_SH_higher_order[71];
						} else
							GC_SH.GC_SH_12[i] = params.GC[i].RGB_SH_higher_order[44];
					}
				} else {
					GC_SH.GC_SH_3[i] = params.GC[i].RGB_SH_higher_order[8];
				}
			}
		}
	} else {
		LoadFromFile(dirPath, params_OptiX.epoch, "gc1", GC_part_1, sizeof(float4) * params_OptiX.numberOfGaussians);
		LoadFromFile(dirPath, params_OptiX.epoch, "gc2", GC_part_2, sizeof(float4) * params_OptiX.numberOfGaussians);
		LoadFromFile(dirPath, params_OptiX.epoch, "gc3", GC_part_3, sizeof(float4) * params_OptiX.numberOfGaussians);
		LoadFromFile(dirPath, params_OptiX.epoch, "gc4", GC_part_4, sizeof(float2) * params_OptiX.numberOfGaussians);

		// *** *** *** *** ***

		// Spherical harmonics
		if constexpr (SH_degree >= 1) {
			LoadFromFile(dirPath, params_OptiX.epoch, "gc_sh_1", GC_SH.GC_SH_1, sizeof(float4) * params_OptiX.numberOfGaussians);
			LoadFromFile(dirPath, params_OptiX.epoch, "gc_sh_2", GC_SH.GC_SH_2, sizeof(float4) * params_OptiX.numberOfGaussians);

			if constexpr (SH_degree >= 2) {
				LoadFromFile(dirPath, params_OptiX.epoch, "gc_sh_3", GC_SH.GC_SH_3, sizeof(float4) * params_OptiX.numberOfGaussians);
				LoadFromFile(dirPath, params_OptiX.epoch, "gc_sh_4", GC_SH.GC_SH_4, sizeof(float4) * params_OptiX.numberOfGaussians);
				LoadFromFile(dirPath, params_OptiX.epoch, "gc_sh_5", GC_SH.GC_SH_5, sizeof(float4) * params_OptiX.numberOfGaussians);
				LoadFromFile(dirPath, params_OptiX.epoch, "gc_sh_6", GC_SH.GC_SH_6, sizeof(float4) * params_OptiX.numberOfGaussians);

				if constexpr (SH_degree >= 3) {
					LoadFromFile(dirPath, params_OptiX.epoch, "gc_sh_7", GC_SH.GC_SH_7, sizeof(float4) * params_OptiX.numberOfGaussians);
					LoadFromFile(dirPath, params_OptiX.epoch, "gc_sh_8", GC_SH.GC_SH_8, sizeof(float4) * params_OptiX.numberOfGaussians);
					LoadFromFile(dirPath, params_OptiX.epoch, "gc_sh_9", GC_SH.GC_SH_9, sizeof(float4) * params_OptiX.numberOfGaussians);
					LoadFromFile(dirPath, params_OptiX.epoch, "gc_sh_10", GC_SH.GC_SH_10, sizeof(float4) * params_OptiX.numberOfGaussians);
					LoadFromFile(dirPath, params_OptiX.epoch, "gc_sh_11", GC_SH.GC_SH_11, sizeof(float4) * params_OptiX.numberOfGaussians);

					if constexpr (SH_degree >= 4) {
						LoadFromFile(dirPath, params_OptiX.epoch, "gc_sh_12", GC_SH.GC_SH_12, sizeof(float4) * params_OptiX.numberOfGaussians);
						LoadFromFile(dirPath, params_OptiX.epoch, "gc_sh_13", GC_SH.GC_SH_13, sizeof(float4) * params_OptiX.numberOfGaussians);
						LoadFromFile(dirPath, params_OptiX.epoch, "gc_sh_14", GC_SH.GC_SH_14, sizeof(float4) * params_OptiX.numberOfGaussians);
						LoadFromFile(dirPath, params_OptiX.epoch, "gc_sh_15", GC_SH.GC_SH_15, sizeof(float4) * params_OptiX.numberOfGaussians);
						LoadFromFile(dirPath, params_OptiX.epoch, "gc_sh_16", GC_SH.GC_SH_16, sizeof(float4) * params_OptiX.numberOfGaussians);
						LoadFromFile(dirPath, params_OptiX.epoch, "gc_sh_17", GC_SH.GC_SH_17, sizeof(float4) * params_OptiX.numberOfGaussians);
						LoadFromFile(dirPath, params_OptiX.epoch, "gc_sh_18", GC_SH.GC_SH_18, sizeof(float4) * params_OptiX.numberOfGaussians);
					} else
						// !!! !!! !!!
						LoadFromFile(dirPath, params_OptiX.epoch, "gc_sh_12", GC_SH.GC_SH_12, sizeof(float) * params_OptiX.numberOfGaussians);
				}
			} else
				// !!! !!! !!!
				LoadFromFile(dirPath, params_OptiX.epoch, "gc_sh_3", GC_SH.GC_SH_3, sizeof(float) * params_OptiX.numberOfGaussians);
		}
	}

	// *********************************************************************************************

	// !!! !!! !!! !!! !!!
	if (densification) {
		error_CUDA = cudaMalloc(&needsToBeRemoved_host, sizeof(int) * params_OptiX.tmpArraysGroup1Size);
		if (error_CUDA != cudaSuccess) goto Error;

		error_CUDA = cudaMemcpyToSymbol(needsToBeRemoved, &needsToBeRemoved_host, sizeof(int *));
		if (error_CUDA != cudaSuccess) goto Error;

		error_CUDA = cudaMalloc(&Gaussians_indices_after_removal_host, sizeof(int) * params_OptiX.scatterBufferSize);
		if (error_CUDA != cudaSuccess) goto Error;
	}

	error_CUDA = cudaMalloc(&scatterBuffer, sizeof(float) * 6 * ((params_OptiX.scatterBufferSize + 31) & -32));
	if (error_CUDA != cudaSuccess) goto Error;
	// !!! !!! !!! !!! !!!

	// *********************************************************************************************

	error_CUDA = cudaMalloc(&params_OptiX.GC_part_1_1, sizeof(float4) * params_OptiX.tmpArraysGroup1Size);
	if (error_CUDA != cudaSuccess) goto Error;

	error_CUDA = cudaMalloc(&params_OptiX.GC_part_2_1, sizeof(float4) * params_OptiX.tmpArraysGroup1Size);
	if (error_CUDA != cudaSuccess) goto Error;

	error_CUDA = cudaMalloc(&params_OptiX.GC_part_3_1, sizeof(float4) * params_OptiX.tmpArraysGroup1Size);
	if (error_CUDA != cudaSuccess) goto Error;

	error_CUDA = cudaMalloc(&params_OptiX.GC_part_4_1, sizeof(float2) * params_OptiX.tmpArraysGroup1Size);
	if (error_CUDA != cudaSuccess) goto Error;

	// *** *** *** *** ***

	// Spherical harmonics
	if constexpr (SH_degree >= 1) {
		error_CUDA = cudaMalloc(&params_OptiX.GC_SH_1, sizeof(float4) * params_OptiX.tmpArraysGroup1Size);
		if (error_CUDA != cudaSuccess) goto Error;

		error_CUDA = cudaMalloc(&params_OptiX.GC_SH_2, sizeof(float4) * params_OptiX.tmpArraysGroup1Size);
		if (error_CUDA != cudaSuccess) goto Error;

		if constexpr (SH_degree >= 2) {
			error_CUDA = cudaMalloc(&params_OptiX.GC_SH_3, sizeof(float4) * params_OptiX.tmpArraysGroup1Size);
			if (error_CUDA != cudaSuccess) goto Error;

			error_CUDA = cudaMalloc(&params_OptiX.GC_SH_4, sizeof(float4) * params_OptiX.tmpArraysGroup1Size);
			if (error_CUDA != cudaSuccess) goto Error;

			error_CUDA = cudaMalloc(&params_OptiX.GC_SH_5, sizeof(float4) * params_OptiX.tmpArraysGroup1Size);
			if (error_CUDA != cudaSuccess) goto Error;

			error_CUDA = cudaMalloc(&params_OptiX.GC_SH_6, sizeof(float4) * params_OptiX.tmpArraysGroup1Size);
			if (error_CUDA != cudaSuccess) goto Error;

			if constexpr (SH_degree >= 3) {
				error_CUDA = cudaMalloc(&params_OptiX.GC_SH_7, sizeof(float4) * params_OptiX.tmpArraysGroup1Size);
				if (error_CUDA != cudaSuccess) goto Error;

				error_CUDA = cudaMalloc(&params_OptiX.GC_SH_8, sizeof(float4) * params_OptiX.tmpArraysGroup1Size);
				if (error_CUDA != cudaSuccess) goto Error;

				error_CUDA = cudaMalloc(&params_OptiX.GC_SH_9, sizeof(float4) * params_OptiX.tmpArraysGroup1Size);
				if (error_CUDA != cudaSuccess) goto Error;

				error_CUDA = cudaMalloc(&params_OptiX.GC_SH_10, sizeof(float4) * params_OptiX.tmpArraysGroup1Size);
				if (error_CUDA != cudaSuccess) goto Error;

				error_CUDA = cudaMalloc(&params_OptiX.GC_SH_11, sizeof(float4) * params_OptiX.tmpArraysGroup1Size);
				if (error_CUDA != cudaSuccess) goto Error;

				if constexpr (SH_degree >= 4) {
					error_CUDA = cudaMalloc(&params_OptiX.GC_SH_12, sizeof(float4) * params_OptiX.tmpArraysGroup1Size);
					if (error_CUDA != cudaSuccess) goto Error;

					error_CUDA = cudaMalloc(&params_OptiX.GC_SH_13, sizeof(float4) * params_OptiX.tmpArraysGroup1Size);
					if (error_CUDA != cudaSuccess) goto Error;

					error_CUDA = cudaMalloc(&params_OptiX.GC_SH_14, sizeof(float4) * params_OptiX.tmpArraysGroup1Size);
					if (error_CUDA != cudaSuccess) goto Error;

					error_CUDA = cudaMalloc(&params_OptiX.GC_SH_15, sizeof(float4) * params_OptiX.tmpArraysGroup1Size);
					if (error_CUDA != cudaSuccess) goto Error;

					error_CUDA = cudaMalloc(&params_OptiX.GC_SH_16, sizeof(float4) * params_OptiX.tmpArraysGroup1Size);
					if (error_CUDA != cudaSuccess) goto Error;

					error_CUDA = cudaMalloc(&params_OptiX.GC_SH_17, sizeof(float4) * params_OptiX.tmpArraysGroup1Size);
					if (error_CUDA != cudaSuccess) goto Error;

					error_CUDA = cudaMalloc(&params_OptiX.GC_SH_18, sizeof(float4) * params_OptiX.tmpArraysGroup1Size);
					if (error_CUDA != cudaSuccess) goto Error;
				} else {
					// !!! !!! !!!
					error_CUDA = cudaMalloc(&params_OptiX.GC_SH_12, sizeof(float) * params_OptiX.tmpArraysGroup1Size);
					if (error_CUDA != cudaSuccess) goto Error;
				}
			}
		} else {
			// !!! !!! !!!
			error_CUDA = cudaMalloc(&params_OptiX.GC_SH_3, sizeof(float) * params_OptiX.tmpArraysGroup1Size);
			if (error_CUDA != cudaSuccess) goto Error;
		}
	}

	// *** *** *** *** ***

	error_CUDA = cudaMemcpy(params_OptiX.GC_part_1_1, GC_part_1, sizeof(float4) * params_OptiX.numberOfGaussians, cudaMemcpyHostToDevice);
	if (error_CUDA != cudaSuccess) goto Error;

	error_CUDA = cudaMemcpy(params_OptiX.GC_part_2_1, GC_part_2, sizeof(float4) * params_OptiX.numberOfGaussians, cudaMemcpyHostToDevice);
	if (error_CUDA != cudaSuccess) goto Error;

	error_CUDA = cudaMemcpy(params_OptiX.GC_part_3_1, GC_part_3, sizeof(float4) * params_OptiX.numberOfGaussians, cudaMemcpyHostToDevice);
	if (error_CUDA != cudaSuccess) goto Error;

	error_CUDA = cudaMemcpy(params_OptiX.GC_part_4_1, GC_part_4, sizeof(float2) * params_OptiX.numberOfGaussians, cudaMemcpyHostToDevice);
	if (error_CUDA != cudaSuccess) goto Error;

	// *** *** *** *** ***

	// Spherical harmonics
	if constexpr (SH_degree >= 1) {
		error_CUDA = cudaMemcpy(params_OptiX.GC_SH_1, GC_SH.GC_SH_1, sizeof(float4) * params_OptiX.numberOfGaussians, cudaMemcpyHostToDevice);
		if (error_CUDA != cudaSuccess) goto Error;

		error_CUDA = cudaMemcpy(params_OptiX.GC_SH_2, GC_SH.GC_SH_2, sizeof(float4) * params_OptiX.numberOfGaussians, cudaMemcpyHostToDevice);
		if (error_CUDA != cudaSuccess) goto Error;

		if constexpr (SH_degree >= 2) {
			error_CUDA = cudaMemcpy(params_OptiX.GC_SH_3, GC_SH.GC_SH_3, sizeof(float4) * params_OptiX.numberOfGaussians, cudaMemcpyHostToDevice);
			if (error_CUDA != cudaSuccess) goto Error;

			error_CUDA = cudaMemcpy(params_OptiX.GC_SH_4, GC_SH.GC_SH_4, sizeof(float4) * params_OptiX.numberOfGaussians, cudaMemcpyHostToDevice);
			if (error_CUDA != cudaSuccess) goto Error;

			error_CUDA = cudaMemcpy(params_OptiX.GC_SH_5, GC_SH.GC_SH_5, sizeof(float4) * params_OptiX.numberOfGaussians, cudaMemcpyHostToDevice);
			if (error_CUDA != cudaSuccess) goto Error;

			error_CUDA = cudaMemcpy(params_OptiX.GC_SH_6, GC_SH.GC_SH_6, sizeof(float4) * params_OptiX.numberOfGaussians, cudaMemcpyHostToDevice);
			if (error_CUDA != cudaSuccess) goto Error;

			if constexpr (SH_degree >= 3) {
				error_CUDA = cudaMemcpy(params_OptiX.GC_SH_7, GC_SH.GC_SH_7, sizeof(float4) * params_OptiX.numberOfGaussians, cudaMemcpyHostToDevice);
				if (error_CUDA != cudaSuccess) goto Error;

				error_CUDA = cudaMemcpy(params_OptiX.GC_SH_8, GC_SH.GC_SH_8, sizeof(float4) * params_OptiX.numberOfGaussians, cudaMemcpyHostToDevice);
				if (error_CUDA != cudaSuccess) goto Error;

				error_CUDA = cudaMemcpy(params_OptiX.GC_SH_9, GC_SH.GC_SH_9, sizeof(float4) * params_OptiX.numberOfGaussians, cudaMemcpyHostToDevice);
				if (error_CUDA != cudaSuccess) goto Error;

				error_CUDA = cudaMemcpy(params_OptiX.GC_SH_10, GC_SH.GC_SH_10, sizeof(float4) * params_OptiX.numberOfGaussians, cudaMemcpyHostToDevice);
				if (error_CUDA != cudaSuccess) goto Error;

				error_CUDA = cudaMemcpy(params_OptiX.GC_SH_11, GC_SH.GC_SH_11, sizeof(float4) * params_OptiX.numberOfGaussians, cudaMemcpyHostToDevice);
				if (error_CUDA != cudaSuccess) goto Error;

				if constexpr (SH_degree >= 4) {
					error_CUDA = cudaMemcpy(params_OptiX.GC_SH_12, GC_SH.GC_SH_12, sizeof(float4) * params_OptiX.numberOfGaussians, cudaMemcpyHostToDevice);
					if (error_CUDA != cudaSuccess) goto Error;

					error_CUDA = cudaMemcpy(params_OptiX.GC_SH_13, GC_SH.GC_SH_13, sizeof(float4) * params_OptiX.numberOfGaussians, cudaMemcpyHostToDevice);
					if (error_CUDA != cudaSuccess) goto Error;

					error_CUDA = cudaMemcpy(params_OptiX.GC_SH_14, GC_SH.GC_SH_14, sizeof(float4) * params_OptiX.numberOfGaussians, cudaMemcpyHostToDevice);
					if (error_CUDA != cudaSuccess) goto Error;

					error_CUDA = cudaMemcpy(params_OptiX.GC_SH_15, GC_SH.GC_SH_15, sizeof(float4) * params_OptiX.numberOfGaussians, cudaMemcpyHostToDevice);
					if (error_CUDA != cudaSuccess) goto Error;

					error_CUDA = cudaMemcpy(params_OptiX.GC_SH_16, GC_SH.GC_SH_16, sizeof(float4) * params_OptiX.numberOfGaussians, cudaMemcpyHostToDevice);
					if (error_CUDA != cudaSuccess) goto Error;

					error_CUDA = cudaMemcpy(params_OptiX.GC_SH_17, GC_SH.GC_SH_17, sizeof(float4) * params_OptiX.numberOfGaussians, cudaMemcpyHostToDevice);
					if (error_CUDA != cudaSuccess) goto Error;

					error_CUDA = cudaMemcpy(params_OptiX.GC_SH_18, GC_SH.GC_SH_18, sizeof(float4) * params_OptiX.numberOfGaussians, cudaMemcpyHostToDevice);
					if (error_CUDA != cudaSuccess) goto Error;
				} else {
					// !!! !!! !!!
					error_CUDA = cudaMemcpy(params_OptiX.GC_SH_12, GC_SH.GC_SH_12, sizeof(float) * params_OptiX.numberOfGaussians, cudaMemcpyHostToDevice);
					if (error_CUDA != cudaSuccess) goto Error;
				}
			}
		} else {
			// !!! !!! !!!
			error_CUDA = cudaMemcpy(params_OptiX.GC_SH_3, GC_SH.GC_SH_3, sizeof(float) * params_OptiX.numberOfGaussians, cudaMemcpyHostToDevice);
			if (error_CUDA != cudaSuccess) goto Error;
		}
	}

	// *** *** *** *** ***

	free(GC_part_1);
	free(GC_part_2);
	free(GC_part_3);
	free(GC_part_4);

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

	// *********************************************************************************************

	float3 *Gaussian_as_polygon_vertices = (float3 *)malloc(sizeof(float3) * 1 * number_of_sides_host);
	int3 *Gaussian_as_polygon_indices = (int3 *)malloc(sizeof(int3) * 1 * (number_of_sides_host - 2));

	for (int i = 0; i < number_of_sides_host; ++i)
		Gaussian_as_polygon_vertices[i] = make_float3(
			0.0f,
			cosf(i * ((2.0f * M_PI) / number_of_sides_host)),
			sinf(i * ((2.0f * M_PI) / number_of_sides_host))
		);
	for (int i = 0; i < number_of_sides_host - 2; ++i)
		Gaussian_as_polygon_indices[i] = make_int3(0, i + 1, i + 2);

	// *********************************************************************************************

	error_CUDA = cudaMalloc(&params_OptiX.Gaussian_as_polygon_vertices, sizeof(float3) * 1 * number_of_sides_host);
	if (error_CUDA != cudaSuccess) goto Error;

	error_CUDA = cudaMemcpy(params_OptiX.Gaussian_as_polygon_vertices, Gaussian_as_polygon_vertices, sizeof(float3) * 1 * number_of_sides_host, cudaMemcpyHostToDevice);
	if (error_CUDA != cudaSuccess) goto Error;

	error_CUDA = cudaMalloc(&params_OptiX.Gaussian_as_polygon_indices, sizeof(int3) * 1 * (number_of_sides_host - 2));
	if (error_CUDA != cudaSuccess) goto Error;

	error_CUDA = cudaMemcpy(params_OptiX.Gaussian_as_polygon_indices, Gaussian_as_polygon_indices, sizeof(int3) * 1 * (number_of_sides_host - 2), cudaMemcpyHostToDevice);
	if (error_CUDA != cudaSuccess) goto Error;

	// *********************************************************************************************

	free(Gaussian_as_polygon_vertices);
	free(Gaussian_as_polygon_indices);

	// *********************************************************************************************

	OptixAccelBuildOptions accel_options = {};
	accel_options.buildFlags = OPTIX_BUILD_FLAG_ALLOW_COMPACTION;
	accel_options.operation  = OPTIX_BUILD_OPERATION_BUILD;

	// *********************************************************************************************

	OptixBuildInput mesh_input = {};
	mesh_input.type                           = OPTIX_BUILD_INPUT_TYPE_TRIANGLES;
	mesh_input.triangleArray.vertexBuffers    = (CUdeviceptr *)&params_OptiX.Gaussian_as_polygon_vertices;
	mesh_input.triangleArray.numVertices      = 1 * number_of_sides_host;
	mesh_input.triangleArray.vertexFormat     = OPTIX_VERTEX_FORMAT_FLOAT3;
	mesh_input.triangleArray.indexBuffer      = (CUdeviceptr)params_OptiX.Gaussian_as_polygon_indices;
	mesh_input.triangleArray.numIndexTriplets = 1 * (number_of_sides_host - 2);
	mesh_input.triangleArray.indexFormat      = OPTIX_INDICES_FORMAT_UNSIGNED_INT3;

	int mesh_input_flags[1]                = {OPTIX_GEOMETRY_FLAG_REQUIRE_SINGLE_ANYHIT_CALL};
	mesh_input.triangleArray.flags         = ((const unsigned int *)mesh_input_flags);
	mesh_input.triangleArray.numSbtRecords = 1;

	// *********************************************************************************************

	OptixAccelBufferSizes blasBufferSizes;
	error_OptiX = optixAccelComputeMemoryUsage(
		params_OptiX.optixContext,
		&accel_options,
		&mesh_input,
		1,
		&blasBufferSizes
	);
	if (error_OptiX != OPTIX_SUCCESS) goto Error;

	// *********************************************************************************************

	error_CUDA = cudaMalloc(&params_OptiX.compactedSizeBuffer, 8);
	if (error_CUDA != cudaSuccess) goto Error;

	OptixAccelEmitDesc emitDesc;
	emitDesc.type   = OPTIX_PROPERTY_TYPE_COMPACTED_SIZE;
	emitDesc.result = (CUdeviceptr)params_OptiX.compactedSizeBuffer;

	params_OptiX.tempBufferSize = blasBufferSizes.tempSizeInBytes; // !!! !!! !!!
	error_CUDA = cudaMalloc(&params_OptiX.tempBuffer, params_OptiX.tempBufferSize);
	if (error_CUDA != cudaSuccess) goto Error;

	params_OptiX.outputBufferSize = blasBufferSizes.outputSizeInBytes; // !!! !!! !!!
	error_CUDA = cudaMalloc(&params_OptiX.outputBuffer, params_OptiX.outputBufferSize);
	if (error_CUDA != cudaSuccess) goto Error;

	// *********************************************************************************************

	error_OptiX = optixAccelBuild(
		params_OptiX.optixContext,
		0,
		&accel_options,
		&mesh_input,
		1,  
		(CUdeviceptr)params_OptiX.tempBuffer,
		blasBufferSizes.tempSizeInBytes,
		(CUdeviceptr)params_OptiX.outputBuffer,
		blasBufferSizes.outputSizeInBytes,
		&params_OptiX.GAS,
		&emitDesc,
		1
	);
	if (error_OptiX != OPTIX_SUCCESS) goto Error;

	cudaDeviceSynchronize();
	error_CUDA = cudaGetLastError();
	if (error_CUDA != cudaSuccess) goto Error;

	unsigned long long compactedSize;
	error_CUDA = cudaMemcpy(&compactedSize, params_OptiX.compactedSizeBuffer, 8, cudaMemcpyDeviceToHost);
	if (error_CUDA != cudaSuccess) goto Error;

	params_OptiX.GASBufferSize = compactedSize; // !!! !!! !!! 
	error_CUDA = cudaMalloc(&params_OptiX.GASBuffer, params_OptiX.GASBufferSize);
	if (error_CUDA != cudaSuccess) goto Error;

	error_OptiX = optixAccelCompact(
		params_OptiX.optixContext,
		0,
		params_OptiX.GAS,
		(CUdeviceptr)params_OptiX.GASBuffer,
		compactedSize,
		&params_OptiX.GAS
	);
	if (error_OptiX != OPTIX_SUCCESS) goto Error;

	cudaDeviceSynchronize();
	error_CUDA = cudaGetLastError();
	if (error_CUDA != cudaSuccess) goto Error;

	// *********************************************************************************************

	error_CUDA = cudaMalloc(&params_OptiX.instancesBuffer, sizeof(OptixInstance) * ((params_OptiX.tmpArraysGroup2Size + 31) & -32)); // !!! !!! !!!
	if (error_CUDA != cudaSuccess) goto Error;

	GenerateInstances<<<(params_OptiX.numberOfGaussians + 63) >> 6, 64, ((20 * 64) + 7) << 2>>>(
		params_OptiX.GC_part_1_1, params_OptiX.GC_part_2_1, params_OptiX.GC_part_3_1, params_OptiX.GC_part_4_1,
		params_OptiX.numberOfGaussians,
		params_OptiX.GAS,
		(float *)params_OptiX.instancesBuffer
	);
	error_CUDA = cudaGetLastError();
	if (error_CUDA != cudaSuccess) goto Error;

	error_CUDA = cudaDeviceSynchronize();
	if (error_CUDA != cudaSuccess) goto Error;

	// *********************************************************************************************

	ComputeAABBs<<<(params_OptiX.numberOfGaussians + 63) >> 6, 64, ((6 * 64) + 3) << 2>>>(
		params_OptiX.GC_part_1_1, params_OptiX.GC_part_2_1, params_OptiX.GC_part_3_1, params_OptiX.GC_part_4_1,
		params_OptiX.numberOfGaussians,
		(float *)scatterBuffer
	);
	error_CUDA = cudaGetLastError();
	if (error_CUDA != cudaSuccess) goto Error;

	error_CUDA = cudaDeviceSynchronize();
	if (error_CUDA != cudaSuccess) goto Error;

	// *********************************************************************************************

	OptixAabb aabb_scene;
	OptixAabb aabb_scene_initial;

	aabb_scene_initial.minX = INFINITY;
	aabb_scene_initial.minY = INFINITY;
	aabb_scene_initial.minZ = INFINITY;

	aabb_scene_initial.maxX = -INFINITY;
	aabb_scene_initial.maxY = -INFINITY;
	aabb_scene_initial.maxZ = -INFINITY;

	try {
		aabb_scene = thrust::reduce(
			thrust::device_pointer_cast((OptixAabb *)scatterBuffer),
			thrust::device_pointer_cast((OptixAabb *)scatterBuffer) + params_OptiX.numberOfGaussians,
			aabb_scene_initial,
			SReductionOperator_OptixAabb()
		);
	} catch (...) {
		goto Error;
	}

	float dX = aabb_scene.maxX - aabb_scene.minX;
	float dY = aabb_scene.maxY - aabb_scene.minY;
	float dZ = aabb_scene.maxZ - aabb_scene.minZ;

	float scene_extent_local = sqrtf((dX * dX) + (dY * dY) + (dZ * dZ));

	error_CUDA = cudaMemcpyToSymbol(scene_extent, &scene_extent_local, sizeof(float) * 1);
	if (error_CUDA != cudaSuccess) goto Error;

	// *********************************************************************************************

	OptixBuildInput instances_input = {};
	instances_input.type                         = OPTIX_BUILD_INPUT_TYPE_INSTANCES;
	instances_input.instanceArray.instances      = (CUdeviceptr)params_OptiX.instancesBuffer;
	instances_input.instanceArray.numInstances   = params_OptiX.numberOfGaussians;
	instances_input.instanceArray.instanceStride = 0;

	// *********************************************************************************************

	error_OptiX = optixAccelComputeMemoryUsage(
		params_OptiX.optixContext,
		&accel_options,
		&instances_input,
		1,
		&blasBufferSizes
	);
	if (error_OptiX != OPTIX_SUCCESS) goto Error;

	// *********************************************************************************************

	if (blasBufferSizes.tempSizeInBytes > params_OptiX.tempBufferSize) {
		params_OptiX.tempBufferSize = blasBufferSizes.tempSizeInBytes;

		error_CUDA = cudaFree(params_OptiX.tempBuffer);
		if (error_CUDA != cudaSuccess) goto Error;

		error_CUDA = cudaMalloc(&params_OptiX.tempBuffer, params_OptiX.tempBufferSize);
		if (error_CUDA != cudaSuccess) goto Error;
	}

	if (blasBufferSizes.outputSizeInBytes > params_OptiX.outputBufferSize) {
		params_OptiX.outputBufferSize = blasBufferSizes.outputSizeInBytes;

		error_CUDA = cudaFree(params_OptiX.outputBuffer);
		if (error_CUDA != cudaSuccess) goto Error;

		error_CUDA = cudaMalloc(&params_OptiX.outputBuffer, params_OptiX.outputBufferSize);
		if (error_CUDA != cudaSuccess) goto Error;
	}
	
	// *********************************************************************************************

	error_OptiX = optixAccelBuild(
		params_OptiX.optixContext,
		0,
		&accel_options,
		&instances_input,
		1,  
		(CUdeviceptr)params_OptiX.tempBuffer,
		blasBufferSizes.tempSizeInBytes,
		(CUdeviceptr)params_OptiX.outputBuffer,
		blasBufferSizes.outputSizeInBytes,
		&params_OptiX.IAS,
		&emitDesc,
		1
	);
	if (error_OptiX != OPTIX_SUCCESS) goto Error;

	cudaDeviceSynchronize();
	error_CUDA = cudaGetLastError();
	if (error_CUDA != cudaSuccess) goto Error;

	error_CUDA = cudaMemcpy(&compactedSize, params_OptiX.compactedSizeBuffer, 8, cudaMemcpyDeviceToHost);
	if (error_CUDA != cudaSuccess) goto Error;

	params_OptiX.IASBufferSize = compactedSize; // !!! !!! !!! 
	error_CUDA = cudaMalloc(&params_OptiX.IASBuffer, params_OptiX.IASBufferSize);
	if (error_CUDA != cudaSuccess) goto Error;

	error_OptiX = optixAccelCompact(
		params_OptiX.optixContext,
		0,
		params_OptiX.IAS,
		(CUdeviceptr)params_OptiX.IASBuffer,
		compactedSize,
		&params_OptiX.IAS
	);
	if (error_OptiX != OPTIX_SUCCESS) goto Error;

	cudaDeviceSynchronize();
	error_CUDA = cudaGetLastError();
	if (error_CUDA != cudaSuccess) goto Error;

	// *********************************************************************************************

	params_OptiX.width = params.w; // !!! !!! !!!
	params_OptiX.height = params.h; // !!! !!! !!!
	params_OptiX.double_tan_half_fov_x = params.double_tan_half_fov_x; // !!! !!! !!!
	params_OptiX.double_tan_half_fov_y = params.double_tan_half_fov_y; // !!! !!! !!!

	error_CUDA = cudaMalloc(&params_OptiX.bitmap_out_device, sizeof(unsigned) * params_OptiX.width * params_OptiX.height);
	if (error_CUDA != cudaSuccess) goto Error;

	params_OptiX.bitmap_out_host = (unsigned *)params.bitmap; // !!! !!! !!!

	// *********************************************************************************************

	error_CUDA = cudaMalloc(&params_OptiX.launchParamsBuffer, sizeof(LaunchParams<SH_degree>) * 1);
	if (error_CUDA != cudaSuccess) goto Error;

	// *********************************************************************************************

	return true;
Error:
	return false;
}

// *************************************************************************************************

bool InitializeOptiXRendererSH0(
	SRenderParams<0> &params,
	SOptiXRenderParams<0> &params_OptiX,
	char *dirPath = NULL
) {
	return InitializeOptiXRenderer<0>(params, params_OptiX, dirPath);
}

// *************************************************************************************************

bool InitializeOptiXRendererSH1(
	SRenderParams<1> &params,
	SOptiXRenderParams<1> &params_OptiX,
	char *dirPath = NULL
) {
	return InitializeOptiXRenderer<1>(params, params_OptiX, dirPath);
}

// *************************************************************************************************

bool InitializeOptiXRendererSH2(
	SRenderParams<2>& params,
	SOptiXRenderParams<2>& params_OptiX,
	char *dirPath = NULL
) {
	return InitializeOptiXRenderer<2>(params, params_OptiX, dirPath);
}

// *************************************************************************************************

bool InitializeOptiXRendererSH3(
	SRenderParams<3>& params,
	SOptiXRenderParams<3>& params_OptiX,
	char *dirPath = NULL
) {
	return InitializeOptiXRenderer<3>(params, params_OptiX, dirPath);
}

// *************************************************************************************************

bool InitializeOptiXRendererSH4(
	SRenderParams<4>& params,
	SOptiXRenderParams<4>& params_OptiX,
	char *dirPath = NULL
) {
	return InitializeOptiXRenderer<4>(params, params_OptiX, dirPath);
}