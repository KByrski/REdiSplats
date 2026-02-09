#include "Header.cuh"
#include "Utils.h"

#include "cuda.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "optix.h"
#include "optix_host.h"
#include "optix_stack_size.h"
#include "optix_stubs.h"

#include <dirent.h>
#include <stdio.h>
#include <string.h>
#include <sys/stat.h>
#include <time.h>

// *** *** *** *** ***

extern void LoadConfigFile(const char* fName, SOptiXRenderConfig &config);

// *** *** *** *** ***

extern void LoadSceneAndCameraCOLMAP(
	const char *dataPath,
	const char *jsonFileName,
	int &numberOfPoses_train,
	int &numberOfPoses_test,
	SCamera *&poses_train,
	SCamera *&poses_test,
	void *&BMPFileHeader,
	int *&bitmap_train,
	int *&bitmap_test,
	int &bitmapWidth, int &bitmapHeight,
	char **&img_names_train,
	char **&img_names_test,
	int &bitmapSize,
	int &scanLineSize,
	float &double_tan_half_fov_x, float &double_tan_half_fov_y
);

// *** *** *** *** ***

extern void LoadSceneAndCamera(
	const char *dataPath,
	const char *jsonFileName,
	int &numberOfPoses,
	SCamera *&poses,
	void *&BMPFileHeader,
	int *&bitmap,
	int &bitmapWidth, int &bitmapHeight,
	char **&img_names,
	int &bitmapSize,
	int &scanLineSize,
	float &double_tan_half_fov_x, float &double_tan_half_fov_y
);

// *** *** *** *** ***

template<int SH_degree>
void LoadPLYFile(SOptiXRenderConfig &config, SGaussianComponent<SH_degree> **GC_ptr, int &numberOfGaussians);

extern template void LoadPLYFile<0>(SOptiXRenderConfig &config, SGaussianComponent<0> **GC_ptr, int &numberOfGaussians);
extern template void LoadPLYFile<1>(SOptiXRenderConfig &config, SGaussianComponent<1> **GC_ptr, int &numberOfGaussians);
extern template void LoadPLYFile<2>(SOptiXRenderConfig &config, SGaussianComponent<2> **GC_ptr, int &numberOfGaussians);
extern template void LoadPLYFile<3>(SOptiXRenderConfig &config, SGaussianComponent<3> **GC_ptr, int &numberOfGaussians);
extern template void LoadPLYFile<4>(SOptiXRenderConfig &config, SGaussianComponent<4> **GC_ptr, int &numberOfGaussians);

// *** *** *** *** ***

SOptiXRenderConfig config;
float ray_termination_T_threshold_training;

int NUMBER_OF_POSES_TRAIN, NUMBER_OF_POSES_TEST;
SCamera *poses_train, *poses_test;
void *BMPFileHeader = NULL;
int *bitmap_ref_train, *bitmap_ref_test;
int bitmapWidth, bitmapHeight;
char **img_names_train, **img_names_test;
int bitmapSize;
int scanLineSize;
float double_tan_half_fov_x, double_tan_half_fov_y;
	
// *** *** *** *** ***

template<int SH_degree>
void main_generic() {
	int next_available_dir_id;
	SGaussianComponent<SH_degree> *GC;
	int numberOfGaussians;
	double training_time;
	
	if (config.start_epoch == 0) {
		mkdir("output", 0777);

		// *****************************************************************************************

		next_available_dir_id = 0; // !!! !!! !!!

		DIR* dir = opendir("output");
		
		struct dirent* entry = readdir(dir);
		while (entry != NULL) {
			if (entry->d_name[0] != '.') {
				int tmp;

				sscanf(entry->d_name, "%d", &tmp);
				if (tmp > next_available_dir_id)
					next_available_dir_id = tmp;
			}
			entry = readdir(dir);
		}
		closedir(dir);

		++next_available_dir_id; // !!! !!! !!!

		// *****************************************************************************************

		char buf[256];

		sprintf(buf, "output/%d", next_available_dir_id);
		mkdir(buf, 0777);

		// !!! !!! !!!
		sprintf(buf, "output/%d/config.txt", next_available_dir_id);
		CopyFile(buf, "config.txt");
		// !!! !!! !!!

		sprintf(buf, "output/%d/checkpoints", next_available_dir_id);
		mkdir(buf, 0777);

		sprintf(buf, "output/%d/PLY files", next_available_dir_id);
		mkdir(buf, 0777);

		sprintf(buf, "output/%d/renders", next_available_dir_id);
		mkdir(buf, 0777);

		sprintf(buf, "output/%d/renders/train", next_available_dir_id);
		mkdir(buf, 0777);

		sprintf(buf, "output/%d/renders/test", next_available_dir_id);
		mkdir(buf, 0777);

		sprintf(buf, "output/%d/stats", next_available_dir_id);
		mkdir(buf, 0777);
		
		// *****************************************************************************************

		// LOAD PRETRAINED MODEL FROM *.PLY FILE
		LoadPLYFile<SH_degree>(config, &GC, numberOfGaussians);
		
		// *****************************************************************************************
		
		seed_dword = 0; // !!! !!! !!!
		training_time = 0.0; // !!! !!! !!!
	} else {
		sscanf(config.pretrained_model_path, "%d", &next_available_dir_id);

		// *****************************************************************************************

		FILE *f;
		char fPath[256];
		
		// *****************************************************************************************

		// !!! !!! !!!
		// Load the seed state before computing the training poses indices permutation
		sprintf(fPath, "output/%d/checkpoints/%d/seed_iter_%d.checkpoint", next_available_dir_id, config.start_epoch, config.start_epoch);
		
		f = fopen(fPath, "rb");
		fread(&seed_dword, sizeof(unsigned) * 1, 1, f);
		fclose(f);
		// !!! !!! !!!
		
		// *****************************************************************************************

		// !!! !!! !!!
		// Load the training time
		sprintf(fPath, "output/%d/checkpoints/%d/training_time_iter_%d.checkpoint", next_available_dir_id, config.start_epoch, config.start_epoch);
		
		f = fopen(fPath, "rb");
		fread(&training_time, sizeof(double) * 1, 1, f);
		fclose(f);
		// !!! !!! !!!
	}
	
	// *********************************************************************************************
	
	// GENERATE SGD INDICES PERMUTATION
	int *poses_indices = (int *)malloc(sizeof(int) * NUMBER_OF_POSES_TRAIN);
	unsigned seed_dword_prev = seed_dword; // !!! !!! !!!

	for (int i = 0; i < NUMBER_OF_POSES_TRAIN; i++) poses_indices[i] = i;
	for (int i = 0; i < NUMBER_OF_POSES_TRAIN - 1; i++) {
		int index = i + (RandomInteger() % (NUMBER_OF_POSES_TRAIN - i));
		if (index != i) {
			poses_indices[i] ^= poses_indices[index];
			poses_indices[index] ^= poses_indices[i];
			poses_indices[i] ^= poses_indices[index];
		}
	}

	int poseNum_training = config.start_epoch % NUMBER_OF_POSES_TRAIN; // !!! !!! !!!
	int poseInd_training = poses_indices[poseNum_training]; // !!! !!! !!!
	
	// *********************************************************************************************
	
	SRenderParams<SH_degree> params;
	SOptiXRenderParams<SH_degree> params_OptiX;
	
	void *bitmap = malloc(4 * bitmapWidth * bitmapHeight); // !!! !!! !!!
	
	params.double_tan_half_fov_x = double_tan_half_fov_x;
	params.double_tan_half_fov_y = double_tan_half_fov_y;
		
	params.bitmap = bitmap;
	params.w = bitmapWidth;
	params.h = bitmapHeight;
	params.GC = GC;
	params.numberOfGaussians = numberOfGaussians;

	params.bitmap_ref = (unsigned *)bitmap_ref_train;
	params.NUMBER_OF_POSES = NUMBER_OF_POSES_TRAIN;
	
	// *********************************************************************************************
	
	// !!! !!! !!!
	int epochNum;
	// !!! !!! !!!
	
	// *********************************************************************************************
	
	bool result;
	
	// *********************************************************************************************
	
	cudaError_t error_CUDA = cudaFree(0);
	if (error_CUDA != cudaSuccess) throw 0;
	
	SetConfigurationOptiX(config);
	if (config.start_epoch == 0) {
		// LOAD FROM PRETRAINED MODEL
		epochNum = 0;
		params_OptiX.epoch = epochNum;
		
		// *****************************************************************************************

		result = InitializeOptiXRenderer<SH_degree>(params, params_OptiX);
		printf("Initializing OptiX renderer: %s", (result ? "OK... .\n" : "Failed... .\n"));
		
		// *****************************************************************************************
		
		result = InitializeOptiXOptimizer<SH_degree>(params, params_OptiX);
		printf("Initializing OptiX optimizer: %s", (result ? "OK... .\n" : "Failed... .\n"));
		
		// *****************************************************************************************
		
		float scene_extent;
		result = GetSceneExtentOptiX(scene_extent);
		printf("GetSceneExtentOptiX: %s (%f)\n", (result ? "OK... ." : "Failed... ."), scene_extent);
	} else {
		epochNum = config.start_epoch; // !!! !!! !!!
		params_OptiX.epoch = epochNum; // !!! !!! !!!

		// *** *** *** *** ***

		char buf[256];	
		sprintf(buf, "output/%d/checkpoints/%d", next_available_dir_id, config.start_epoch);
		
		// *****************************************************************************************

		result = InitializeOptiXRenderer<SH_degree>(params, params_OptiX, buf); // !!! !!! !!!
		printf("Initializing OptiX renderer: %s", (result ? "OK... .\n" : "Failed... .\n"));
		
		// *****************************************************************************************
		
		result = InitializeOptiXOptimizer<SH_degree>(params, params_OptiX, buf); // !!! !!! !!!
		printf("Initializing OptiX optimizer: %s", (result ? "OK... .\n" : "Failed... .\n"));
		
		// *****************************************************************************************
		
		float scene_extent;
		result = GetSceneExtentOptiX(scene_extent);
		printf("GetSceneExtentOptiX: %s (%f)\n", (result ? "OK... ." : "Failed... ."), scene_extent);
	}
	
	// *********************************************************************************************
	
	while (true) {		
		struct timespec start, end;
		
		// *****************************************************************************************
		
		bool needs_to_visualize_train = (
			((epochNum == config.start_epoch) && config.visualization_on_startup_train) ||
			((epochNum > config.start_epoch) && (epochNum % config.visualization_frequency_train == config.visualization_iter_train)) ||
			((epochNum == config.end_epoch) && config.visualization_on_finish_train)
		);
		bool needs_to_visualize_test = (
			((epochNum == config.start_epoch) && config.visualization_on_startup_test) ||
			((epochNum > config.start_epoch) && (epochNum % config.visualization_frequency_test == config.visualization_iter_test)) ||
			((epochNum == config.end_epoch) && config.visualization_on_finish_test)
		);
		bool needs_to_evaluate_train = (
			((epochNum == config.start_epoch) && config.evaluation_on_startup_train) ||
			((epochNum > config.start_epoch) && (epochNum % config.evaluation_frequency_train == config.evaluation_iter_train)) ||
			((epochNum == config.end_epoch) && config.evaluation_on_finish_train)
		);
		bool needs_to_evaluate_test = (
			((epochNum == config.start_epoch) && config.evaluation_on_startup_test) ||
			((epochNum > config.start_epoch) && (epochNum % config.evaluation_frequency_test == config.evaluation_iter_test)) ||
			((epochNum == config.end_epoch) && config.evaluation_on_finish_test)
		);

		bool needs_to_render_train_poses = needs_to_visualize_train || needs_to_evaluate_train;
		bool needs_to_render_test_poses = needs_to_visualize_test || needs_to_evaluate_test;

		bool needs_to_render_train_or_test_poses = needs_to_render_train_poses || needs_to_render_test_poses;
		bool needs_to_visualize_train_or_test = needs_to_visualize_train || needs_to_visualize_test;

		// *****************************************************************************************

		if (needs_to_render_train_or_test_poses) {
			config.ray_termination_T_threshold = config.ray_termination_T_threshold_inference; // !!! !!! !!!
			SetConfigurationOptiX(config);
			
			// *************************************************************************************
			
			char *bitmap = NULL;
			if (needs_to_visualize_train_or_test)
				bitmap = (char *)malloc(sizeof(char) * bitmapSize);

			// *************************************************************************************

			// TRAIN
			if (needs_to_render_train_poses) {
				char buf[256];

				// *********************************************************************************

				if (needs_to_visualize_train) {
					sprintf(buf, "output/%d/renders/train/%d", next_available_dir_id, params_OptiX.epoch);
					mkdir(buf, 0777);
				}

				// *********************************************************************************

				double MSE = 0.0;
				double PSNR = 0.0;
				double FPS = 0.0;

				for (int pose = 0; pose < NUMBER_OF_POSES_TRAIN; ++pose) {
					params_OptiX.O = make_float3(poses_train[pose].Ox, poses_train[pose].Oy, poses_train[pose].Oz);
					params_OptiX.R = make_float3(poses_train[pose].Rx, poses_train[pose].Ry, poses_train[pose].Rz);
					params_OptiX.D = make_float3(poses_train[pose].Dx, poses_train[pose].Dy, poses_train[pose].Dz);
					params_OptiX.F = make_float3(poses_train[pose].Fx, poses_train[pose].Fy, poses_train[pose].Fz);
					params_OptiX.copyBitmapToHostMemory = true;
					
					clock_gettime(CLOCK_MONOTONIC_RAW, &start);
					result = RenderOptiX<SH_degree>(params_OptiX);
					clock_gettime(CLOCK_MONOTONIC_RAW, &end);
					
					printf("Render OptiX: %s", (result ? "OK... .\n" : "Failed... .\n"));
					
					// *****************************************************************************
					
					if (needs_to_visualize_train) {
						char *fName;
						char fPath[256];

						if (strcmp(config.data_format, "colmap") != 0)
							fName = strrchr(img_names_train[pose], '/') + 1; // !!! !!! !!!
						else
							fName = img_names_train[pose];

						sprintf(fPath, "%s/%s_iter_%d.bmp", buf, fName, epochNum);

						SaveBMPFile(fPath, BMPFileHeader, bitmapWidth, bitmapHeight, bitmapSize, scanLineSize, params_OptiX.bitmap_out_host, bitmap);

						printf("Saving render to file %s... .\n", fPath);
					}
					
					// *****************************************************************************
					
					if (needs_to_evaluate_train) {	
						double poseMSE = 0.0;

						for (int i = 0; i < params_OptiX.height; ++i) {
							for (int j = 0; j < params_OptiX.width; ++j) {
								int color_out = params_OptiX.bitmap_out_host[(i * params_OptiX.width) + j];
								int R_out_i = color_out >> 16;
								int G_out_i = (color_out >> 8) & 255;
								int B_out_i = color_out & 255;
								float R_out = R_out_i / 256.0f;
								float G_out = G_out_i / 256.0f;
								float B_out = B_out_i / 256.0f;

								int color_ref = bitmap_ref_train[(pose * params_OptiX.width * params_OptiX.height) + ((i * params_OptiX.width) + j)];
								int R_ref_i = color_ref >> 16;
								int G_ref_i = (color_ref >> 8) & 255;
								int B_ref_i = color_ref & 255;
								float R_ref = R_ref_i / 256.0f;
								float G_ref = G_ref_i / 256.0f;
								float B_ref = B_ref_i / 256.0f;

								poseMSE += (((R_out - R_ref) * (R_out - R_ref)) + ((G_out - G_ref) * (G_out - G_ref)) + ((B_out - B_ref) * (B_out - B_ref)));
							}
						}
						poseMSE /= 3.0 * params_OptiX.width * params_OptiX.height;
						double posePSNR = -10.0 * (log(poseMSE) / log(10.0));
						double poseFPS = 1.0 / ((end.tv_sec - start.tv_sec) + ((end.tv_nsec - start.tv_nsec) / 1e9));
						
						printf("TRAIN POSE: %d, PSNR: %.30lf, FPS: %.4lf;\n", pose + 1, posePSNR, poseFPS);
					
						FPS += poseFPS;
						MSE += poseMSE;
						PSNR += posePSNR;
					}
				}

				FPS /= NUMBER_OF_POSES_TRAIN;
				MSE /= NUMBER_OF_POSES_TRAIN;
				PSNR /= NUMBER_OF_POSES_TRAIN;
				
				// *********************************************************************************
				
				if (needs_to_evaluate_train) {
					char fPath[256];
					FILE *f;

					sprintf(fPath, "output/%d/stats/MSE_train.txt", next_available_dir_id);

					f = fopen(fPath, "at");
					fprintf(f, "%d: %.30lf,\n", epochNum, MSE);
					fclose(f);

					sprintf(fPath, "output/%d/stats/PSNR_train.txt", next_available_dir_id);

					f = fopen(fPath, "at");
					fprintf(f, "%d: %.30lf,\n", epochNum, PSNR);
					fclose(f);

					sprintf(fPath, "output/%d/stats/FPS_train.txt", next_available_dir_id);

					f = fopen(fPath, "at");
					fprintf(f, "%d: %.4lf,\n", epochNum, FPS);
					fclose(f);

					// *** *** *** *** ***

					printf("MSE TRAIN: %.30lf;\n", MSE);
					printf("PSNR TRAIN: %.30lf;\n", PSNR);
					printf("FPS TRAIN: %.4lf;\n", FPS);
				}
			}
			
			// *************************************************************************************
			
			// TEST
			if (needs_to_render_test_poses) {
				char buf[256];

				// *********************************************************************************

				if (needs_to_visualize_test) {
					sprintf(buf, "output/%d/renders/test/%d", next_available_dir_id, params_OptiX.epoch);
					mkdir(buf, 0777);
				}

				// *********************************************************************************

				double MSE = 0.0;
				double PSNR = 0.0;
				double FPS = 0.0;

				for (int pose = 0; pose < NUMBER_OF_POSES_TEST; ++pose) {
					params_OptiX.O = make_float3(poses_test[pose].Ox, poses_test[pose].Oy, poses_test[pose].Oz);
					params_OptiX.R = make_float3(poses_test[pose].Rx, poses_test[pose].Ry, poses_test[pose].Rz);
					params_OptiX.D = make_float3(poses_test[pose].Dx, poses_test[pose].Dy, poses_test[pose].Dz);
					params_OptiX.F = make_float3(poses_test[pose].Fx, poses_test[pose].Fy, poses_test[pose].Fz);
					params_OptiX.copyBitmapToHostMemory = true;
					
					clock_gettime(CLOCK_MONOTONIC_RAW, &start);
					result = RenderOptiX<SH_degree>(params_OptiX);
					clock_gettime(CLOCK_MONOTONIC_RAW, &end);
					
					printf("Render OptiX: %s", (result ? "OK... .\n" : "Failed... .\n"));
					
					// *****************************************************************************
					
					if (needs_to_visualize_test) {
						char *fName;
						char fPath[256];

						if (strcmp(config.data_format, "colmap") != 0)
							fName = strrchr(img_names_test[pose], '/') + 1; // !!! !!! !!!
						else
							fName = img_names_test[pose];

						sprintf(fPath, "%s/%s_iter_%d.bmp", buf, fName, epochNum);

						SaveBMPFile(fPath, BMPFileHeader, bitmapWidth, bitmapHeight, bitmapSize, scanLineSize, params_OptiX.bitmap_out_host, bitmap);

						printf("Saving render to file %s... .\n", fPath);
					}
					
					// *****************************************************************************
					
					if (needs_to_evaluate_test) {	
						double poseMSE = 0.0;

						for (int i = 0; i < params_OptiX.height; ++i) {
							for (int j = 0; j < params_OptiX.width; ++j) {
								int color_out = params_OptiX.bitmap_out_host[(i * params_OptiX.width) + j];
								int R_out_i = color_out >> 16;
								int G_out_i = (color_out >> 8) & 255;
								int B_out_i = color_out & 255;
								float R_out = R_out_i / 256.0f;
								float G_out = G_out_i / 256.0f;
								float B_out = B_out_i / 256.0f;

								int color_ref = bitmap_ref_test[(pose * params_OptiX.width * params_OptiX.height) + ((i * params_OptiX.width) + j)];
								int R_ref_i = color_ref >> 16;
								int G_ref_i = (color_ref >> 8) & 255;
								int B_ref_i = color_ref & 255;
								float R_ref = R_ref_i / 256.0f;
								float G_ref = G_ref_i / 256.0f;
								float B_ref = B_ref_i / 256.0f;

								poseMSE += (((R_out - R_ref) * (R_out - R_ref)) + ((G_out - G_ref) * (G_out - G_ref)) + ((B_out - B_ref) * (B_out - B_ref)));
							}
						}
						poseMSE /= 3.0 * params_OptiX.width * params_OptiX.height;
						double posePSNR = -10.0 * (log(poseMSE) / log(10.0));
						double poseFPS = 1.0 / ((end.tv_sec - start.tv_sec) + ((end.tv_nsec - start.tv_nsec) / 1e9));
						
						printf("TEST POSE: %d, PSNR: %.30lf, FPS: %.4lf;\n", pose + 1, posePSNR, poseFPS);
					
						FPS += poseFPS;
						MSE += poseMSE;
						PSNR += posePSNR;
					}
				}

				FPS /= NUMBER_OF_POSES_TEST;
				MSE /= NUMBER_OF_POSES_TEST;
				PSNR /= NUMBER_OF_POSES_TEST;
				
				// *********************************************************************************
				
				if (needs_to_evaluate_test) {
					char fPath[256];
					FILE *f;

					sprintf(fPath, "output/%d/stats/MSE_test.txt", next_available_dir_id);

					f = fopen(fPath, "at");
					fprintf(f, "%d: %.30lf,\n", epochNum, MSE);
					fclose(f);

					sprintf(fPath, "output/%d/stats/PSNR_test.txt", next_available_dir_id);

					f = fopen(fPath, "at");
					fprintf(f, "%d: %.30lf,\n", epochNum, PSNR);
					fclose(f);

					sprintf(fPath, "output/%d/stats/FPS_test.txt", next_available_dir_id);

					f = fopen(fPath, "at");
					fprintf(f, "%d: %.4lf,\n", epochNum, FPS);
					fclose(f);

					// *** *** *** *** ***

					printf("MSE TEST: %.30lf;\n", MSE);
					printf("PSNR TEST: %.30lf;\n", PSNR);
					printf("FPS TEST: %.4lf;\n", FPS);
				}
			}
			
			// *************************************************************************************

			if (needs_to_visualize_train_or_test)
				free(bitmap);

			// *************************************************************************************

			config.ray_termination_T_threshold = ray_termination_T_threshold_training; // !!! !!! !!!
			SetConfigurationOptiX(config);
		}
			
		// *****************************************************************************************
		
		++epochNum;
		
		// !!! !!! !!!
		if (epochNum > config.end_epoch)
			return;
		// !!! !!! !!!
		
		// *****************************************************************************************
		
		clock_gettime(CLOCK_MONOTONIC_RAW, &start); 
		result = ZeroGradientOptiX<SH_degree>(params_OptiX);
		clock_gettime(CLOCK_MONOTONIC_RAW, &end);
		
		training_time += ((end.tv_sec - start.tv_sec) + ((end.tv_nsec - start.tv_nsec) / 1e9));
		
		printf("Zero OptiX gradient: %s", (result ? "OK... .\n" : "Failed... .\n"));
		
		// *****************************************************************************************
		
		clock_gettime(CLOCK_MONOTONIC_RAW, &start);
		
		params_OptiX.O = make_float3(poses_train[poseInd_training].Ox, poses_train[poseInd_training].Oy, poses_train[poseInd_training].Oz);
		params_OptiX.R = make_float3(poses_train[poseInd_training].Rx, poses_train[poseInd_training].Ry, poses_train[poseInd_training].Rz);
		params_OptiX.D = make_float3(poses_train[poseInd_training].Dx, poses_train[poseInd_training].Dy, poses_train[poseInd_training].Dz);
		params_OptiX.F = make_float3(poses_train[poseInd_training].Fx, poses_train[poseInd_training].Fy, poses_train[poseInd_training].Fz);

		params_OptiX.poseNum = poseInd_training;
		params_OptiX.epoch = epochNum;
		params_OptiX.copyBitmapToHostMemory = false;

		result = RenderOptiX<SH_degree>(params_OptiX, false);
		
		clock_gettime(CLOCK_MONOTONIC_RAW, &end);
		
		training_time += ((end.tv_sec - start.tv_sec) + ((end.tv_nsec - start.tv_nsec) / 1e9));
		
		printf("Render OptiX: %s", (result ? "OK... .\n" : "Failed... .\n"));
		
		// *****************************************************************************************
		
		clock_gettime(CLOCK_MONOTONIC_RAW, &start);
		result = UpdateGradientOptiX<SH_degree>(params_OptiX);
		clock_gettime(CLOCK_MONOTONIC_RAW, &end);
		
		training_time += ((end.tv_sec - start.tv_sec) + ((end.tv_nsec - start.tv_nsec) / 1e9));
		
		printf("Update gradient OptiX: %s", (result ? "OK... .\n" : "Failed... .\n"));
		printf("EPOCH: %d, GAUSSIANS: %d, MSE: %.20lf\n", epochNum, params_OptiX.numberOfGaussians, params_OptiX.loss_host / (3.0 * bitmapWidth * bitmapHeight * 1));
	
		// *****************************************************************************************
		
		clock_gettime(CLOCK_MONOTONIC_RAW, &start);
		if (poseNum_training < NUMBER_OF_POSES_TRAIN - 1)
			++poseNum_training;
		else {
			seed_dword_prev = seed_dword; // !!! !!! !!!

			for (int i = 0; i < NUMBER_OF_POSES_TRAIN; i++) poses_indices[i] = i;
			for (int i = 0; i < NUMBER_OF_POSES_TRAIN - 1; i++) {
				int index = i + (RandomInteger() % (NUMBER_OF_POSES_TRAIN - i));
				if (index != i) {
					poses_indices[i] ^= poses_indices[index];
					poses_indices[index] ^= poses_indices[i];
					poses_indices[i] ^= poses_indices[index];
				}
			}

			poseNum_training = 0; // !!! !!! !!!
		}
		
		poseInd_training = poses_indices[poseNum_training]; // !!! !!! !!!
		clock_gettime(CLOCK_MONOTONIC_RAW, &end);
		
		training_time += ((end.tv_sec - start.tv_sec) + ((end.tv_nsec - start.tv_nsec) / 1e9));
		
		// *****************************************************************************************
		
		// SAVE CHECKPOINT
		if (
			(params_OptiX.epoch % config.saving_frequency == config.saving_iter) ||
			(params_OptiX.epoch == config.end_epoch)
		) {
			char buf[256];
			
			sprintf(buf, "output/%d/checkpoints/%d", next_available_dir_id, params_OptiX.epoch);
			mkdir(buf, 0777);

			result = DumpParametersOptiX<SH_degree>(params_OptiX, buf);
			
			// *************************************************************************************
			
			char fPath[256];
			FILE *f;
			
			// *************************************************************************************
			
			// !!! !!! !!!
			// Save the seed state before computing the training poses indices permutation
			sprintf(fPath, "%s/seed_iter_%d.checkpoint", buf, params_OptiX.epoch);

			f = fopen(fPath, "wb");
			fwrite(&seed_dword_prev, sizeof(unsigned) * 1, 1, f);
			fclose(f);
			// !!! !!! !!!
			
			// *************************************************************************************
			
			// !!! !!! !!!
			// Save the training time
			sprintf(fPath, "%s/training_time_iter_%d.checkpoint", buf, params_OptiX.epoch);

			f = fopen(fPath, "wb");
			fwrite(&training_time, sizeof(double) * 1, 1, f);
			fclose(f);
			// !!! !!! !!!
		}
		
		// *****************************************************************************************
		
		// SAVE TO *.PLY FILE
		if (
			(params_OptiX.epoch % config.saving_frequency_PLY == config.saving_iter_PLY) ||
			(params_OptiX.epoch == config.end_epoch)
		) {
			char buf[256];
			
			sprintf(buf, "output/%d/PLY files/%d", next_available_dir_id, params_OptiX.epoch);
			mkdir(buf, 0777);

			result = DumpParametersToPLYFileOptiX<SH_degree>(params_OptiX, buf);
		}
		
		// *****************************************************************************************
		
		// REPORT TRAINING TIME
		if (
			(params_OptiX.epoch % config.evaluation_frequency_train == config.evaluation_iter_train) ||
			(params_OptiX.epoch % config.evaluation_frequency_test == config.evaluation_iter_test) ||
			(params_OptiX.epoch == config.end_epoch)
		) {
			char fPath[256];
			FILE *f;

			sprintf(fPath, "output/%d/stats/training_time.txt", next_available_dir_id);

			f = fopen(fPath, "at");
			fprintf(f, "%d: %.2lf,\n", params_OptiX.epoch, training_time);
			fclose(f);
		}
	}
}

// *** *** *** *** ***

int main() {
	LoadConfigFile("config.txt", config);
	ray_termination_T_threshold_training = config.ray_termination_T_threshold;
	
	// *********************************************************************************************
	
	if (strcmp(config.data_format, "colmap") == 0)
		LoadSceneAndCameraCOLMAP(
			config.data_path,
			"cameras.json",
			NUMBER_OF_POSES_TRAIN,
			NUMBER_OF_POSES_TEST,
			poses_train,
			poses_test,
			BMPFileHeader,
			bitmap_ref_train,
			bitmap_ref_test,
			bitmapWidth, bitmapHeight,
			img_names_train,
			img_names_test,
			bitmapSize,
			scanLineSize,
			double_tan_half_fov_x, double_tan_half_fov_y
		);
	else {
		LoadSceneAndCamera(
			config.data_path,
			"transforms_train.json",
			NUMBER_OF_POSES_TRAIN,
			poses_train,
			BMPFileHeader,
			bitmap_ref_train,
			bitmapWidth, bitmapHeight,
			img_names_train,
			bitmapSize,
			scanLineSize,
			double_tan_half_fov_x, double_tan_half_fov_y
		);
		LoadSceneAndCamera(
			config.data_path,
			"transforms_test.json",
			NUMBER_OF_POSES_TEST,
			poses_test,
			BMPFileHeader,
			bitmap_ref_test,
			bitmapWidth, bitmapHeight,
			img_names_test,
			bitmapSize,
			scanLineSize,
			double_tan_half_fov_x, double_tan_half_fov_y
		);
	}
	
	// *********************************************************************************************
	
	switch (config.SH_degree) {
		case 0 : {
			main_generic<0>();
			break;
		}
		case 1 : {
			main_generic<1>();
			break;
		}
		case 2 : {
			main_generic<2>();
			break;
		}
		case 3 : {
			main_generic<3>();
			break;
		}
		case 4 : {
			main_generic<4>();
			break;
		}
	}
}