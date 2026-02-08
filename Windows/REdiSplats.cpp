#define _USE_MATH_DEFINES

// *** *** *** *** ***

#include <intrin.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include "framework.h"
#include <Windowsx.h>
#include <CommCtrl.h>

#include "Renderer.h"
#include "REdiSplats.h"

// *** *** *** *** ***

#define MAX_LOADSTRING 100

// *** *** *** *** ***

#pragma comment(linker,"\"/manifestdependency:type='win32' \
name='Microsoft.Windows.Common-Controls' version='6.0.0.0' \
processorArchitecture='*' publicKeyToken='6595b64144ccf1df' language='*'\"")

// *** *** *** *** ***

HINSTANCE hInst;
HWND hWnd;
WCHAR szTitle[MAX_LOADSTRING];                 
WCHAR szWindowClass[MAX_LOADSTRING];

// *** *** *** *** ***

ATOM MyRegisterClass(HINSTANCE hInstance);

template<int SH_degree>
BOOL InitInstance(HINSTANCE, int);

template<int SH_degree>
LRESULT CALLBACK WndProc(HWND, UINT, WPARAM, LPARAM);

INT_PTR CALLBACK About(HWND, UINT, WPARAM, LPARAM);

// *** *** *** *** ***

#define TIMER1 0
#define LABEL1 1

// *** *** *** *** ***

// !!! !!! !!!
unsigned seed_dword_prev;
// !!! !!! !!!

// *** *** *** *** ***

int next_available_dir_id;
int next_available_screenshot_id;

// *** *** *** *** ***

LARGE_INTEGER lpFrequency;
double training_time = 0.0f;

// *** *** *** *** ***

int mouseX;
int mouseY;
bool cursorShown = true;
bool customPose = false;
bool cameraChanged;

bool A_down = false;
bool D_down = false;
bool W_down = false;
bool S_down = false;
bool C_down = false;
bool Space_down = false;
bool left_square_bracket_down = false;
bool right_square_bracket_down = false;
bool print_screen_down = false;

bool flyMode = false;

int XCnt = 0;
int YCnt = 0;

// *** *** *** *** ***

int bitmapSize;
void *BMPFileHeader = NULL;
int bitmapWidth;
int bitmapHeight;
int scanLineSize;

wchar_t *text = NULL;
wchar_t consoleBuffer[256];
HBITMAP hBitmap;
void *bitmap = NULL;
int iteration = 0;

float Ox = 0.0f; float Oy = 0.0f; float Oz = 0.0f;
float Rx = 1.0f; float Ry = 0.0f; float Rz = 0.0f;
float Dx = 0.0f; float Dy = 1.0f; float Dz = 0.0f;
float Fx = 0.0f; float Fy = 0.0f; float Fz = 1.0f;

float yaw = 0.0f;
float pitch = 0.0f;

float double_tan_half_fov_x;
float double_tan_half_fov_y;

void *params;
void *params_OptiX;

// *** *** *** *** ***

int NUMBER_OF_POSES;
int NUMBER_OF_POSES_TEST;

SCamera *poses;
SCamera *poses_test;

int *bitmap_ref;
int *bitmap_ref_test;

char **img_names; // !!! !!! !!!
char **img_names_test; // !!! !!! !!!

int poseNum_rendering;
int poseNum_training;
int poseInd_training;
int epochNum;

int numberOfGaussians;

void *GC; // !!! !!! !!!

int *poses_indices;

// *** *** *** *** ***

SOptiXRenderConfig config;
float ray_termination_T_threshold_training;

// *** *** *** *** ***

double FPS_inference = INFINITY;

// *************************************************************************************************

void LoadSceneAndCamera(
	const char *dataPath,
	const char *jsonFileName,
	int &numberOfPoses,
	SCamera *&poses,
	void *&BMPFileHeader,
	int *&bitmap,
	int &bitmapWidth, int &bitmapHeight,
	char **&img_names
) {
	FILE *f;

	// *** *** *** *** ***

	char filePath[256];
	strcpy_s(filePath, dataPath);
	strcat_s(filePath, "/");
	strcat_s(filePath, jsonFileName);
	fopen_s(&f, filePath, "rt");
		
	char buf[256];
	numberOfPoses = 0;
	while (fgets(buf, 256, f) != NULL) {
		char *str = strstr(buf, "file_path");
		if (str != NULL) ++numberOfPoses;
	}
	
	poses = (SCamera*)malloc(sizeof(SCamera) * numberOfPoses);
	void *bitmap_tmp = NULL;
	img_names = (char **)malloc(sizeof(char *) * numberOfPoses); // !!! !!! !!!

	int poseNum = 0;
	fseek(f, 0, SEEK_SET);

	fgets(buf, 256, f);

	float FOV;
	fgets(buf, 256, f);
	sscanf_s(buf, " \"camera_angle_x\": %f", &FOV);

	while (fgets(buf, 256, f) != NULL) {
		char *str = strstr(buf, "file_path");
		if (str != NULL) {
			char fileName[256];
			sscanf_s(str, "file_path\": \"%s", fileName, 256);

			char *next ;
			char *tmp = strtok_s(fileName, "\"", &next);
			img_names[poseNum] = (char *)malloc(sizeof(char) * (strlen(tmp) + 1));
			strcpy_s(img_names[poseNum], strlen(tmp) + 1, tmp);

			FILE *f_bitmap;
			strcpy_s(filePath, dataPath);
			strcat_s(filePath, "/");
			strcat_s(filePath, tmp);
			strcat_s(filePath, ".bmp");

			fopen_s(&f_bitmap, filePath, "rb");

			if (poseNum == 0) {
				fseek(f_bitmap, 0, SEEK_END);
				bitmapSize = ftell(f_bitmap);
				fseek(f_bitmap, 0, SEEK_SET);

				if (BMPFileHeader == NULL)
					BMPFileHeader = malloc(54);

				fread(BMPFileHeader, 54, 1, f_bitmap);

				int *ptr = ((int *)(((unsigned long long)BMPFileHeader) + 18)); // !!! !!! !!!

				bitmapWidth = ptr[0]; // !!! !!! !!!
				bitmapHeight = ptr[1]; // !!! !!! !!!

				scanLineSize = (((bitmapWidth * 3) + 3) & -4);
				bitmap = (int *)malloc(sizeof(int) * bitmapWidth * bitmapHeight * numberOfPoses);
				bitmap_tmp = malloc(scanLineSize * bitmapHeight);

				double_tan_half_fov_x = 2.0f * tanf(FOV * 0.5f);
				double_tan_half_fov_y = 2.0f * tanf(FOV * 0.5f);
			}
			fseek(f_bitmap, 54, SEEK_SET);
			fread(bitmap_tmp, scanLineSize * bitmapHeight, 1, f_bitmap);
			for (int i = 0; i < bitmapHeight; ++i) {
				for (int j = 0; j < bitmapWidth; ++j) {
					unsigned char B = ((char *)bitmap_tmp)[((bitmapHeight - 1 - i) * scanLineSize) + (j * 3) + 0];
					unsigned char G = ((char *)bitmap_tmp)[((bitmapHeight - 1 - i) * scanLineSize) + (j * 3) + 1];
					unsigned char R = ((char *)bitmap_tmp)[((bitmapHeight - 1 - i) * scanLineSize) + (j * 3) + 2];
					bitmap[(poseNum * (bitmapWidth * bitmapHeight)) + (i * bitmapWidth) + j] = (R << 16) + (G << 8) + B;
				}
			}

			fclose(f_bitmap);
		}

		str = strstr(buf, "transform_matrix");
		if (str != NULL) {
			fgets(buf, 256, f);

			fgets(buf, 256, f);
			sscanf_s(buf, "%f", &poses[poseNum].Rx);
			fgets(buf, 256, f);
			sscanf_s(buf, "%f", &poses[poseNum].Dx);
			fgets(buf, 256, f);
			sscanf_s(buf, "%f", &poses[poseNum].Fx);
			fgets(buf, 256, f);
			sscanf_s(buf, "%f", &poses[poseNum].Ox);

			fgets(buf, 256, f);
			fgets(buf, 256, f);

			fgets(buf, 256, f);
			sscanf_s(buf, "%f", &poses[poseNum].Ry);
			fgets(buf, 256, f);
			sscanf_s(buf, "%f", &poses[poseNum].Dy);
			fgets(buf, 256, f);
			sscanf_s(buf, "%f", &poses[poseNum].Fy);
			fgets(buf, 256, f);
			sscanf_s(buf, "%f", &poses[poseNum].Oy);

			fgets(buf, 256, f);
			fgets(buf, 256, f);

			fgets(buf, 256, f);
			sscanf_s(buf, "%f", &poses[poseNum].Rz);
			fgets(buf, 256, f);
			sscanf_s(buf, "%f", &poses[poseNum].Dz);
			fgets(buf, 256, f);
			sscanf_s(buf, "%f", &poses[poseNum].Fz);
			fgets(buf, 256, f);
			sscanf_s(buf, "%f", &poses[poseNum].Oz);

			poses[poseNum].Dx = -poses[poseNum].Dx;
			poses[poseNum].Dy = -poses[poseNum].Dy;
			poses[poseNum].Dz = -poses[poseNum].Dz;

			poses[poseNum].Fx = -poses[poseNum].Fx;
			poses[poseNum].Fy = -poses[poseNum].Fy;
			poses[poseNum].Fz = -poses[poseNum].Fz;

			++poseNum;
		}
	}

	free(bitmap_tmp);
	fclose(f);
}

// *************************************************************************************************

void LoadConfigFile(const char* fName, SOptiXRenderConfig &config) {
	FILE *f;

	char buf[256];
	char tmp[256];
	int pos;

	// *********************************************************************************************

	fopen_s(&f, fName, "rt");

	// *********************************************************************************************

	// ESSENTIAL
	fgets(buf, 256, f);
	fgets(buf, 256, f);
	fgets(buf, 256, f);

	// *********************************************************************************************

	// Data directory path
	fgets(buf, 256, f);
	pos = (strstr(buf, "<") - buf);
	--pos;
	while (buf[pos] == ' ') --pos;
	config.data_path = (char *)malloc(sizeof(char) * (pos + 2));
	strncpy_s(config.data_path, 256, buf, pos + 1);
	config.data_path[pos + 1] = 0;

	// Pretrained 3DGS model path
	fgets(buf, 256, f);
	pos = (strstr(buf, "<") - buf);
	--pos;
	while (buf[pos] == ' ') --pos;
	config.pretrained_model_path = (char *)malloc(sizeof(char) * (pos + 2));
	strncpy_s(config.pretrained_model_path, 256, buf, pos + 1);
	config.pretrained_model_path[pos + 1] = 0;

	// Data format
	fgets(buf, 256, f);
	pos = (strstr(buf, "<") - buf);
	--pos;
	while (buf[pos] == ' ') --pos;
	config.data_format = (char *)malloc(sizeof(char) * (pos + 2));
	strncpy_s(config.data_format, 256, buf, pos + 1);
	config.data_format[pos + 1] = 0;

	// *********************************************************************************************

	fgets(buf, 256, f);

	// *********************************************************************************************

	// Start iteration
	fgets(buf, 256, f);
	sscanf_s(buf, "%d", &config.start_epoch, 256);

	// End iteration
	fgets(buf, 256, f);
	sscanf_s(buf, "%d", &config.end_epoch, 256);

	// *********************************************************************************************

	fgets(buf, 256, f);

	// *********************************************************************************************

	// Spherical harmonics degree
	fgets(buf, 256, f);
	sscanf_s(buf, "%d", &config.SH_degree, 256);

	// *********************************************************************************************

	fgets(buf, 256, f);

	// *********************************************************************************************

	// Background color R component
	fgets(buf, 256, f);
	sscanf_s(buf, "%f", &config.bg_color_R, 256);

	// Background color G component
	fgets(buf, 256, f);
	sscanf_s(buf, "%f", &config.bg_color_G, 256);

	// Background color B component
	fgets(buf, 256, f);
	sscanf_s(buf, "%f", &config.bg_color_B, 256);

	// *********************************************************************************************

	fgets(buf, 256, f);

	// *********************************************************************************************

	// PERFORMANCE
	fgets(buf, 256, f);
	fgets(buf, 256, f);
	fgets(buf, 256, f);

	// *********************************************************************************************

	// Number of sides of the polygon approximating each of the Gaussians
	fgets(buf, 256, f);
	sscanf_s(buf, "%d", &config.number_of_sides, 256);

	// Epsilon value set as the Gaussians scale parameter on the OX axis
	fgets(buf, 256, f);
	sscanf_s(buf, "%f", &config.epsilon, 256);

	// *********************************************************************************************

	fgets(buf, 256, f);

	// *********************************************************************************************

	// Chi-square squared radius for the Gaussian ellipsoid of confidence
	fgets(buf, 256, f);
	sscanf_s(buf, "%f", &config.chi_square_squared_radius, 256);

	// Ray termination T threshold
	fgets(buf, 256, f);
	sscanf_s(buf, "%f", &config.ray_termination_T_threshold, 256);

	// Ray termination T threshold for inference
	fgets(buf, 256, f);
	sscanf_s(buf, "%f", &config.ray_termination_T_threshold_inference, 256);

	// Opacity threshold for Gauss removal
	fgets(buf, 256, f);
	sscanf_s(buf, "%f", &config.alpha_threshold_for_Gauss_removal, 256);

	// Maximum number of Gaussians per ray
	fgets(buf, 256, f);
	sscanf_s(buf, "%d", &config.max_Gaussians_per_ray, 256);

	// *********************************************************************************************

	fgets(buf, 256, f);

	// *********************************************************************************************

	// Densification frequency
	fgets(buf, 256, f);
	sscanf_s(buf, "%d", &config.densification_frequency, 256);

	// Densification start iteration
	fgets(buf, 256, f);
	sscanf_s(buf, "%d", &config.densification_start_epoch, 256);

	// Densification end iteration
	fgets(buf, 256, f);
	sscanf_s(buf, "%d", &config.densification_end_epoch, 256);

	// Maximum number of Gaussians per model threshold
	fgets(buf, 256, f);
	sscanf_s(buf, "%d", &config.max_Gaussians_per_model, 256);

	// *********************************************************************************************

	fgets(buf, 256, f);

	// *********************************************************************************************

	// mu gradient norm threshold for densification
	fgets(buf, 256, f);
	sscanf_s(buf, "%f", &config.mu_grad_norm_threshold_for_densification, 256);

	// s gradient norm threshold for Gaussian split streategy
	fgets(buf, 256, f);
	sscanf_s(buf, "%f", &config.s_norm_threshold_for_split_strategy, 256);

	// Split ratio for Gaussian split strategy
	fgets(buf, 256, f);
	sscanf_s(buf, "%f", &config.split_ratio, 256);

	// *********************************************************************************************

	fgets(buf, 256, f);

	// *********************************************************************************************

	// Minimum s norm threshold for Gauss removal
	fgets(buf, 256, f);
	sscanf_s(buf, "%f", &config.min_s_norm_threshold_for_Gauss_removal, 256);

	// Minimum s coefficients clipping threshold
	fgets(buf, 256, f);
	sscanf_s(buf, "%f", &config.min_s_coefficients_clipping_threshold, 256);

	// Maximum s norm threshold for Gauss removal
	fgets(buf, 256, f);
	sscanf_s(buf, "%f", &config.max_s_norm_threshold_for_Gauss_removal, 256);

	// *********************************************************************************************

	fgets(buf, 256, f);

	// *********************************************************************************************

	// Lambda parameter for the cost function
	fgets(buf, 256, f);
	sscanf_s(buf, "%f", &config.lambda, 256);

	// *********************************************************************************************

	fgets(buf, 256, f);

	// *********************************************************************************************

	// OUTPUT
	fgets(buf, 256, f);
	fgets(buf, 256, f);
	fgets(buf, 256, f);

	// *********************************************************************************************

	// Model parameters saving frequency
	fgets(buf, 256, f);
	sscanf_s(buf, "%d", &config.saving_frequency, 256);

	// Model parameters saving iteration modulo frequency
	fgets(buf, 256, f);
	sscanf_s(buf, "%d", &config.saving_iter, 256);

	// *********************************************************************************************

	fgets(buf, 256, f);

	// *********************************************************************************************

	// PLY file saving frequency
	fgets(buf, 256, f);
	sscanf_s(buf, "%d", &config.saving_frequency_PLY, 256);

	// PLY file saving iteration modulo frequency
	fgets(buf, 256, f);
	sscanf_s(buf, "%d", &config.saving_iter_PLY, 256);

	// *********************************************************************************************

	fgets(buf, 256, f);

	// *********************************************************************************************

	// Model evaluation on startup for train data set
	fgets(buf, 256, f);
	pos = (strstr(buf, "<") - buf);
	--pos;
	while (buf[pos] == ' ') --pos;
	buf[pos + 1] = 0;
	config.evaluation_on_startup_train = (strcmp(buf, "Yes") == 0);

	// Model evaluation frequency for train data set
	fgets(buf, 256, f);
	sscanf_s(buf, "%d", &config.evaluation_frequency_train, 256);

	// Model evaluation iteration for train data set modulo frequency
	fgets(buf, 256, f);
	sscanf_s(buf, "%d", &config.evaluation_iter_train, 256);

	// Model evaluation on finish for train data set
	fgets(buf, 256, f);
	pos = (strstr(buf, "<") - buf);
	--pos;
	while (buf[pos] == ' ') --pos;
	buf[pos + 1] = 0;
	config.evaluation_on_finish_train = (strcmp(buf, "Yes") == 0);

	// *********************************************************************************************

	fgets(buf, 256, f);

	// *********************************************************************************************

	// Model evaluation on startup for test data set
	fgets(buf, 256, f);
	pos = (strstr(buf, "<") - buf);
	--pos;
	while (buf[pos] == ' ') --pos;
	buf[pos + 1] = 0;
	config.evaluation_on_startup_test = (strcmp(buf, "Yes") == 0);

	// Model evaluation frequency for test data set
	fgets(buf, 256, f);
	sscanf_s(buf, "%d", &config.evaluation_frequency_test, 256);

	// Model evaluation iteration for test data set modulo frequency
	fgets(buf, 256, f);
	sscanf_s(buf, "%d", &config.evaluation_iter_test, 256);

	// Model evaluation on finish for test data set
	fgets(buf, 256, f);
	pos = (strstr(buf, "<") - buf);
	--pos;
	while (buf[pos] == ' ') --pos;
	buf[pos + 1] = 0;
	config.evaluation_on_finish_test = (strcmp(buf, "Yes") == 0);

	// *********************************************************************************************

	fgets(buf, 256, f);

	// *********************************************************************************************

	// Model visualization on startup for train data set
	fgets(buf, 256, f);
	pos = (strstr(buf, "<") - buf);
	--pos;
	while (buf[pos] == ' ') --pos;
	buf[pos + 1] = 0;
	config.visualization_on_startup_train = (strcmp(buf, "Yes") == 0);

	// Model visualization frequency for train data set
	fgets(buf, 256, f);
	sscanf_s(buf, "%d", &config.visualization_frequency_train, 256);

	// Model visualization iteration for train data set modulo frequency
	fgets(buf, 256, f);
	sscanf_s(buf, "%d", &config.visualization_iter_train, 256);

	// Model visualization on finish for train data set
	fgets(buf, 256, f);
	pos = (strstr(buf, "<") - buf);
	--pos;
	while (buf[pos] == ' ') --pos;
	buf[pos + 1] = 0;
	config.visualization_on_finish_train = (strcmp(buf, "Yes") == 0);

	// *********************************************************************************************

	fgets(buf, 256, f);

	// *********************************************************************************************

	// Model visualization on startup for test data set
	fgets(buf, 256, f);
	pos = (strstr(buf, "<") - buf);
	--pos;
	while (buf[pos] == ' ') --pos;
	buf[pos + 1] = 0;
	config.visualization_on_startup_test = (strcmp(buf, "Yes") == 0);

	// Model visualization frequency for test data set
	fgets(buf, 256, f);
	sscanf_s(buf, "%d", &config.visualization_frequency_test, 256);

	// Model visualization iteration for test data set modulo frequency
	fgets(buf, 256, f);
	sscanf_s(buf, "%d", &config.visualization_iter_test, 256);

	// Model visualization on finish for test data set
	fgets(buf, 256, f);
	pos = (strstr(buf, "<") - buf);
	--pos;
	while (buf[pos] == ' ') --pos;
	buf[pos + 1] = 0;
	config.visualization_on_finish_test = (strcmp(buf, "Yes") == 0);

	// *********************************************************************************************

	fgets(buf, 256, f);

	// *********************************************************************************************
	
	// LEARNING RATES
	fgets(buf, 256, f);
	fgets(buf, 256, f);
	fgets(buf, 256, f);

	// *********************************************************************************************

	// Learning rate for Gaussian spherical harmonics of degree 0
	fgets(buf, 256, f);
	sscanf_s(buf, "%f", &config.lr_SH0, 256);

	// Exponential decay coefficient for learning rate for Gaussian Gaussian spherical harmonics of degree 0
	fgets(buf, 256, f);
	sscanf_s(buf, "%f", &config.lr_SH0_exponential_decay_coefficient, 256);

	// Final value of learning rate for Gaussian spherical harmonics of degree 0
	fgets(buf, 256, f);
	sscanf_s(buf, "%f", &config.lr_SH0_final, 256);

	// *********************************************************************************************

	fgets(buf, 256, f);

	// *********************************************************************************************

	// Spherical harmonics of degree 1 activation iteration
	fgets(buf, 256, f);
	sscanf_s(buf, "%d", &config.SH1_activation_iter, 256);

	// Learning rate for Gaussian spherical harmonics of degree 1
	fgets(buf, 256, f);
	sscanf_s(buf, "%f", &config.lr_SH1, 256);

	// Exponential decay coefficient for learning rate for Gaussian Gaussian spherical harmonics of degree 1
	fgets(buf, 256, f);
	sscanf_s(buf, "%f", &config.lr_SH1_exponential_decay_coefficient, 256);

	// Final value of learning rate for Gaussian spherical harmonics of degree 1
	fgets(buf, 256, f);
	sscanf_s(buf, "%f", &config.lr_SH1_final, 256);

	// *********************************************************************************************

	fgets(buf, 256, f);

	// *********************************************************************************************

	// Spherical harmonics of degree 2 activation iteration
	fgets(buf, 256, f);
	sscanf_s(buf, "%d", &config.SH2_activation_iter, 256);

	// Learning rate for Gaussian spherical harmonics of degree 2
	fgets(buf, 256, f);
	sscanf_s(buf, "%f", &config.lr_SH2, 256);

	// Exponential decay coefficient for learning rate for Gaussian Gaussian spherical harmonics of degree 2
	fgets(buf, 256, f);
	sscanf_s(buf, "%f", &config.lr_SH2_exponential_decay_coefficient, 256);

	// Final value of learning rate for Gaussian spherical harmonics of degree 2
	fgets(buf, 256, f);
	sscanf_s(buf, "%f", &config.lr_SH2_final, 256);

	// *********************************************************************************************

	fgets(buf, 256, f);

	// *********************************************************************************************

	// Spherical harmonics of degree 3 activation iteration
	fgets(buf, 256, f);
	sscanf_s(buf, "%d", &config.SH3_activation_iter, 256);

	// Learning rate for Gaussian spherical harmonics of degree 3
	fgets(buf, 256, f);
	sscanf_s(buf, "%f", &config.lr_SH3, 256);

	// Exponential decay coefficient for learning rate for Gaussian Gaussian spherical harmonics of degree 3
	fgets(buf, 256, f);
	sscanf_s(buf, "%f", &config.lr_SH3_exponential_decay_coefficient, 256);

	// Final value of learning rate for Gaussian spherical harmonics of degree 3
	fgets(buf, 256, f);
	sscanf_s(buf, "%f", &config.lr_SH3_final, 256);

	// *********************************************************************************************

	fgets(buf, 256, f);

	// *********************************************************************************************

	// Spherical harmonics of degree 4 activation iteration
	fgets(buf, 256, f);
	sscanf_s(buf, "%d", &config.SH4_activation_iter, 256);

	// Learning rate for Gaussian spherical harmonics of degree 4
	fgets(buf, 256, f);
	sscanf_s(buf, "%f", &config.lr_SH4, 256);

	// Exponential decay coefficient for learning rate for Gaussian Gaussian spherical harmonics of degree 4
	fgets(buf, 256, f);
	sscanf_s(buf, "%f", &config.lr_SH4_exponential_decay_coefficient, 256);

	// Final value of learning rate for Gaussian spherical harmonics of degree 4
	fgets(buf, 256, f);
	sscanf_s(buf, "%f", &config.lr_SH4_final, 256);

	// *********************************************************************************************

	fgets(buf, 256, f);

	// *********************************************************************************************

	// Learning rate for Gaussian opacities
	fgets(buf, 256, f);
	sscanf_s(buf, "%f", &config.lr_alpha, 256);

	// Exponential decay coefficient for learning rate for Gaussian opacities
	fgets(buf, 256, f);
	sscanf_s(buf, "%f", &config.lr_alpha_exponential_decay_coefficient, 256);

	// Final value of learning rate for Gaussian opacities
	fgets(buf, 256, f);
	sscanf_s(buf, "%f", &config.lr_alpha_final, 256);

	// *********************************************************************************************

	fgets(buf, 256, f);

	// *********************************************************************************************

	// Learning rate for Gaussian means
	fgets(buf, 256, f);
	sscanf_s(buf, "%f", &config.lr_m, 256);

	// Exponential decay coefficient for learning rate for Gaussian means
	fgets(buf, 256, f);
	sscanf_s(buf, "%f", &config.lr_m_exponential_decay_coefficient, 256);

	// Final value of learning rate for Gaussian means
	fgets(buf, 256, f);
	sscanf_s(buf, "%f", &config.lr_m_final, 256);

	// *********************************************************************************************

	fgets(buf, 256, f);

	// *********************************************************************************************

	// Learning rate for Gaussian scales
	fgets(buf, 256, f);
	sscanf_s(buf, "%f", &config.lr_s, 256);

	// Exponential decay coefficient for learning rate for Gaussian scales
	fgets(buf, 256, f);
	sscanf_s(buf, "%f", &config.lr_s_exponential_decay_coefficient, 256);

	// Final value of learning rate for Gaussian scales
	fgets(buf, 256, f);
	sscanf_s(buf, "%f", &config.lr_s_final, 256);

	// *********************************************************************************************

	fgets(buf, 256, f);

	// *********************************************************************************************

	// Learning rate for Gaussian quaternions
	fgets(buf, 256, f);
	sscanf_s(buf, "%f", &config.lr_q, 256);

	// Exponential decay coefficient for learning rate for Gaussian quaternions
	fgets(buf, 256, f);
	sscanf_s(buf, "%f", &config.lr_q_exponential_decay_coefficient, 256);

	// Final value of learning rate for Gaussian quaternions
	fgets(buf, 256, f);
	sscanf_s(buf, "%f", &config.lr_q_final, 256);

	// *********************************************************************************************

	fgets(buf, 256, f);

	// *********************************************************************************************

	// MEMORY MANAGEMENT
	fgets(buf, 256, f);
	fgets(buf, 256, f);
	fgets(buf, 256, f);

	// *********************************************************************************************

	// Temporary arrays growth factor
	fgets(buf, 256, f);
	sscanf_s(buf, "%lf", &config.tmp_arrays_growth_factor, 256);

	// *********************************************************************************************

	fclose(f);
}

// *************************************************************************************************

void LoadSceneAndCameraCOLMAP(
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
	char **&img_names,
	char **&img_names_test,
	float &double_tan_half_fov_x, float &double_tan_half_fov_y
) {
	FILE *f;
	
	char filePath[256];
	strcpy_s(filePath, dataPath);
	strcat_s(filePath, "/");
	strcat_s(filePath, jsonFileName);

	fopen_s(&f, filePath, "rb");
	fseek(f, 0, SEEK_END);
	int fSize = ftell(f);
	fclose(f);

	char *buf = (char *)malloc(sizeof(char) * (fSize + 1));

	fopen_s(&f, filePath, "rt");
	fread(buf, fSize, 1, f);
	buf[fSize] = 0;
	fclose(f);

	int numberOfPoses = 0;
	char *tmp1 = buf;
	char *tmp2 = strstr(tmp1, "\"id\"");
	while (tmp2 != NULL) {
		++numberOfPoses;

		tmp1 = tmp2 + 2;
		tmp2 = strstr(tmp1, "\"id\"");
	}

	numberOfPoses_test = (numberOfPoses + 7) >> 3;
	numberOfPoses_train = numberOfPoses - numberOfPoses_test;
	poses_train = (SCamera*)malloc(sizeof(SCamera) * numberOfPoses_train);
	poses_test = (SCamera*)malloc(sizeof(SCamera) * numberOfPoses_test);
	
	img_names = (char **)malloc(sizeof(char *) * numberOfPoses_train); // !!! !!! !!!
	img_names_test = (char **)malloc(sizeof(char *) * numberOfPoses_test); // !!! !!! !!!

	void *bitmap_tmp = NULL;

	tmp1 = buf;
	for (int poseNum = 0; poseNum < numberOfPoses; ++poseNum) {
		tmp2 = strstr(tmp1, "\"img_name\"");		
		char tmp3[256];
		sscanf_s(tmp2, "\"img_name\": \"%s", tmp3, 256);
		char *next;
		char *fName = strtok_s(tmp3, "\"", &next);
		tmp1 = tmp2 + strlen("\"img_name\":");

		// *** *** *** *** ***

		FILE *f_bitmap;
		strcpy_s(filePath, dataPath);
		strcat_s(filePath, "/images/");
		strcat_s(filePath, fName);
		strcat_s(filePath, ".bmp");

		fopen_s(&f_bitmap, filePath, "rb");

		if (poseNum == 0) {
			fseek(f_bitmap, 0, SEEK_END);
			bitmapSize = ftell(f_bitmap);
			fseek(f_bitmap, 0, SEEK_SET);

			if (BMPFileHeader == NULL)
				BMPFileHeader = malloc(54);

			fread(BMPFileHeader, 54, 1, f_bitmap);

			int *ptr = ((int *)(((unsigned long long)BMPFileHeader) + 18)); // !!! !!! !!!

			bitmapWidth = ptr[0]; // !!! !!! !!!
			bitmapHeight = ptr[1]; // !!! !!! !!!

			scanLineSize = (((bitmapWidth * 3) + 3) & -4);
			bitmap_train = (int *)malloc(sizeof(int) * bitmapWidth * bitmapHeight * numberOfPoses_train);
			bitmap_test = (int *)malloc(sizeof(int) * bitmapWidth * bitmapHeight * numberOfPoses_test);
			bitmap_tmp = malloc(scanLineSize * bitmapHeight);

			numberOfPoses_train = 0; // !!! !!! !!!
			numberOfPoses_test = 0; // !!! !!! !!!
		}

		if ((poseNum & 7) != 0) {
			img_names[numberOfPoses_train] = (char *)malloc(sizeof(char) * (strlen(fName) + 1));
			strcpy_s(img_names[numberOfPoses_train], strlen(fName) + 1, fName);
		} else {
			img_names_test[numberOfPoses_test] = (char *)malloc(sizeof(char) * (strlen(fName) + 1));
			strcpy_s(img_names_test[numberOfPoses_test], strlen(fName) + 1, fName);
		}

		fseek(f_bitmap, 54, SEEK_SET);
		fread(bitmap_tmp, scanLineSize * bitmapHeight, 1, f_bitmap);
		for (int i = 0; i < bitmapHeight; ++i) {
			for (int j = 0; j < bitmapWidth; ++j) {
				unsigned char B = ((char *)bitmap_tmp)[((bitmapHeight - 1 - i) * scanLineSize) + (j * 3) + 0];
				unsigned char G = ((char *)bitmap_tmp)[((bitmapHeight - 1 - i) * scanLineSize) + (j * 3) + 1];
				unsigned char R = ((char *)bitmap_tmp)[((bitmapHeight - 1 - i) * scanLineSize) + (j * 3) + 2];
				if ((poseNum & 7) != 0)
					bitmap_train[(numberOfPoses_train * (bitmapWidth * bitmapHeight)) + (i * bitmapWidth) + j] = (R << 16) + (G << 8) + B;
				else
					bitmap_test[(numberOfPoses_test * (bitmapWidth * bitmapHeight)) + (i * bitmapWidth) + j] = (R << 16) + (G << 8) + B;
			}
		}

		fclose(f_bitmap);

		// *** *** *** *** ***
		
		int width;
		tmp2 = strstr(tmp1, "\"width\"");
		sscanf_s(tmp2, "\"width\": %d", &width, 256);
		tmp1 = tmp2 + strlen("\"width\":");

		int height;
		tmp2 = strstr(tmp1, "\"height\"");
		sscanf_s(tmp2, "\"height\": %d", &height, 256);
		tmp1 = tmp2 + strlen("\"height\":");
		
		tmp2 = strstr(tmp1, "\"position\"");
		if ((poseNum & 7) != 0)
			sscanf_s(
				tmp2,
				"\"position\": [%f, %f, %f]",
				&poses_train[numberOfPoses_train].Ox, &poses_train[numberOfPoses_train].Oy, &poses_train[numberOfPoses_train].Oz,
				256
			);
		else
			sscanf_s(
				tmp2,
				"\"position\": [%f, %f, %f]",
				&poses_test[numberOfPoses_test].Ox, &poses_test[numberOfPoses_test].Oy, &poses_test[numberOfPoses_test].Oz,
				256
			);
		tmp1 = tmp2 + strlen("\"position\":");

		tmp2 = strstr(tmp1, "\"rotation\"");
		if ((poseNum & 7) != 0)
			sscanf_s(
				tmp2,
				"\"rotation\": [[%f, %f, %f], [%f, %f, %f], [%f, %f, %f]]",
				&poses_train[numberOfPoses_train].Rx, &poses_train[numberOfPoses_train].Dx, &poses_train[numberOfPoses_train].Fx,
				&poses_train[numberOfPoses_train].Ry, &poses_train[numberOfPoses_train].Dy, &poses_train[numberOfPoses_train].Fy,
				&poses_train[numberOfPoses_train].Rz, &poses_train[numberOfPoses_train].Dz, &poses_train[numberOfPoses_train].Fz,
				256
			);
		else
			sscanf_s(
				tmp2,
				"\"rotation\": [[%f, %f, %f], [%f, %f, %f], [%f, %f, %f]]",
				&poses_test[numberOfPoses_test].Rx, &poses_test[numberOfPoses_test].Dx, &poses_test[numberOfPoses_test].Fx,
				&poses_test[numberOfPoses_test].Ry, &poses_test[numberOfPoses_test].Dy, &poses_test[numberOfPoses_test].Fy,
				&poses_test[numberOfPoses_test].Rz, &poses_test[numberOfPoses_test].Dz, &poses_test[numberOfPoses_test].Fz,
				256
			);
		tmp1 = tmp2 + strlen("\"rotation\":");

		float fy;
		tmp2 = strstr(tmp1, "\"fy\"");
		sscanf_s(tmp2, "\"fy\": %f", &fy, 256);
		double_tan_half_fov_y = height / fy;
		tmp1 = tmp2 + strlen("\"fy\":");

		float fx;
		tmp2 = strstr(tmp1, "\"fx\"");
		sscanf_s(tmp2, "\"fx\": %f", &fx, 256);
		double_tan_half_fov_x = width / fx;
		tmp1 = tmp2 + strlen("\"fx\":");

		if ((poseNum & 7) != 0)
			++numberOfPoses_train;
		else
			++numberOfPoses_test;
	}
}

// *************************************************************************************************

template<int SH_degree>
void LoadPLYFile(const char* fName, SGaussianComponent<SH_degree> **GC_ptr) {
	FILE *f;

	fopen_s(&f, config.pretrained_model_path, "rb");

	char buffer[256];
	int numberOfProperties = 0;
	do {
		fgets(buffer, 256, f);

		char *str = strstr(buffer, "element vertex");
		if (str != NULL)
			sscanf_s(str, "element vertex %d", &numberOfGaussians);

		str = strstr(buffer, "property");
		if (str != NULL) ++numberOfProperties;

	} while (strstr(buffer, "end_header") == NULL);

	int SH_degree_source = ((int)sqrt((numberOfProperties - 13) / 3)) - 1;

	// *** *** ***

	float *pfs = (float *)malloc(sizeof(float) * numberOfProperties);

	*GC_ptr = (SGaussianComponent<SH_degree> *)malloc(sizeof(SGaussianComponent<SH_degree>) * numberOfGaussians);
	SGaussianComponent<SH_degree> *GC = *GC_ptr;
	for (int i = 0; i < numberOfGaussians; ++i) {
		fread(pfs, sizeof(float) * numberOfProperties, 1, f);

		GC[i].mX = pfs[0];
		GC[i].mY = pfs[1];
		GC[i].mZ = pfs[2];

		// !!! !!! !!!
		GC[i].sX = log(config.epsilon);
		// !!! !!! !!!
		
		GC[i].sY = pfs[numberOfProperties - 6];
		GC[i].sZ = pfs[numberOfProperties - 5];

		// *** *** *** *** ***

		double qr = pfs[numberOfProperties - 4];
		double qi = pfs[numberOfProperties - 3];
		double qj = pfs[numberOfProperties - 2];
		double qk = pfs[numberOfProperties - 1];
		double invNorm = 1.0 / sqrt((qr * qr) + (qi * qi) + (qj * qj) + (qk * qk));
		qr = qr * invNorm;
		qi = qi * invNorm;
		qj = qj * invNorm;
		qk = qk * invNorm;

		GC[i].qr = qr;
		GC[i].qi = qi;
		GC[i].qj = qj;
		GC[i].qk = qk;

		// *** *** *** *** ***

		GC[i].R = pfs[6];
		GC[i].G = pfs[7];
		GC[i].B = pfs[8];

		if constexpr (SH_degree > 0) {
			int ind_src = 0;
			for (int j = 0; j < 3; ++j) {
				if (SH_degree <= SH_degree_source) {
					for (int k = 1; k <= SH_degree_source; ++k) {
						for (int l = 0; l < (2 * k) + 1; ++l) {
							if (k <= SH_degree) {
								int ind_dest = ((((k * k) - 1) + l) * 3) + j;
								GC[i].RGB_SH_higher_order[ind_dest] = pfs[8 + ind_src];
							}
													
							++ind_src;
						}
					}
				} else {
					for (int k = 1; k <= SH_degree; ++k) {
						for (int l = 0; l < (2 * k) + 1; ++l) {
							int ind_dest = ((((k * k) - 1) + l) * 3) + j;

							if (k <= SH_degree_source) {
								GC[i].RGB_SH_higher_order[ind_dest] = pfs[8 + ind_src];
								++ind_src;
							} else
								GC[i].RGB_SH_higher_order[ind_dest] = 0.0f; // !!! !!! !!!
						}
					}
				}
			}
		}
		
		GC[i].alpha = pfs[numberOfProperties - 7];
	}

	fclose(f);
}

// *************************************************************************************************

void PrepareScene() {
	LoadConfigFile("config.txt", config); // !!! !!! !!!
	ray_termination_T_threshold_training = config.ray_termination_T_threshold;

	// *** *** *** *** ***

	if (strcmp(config.data_format, "colmap") == 0)
		LoadSceneAndCameraCOLMAP(
			config.data_path,
			"cameras.json",
			NUMBER_OF_POSES,
			NUMBER_OF_POSES_TEST,
			poses,
			poses_test,
			BMPFileHeader,
			bitmap_ref,
			bitmap_ref_test,
			bitmapWidth, bitmapHeight,
			img_names,
			img_names_test,
			double_tan_half_fov_x, double_tan_half_fov_y
		); // !!! !!! !!!
	else {
		LoadSceneAndCamera(
			config.data_path,
			"transforms_train.json",
			NUMBER_OF_POSES,
			poses,
			BMPFileHeader,
			bitmap_ref,
			bitmapWidth, bitmapHeight,
			img_names
		);
		LoadSceneAndCamera(
			config.data_path,
			"transforms_test.json",
			NUMBER_OF_POSES_TEST,
			poses_test,
			BMPFileHeader,
			bitmap_ref_test,
			bitmapWidth, bitmapHeight,
			img_names_test
		);
	}

	// *** *** ***
	
	if (config.start_epoch == 0) {
		CreateDirectory(L"output", NULL);

		// *** *** *** *** ***

		HANDLE hFindFile;
		WIN32_FIND_DATAA findFileData;

		next_available_dir_id = 0; // !!! !!! !!!

		hFindFile = FindFirstFileA("output\\*.*", &findFileData);
		do {
			if (
				(strcmp(findFileData.cFileName, ".") != 0) &&
				(strcmp(findFileData.cFileName, "..") != 0)
			) {
				int tmp;

				sscanf_s(findFileData.cFileName, "%d", &tmp, 256);
				if (tmp > next_available_dir_id)
					next_available_dir_id = tmp;
			}
		} while (FindNextFileA(hFindFile, &findFileData) != 0);

		++next_available_dir_id; // !!! !!! !!!
		next_available_screenshot_id = 1; // !!! !!! !!!

		// *** *** *** *** ***

		wchar_t bufw[256];

		swprintf(bufw, 256, L"output\\%d", next_available_dir_id);
		CreateDirectory(bufw, NULL);

		// !!! !!! !!!
		swprintf(bufw, 256, L"output\\%d\\config.txt", next_available_dir_id);
		CopyFile(L"config.txt", bufw, true);
		// !!! !!! !!!

		swprintf(bufw, 256, L"output\\%d\\checkpoints", next_available_dir_id);
		CreateDirectory(bufw, NULL);

		swprintf(bufw, 256, L"output\\%d\\PLY files", next_available_dir_id);
		CreateDirectory(bufw, NULL);

		swprintf(bufw, 256, L"output\\%d\\renders", next_available_dir_id);
		CreateDirectory(bufw, NULL);

		swprintf(bufw, 256, L"output\\%d\\renders\\train", next_available_dir_id);
		CreateDirectory(bufw, NULL);

		swprintf(bufw, 256, L"output\\%d\\renders\\test", next_available_dir_id);
		CreateDirectory(bufw, NULL);

		swprintf(bufw, 256, L"output\\%d\\screenshots", next_available_dir_id);
		CreateDirectory(bufw, NULL);

		swprintf(bufw, 256, L"output\\%d\\stats", next_available_dir_id);
		CreateDirectory(bufw, NULL);

		// *** *** *** *** ***

		// LOAD PRETRAINED MODEL FROM *.PLY FILE
		FILE *f;

		if      (config.SH_degree == 0) LoadPLYFile<0>(config.pretrained_model_path, (SGaussianComponent<0> **)&GC);
		else if (config.SH_degree == 1) LoadPLYFile<1>(config.pretrained_model_path, (SGaussianComponent<1> **)&GC);
		else if (config.SH_degree == 2) LoadPLYFile<2>(config.pretrained_model_path, (SGaussianComponent<2> **)&GC);
		else if (config.SH_degree == 3) LoadPLYFile<3>(config.pretrained_model_path, (SGaussianComponent<3> **)&GC);
		else if (config.SH_degree == 4) LoadPLYFile<4>(config.pretrained_model_path, (SGaussianComponent<4> **)&GC);
	} else {
		sscanf_s(config.pretrained_model_path, "%d", &next_available_dir_id, 256);

		// *** *** *** *** ***

		char fPath[256];

		// *** *** *** *** ***

		sprintf_s(fPath, "output\\%d\\screenshots\\*.*", next_available_dir_id);

		// *** *** *** *** ***

		HANDLE hFindFile;
		WIN32_FIND_DATAA findFileData;

		next_available_screenshot_id = 0; // !!! !!! !!!

		hFindFile = FindFirstFileA(fPath, &findFileData);
		do {
			if (
				(strcmp(findFileData.cFileName, ".") != 0) &&
				(strcmp(findFileData.cFileName, "..") != 0)
			) {
				int tmp;

				sscanf_s(findFileData.cFileName, "%d", &tmp, 256);
				if (tmp > next_available_screenshot_id)
					next_available_screenshot_id = tmp;
			}
		} while (FindNextFileA(hFindFile, &findFileData) != 0);

		++next_available_screenshot_id; // !!! !!! !!!

		// *** *** *** *** ***

		// !!! !!! !!!
		// Load the seed state before computing the training poses indices permutation
		FILE *f;

		sprintf_s(fPath, "output\\%d\\checkpoints\\%d\\seed_iter_%d.checkpoint", next_available_dir_id, config.start_epoch, config.start_epoch);
		
		fopen_s(&f, fPath, "rb");
		fread(&seed_dword, sizeof(unsigned) * 1, 1, f);
		fclose(f);
		// !!! !!! !!!
	}

	// *** *** *** *** ***

	poses_indices = (int *)malloc(sizeof(int) * NUMBER_OF_POSES);

	// !!! !!! !!!
	seed_dword_prev = seed_dword;
	// !!! !!! !!!

	for (int i = 0; i < NUMBER_OF_POSES; i++) poses_indices[i] = i;
	for (int i = 0; i < NUMBER_OF_POSES - 1; i++) {
		int index = i + (RandomInteger() % (NUMBER_OF_POSES - i));
		if (index != i) {
			poses_indices[i] ^= poses_indices[index];
			poses_indices[index] ^= poses_indices[i];
			poses_indices[i] ^= poses_indices[index];
		}
	}

	// !!! !!! !!!
	poseNum_training = config.start_epoch % NUMBER_OF_POSES;
	// !!! !!! !!!

	poseInd_training = poses_indices[poseNum_training]; // !!! !!! !!!

	// *** *** *** *** ***

	poseNum_rendering = 0;

	Ox = poses[poseNum_rendering].Ox; Oy = poses[poseNum_rendering].Oy; Oz = poses[poseNum_rendering].Oz;
	Rx = poses[poseNum_rendering].Rx; Ry = poses[poseNum_rendering].Ry; Rz = poses[poseNum_rendering].Rz;
	Dx = poses[poseNum_rendering].Dx; Dy = poses[poseNum_rendering].Dy; Dz = poses[poseNum_rendering].Dz;
	Fx = poses[poseNum_rendering].Fx; Fy = poses[poseNum_rendering].Fy; Fz = poses[poseNum_rendering].Fz;
	
	cameraChanged = true;
}

// *************************************************************************************************

int APIENTRY wWinMain(
	_In_ HINSTANCE hInstance,
	_In_opt_ HINSTANCE hPrevInstance,
	_In_ LPWSTR    lpCmdLine,
	_In_ int       nCmdShow
) {
    UNREFERENCED_PARAMETER(hPrevInstance);
    UNREFERENCED_PARAMETER(lpCmdLine);

	// *** *** *** *** ***

	AllocConsole();

	text = (wchar_t *)malloc(sizeof(wchar_t) * 256);
	if (text == NULL) return false;

	PrepareScene();

	// *** *** *** *** ***

    LoadStringW(hInstance, IDS_APP_TITLE, szTitle, MAX_LOADSTRING);
    LoadStringW(hInstance, IDC_REDISPLATS, szWindowClass, MAX_LOADSTRING);
    MyRegisterClass(hInstance);

	// *** *** *** *** ***

	BOOL result;

	if      (config.SH_degree == 0) result = InitInstance<0>(hInstance, nCmdShow);
	else if (config.SH_degree == 1) result = InitInstance<1>(hInstance, nCmdShow);
	else if (config.SH_degree == 2) result = InitInstance<2>(hInstance, nCmdShow);
	else if (config.SH_degree == 3) result = InitInstance<3>(hInstance, nCmdShow);
	else if (config.SH_degree == 4) result = InitInstance<4>(hInstance, nCmdShow);

    if (!result) {
        return FALSE;
    }

	// *** *** *** *** ***

	// !!! !!! !!!
	RegisterHotKey(hWnd, VK_SNAPSHOT, MOD_NOREPEAT, VK_SNAPSHOT);
	// !!! !!! !!!

	// *** *** *** *** ***

    HACCEL hAccelTable = LoadAccelerators(hInstance, MAKEINTRESOURCE(IDC_REDISPLATS));
    MSG msg;

    while (GetMessage(&msg, nullptr, 0, 0)) {
        if (!TranslateAccelerator(msg.hwnd, hAccelTable, &msg)) {
            TranslateMessage(&msg);
            DispatchMessage(&msg);
        }
    }

    return (int) msg.wParam;
}

// *************************************************************************************************

ATOM MyRegisterClass(HINSTANCE hInstance) {
    WNDCLASSEXW wcex;

    wcex.cbSize = sizeof(WNDCLASSEX);

    wcex.style = CS_HREDRAW | CS_VREDRAW | CS_DBLCLKS;

	if      (config.SH_degree == 0) wcex.lpfnWndProc = WndProc<0>;
	else if (config.SH_degree == 1) wcex.lpfnWndProc = WndProc<1>;
	else if (config.SH_degree == 2) wcex.lpfnWndProc = WndProc<2>;
	else if (config.SH_degree == 3) wcex.lpfnWndProc = WndProc<3>;
	else if (config.SH_degree == 4) wcex.lpfnWndProc = WndProc<4>;
    
    wcex.cbClsExtra     = 0;
    wcex.cbWndExtra     = 0;
    wcex.hInstance      = hInstance;
    wcex.hIcon          = LoadIcon(hInstance, MAKEINTRESOURCE(IDI_REDISPLATS));
    wcex.hCursor        = LoadCursor(nullptr, IDC_ARROW);
    wcex.hbrBackground  = (HBRUSH)(COLOR_MENU+1);
    wcex.lpszMenuName   = MAKEINTRESOURCEW(IDC_REDISPLATS);
    wcex.lpszClassName  = szWindowClass;
    wcex.hIconSm        = LoadIcon(wcex.hInstance, MAKEINTRESOURCE(IDI_SMALL));

    return RegisterClassExW(&wcex);
}

// *************************************************************************************************

template<int SH_degree>
BOOL InitInstance(HINSTANCE hInstance, int nCmdShow) {
	hInst = hInstance;
   
	hWnd = CreateWindowW(szWindowClass, szTitle, WS_SYSMENU | WS_MINIMIZEBOX,
		CW_USEDEFAULT, 0, CW_USEDEFAULT, 0, nullptr, nullptr, hInstance, nullptr);

	if (!hWnd)
		return FALSE;

	HWND hwndLabel1 = CreateWindowW (
		L"static",
		L"Iteration: 0;",
		WS_CHILD | WS_VISIBLE | WS_TABSTOP | SS_CENTERIMAGE,
        8,
		8,
		384,
		24,
        hWnd,
		(HMENU)LABEL1,
        hInstance,
		NULL
	);
	if (!hwndLabel1)
		return false;
	SendMessage(hwndLabel1, WM_SETFONT, (LPARAM)GetStockObject(DEFAULT_GUI_FONT), true);

	BITMAPINFO bmi;
	memset(&bmi, 0, sizeof(BITMAPINFO));
	bmi.bmiHeader.biSize = sizeof(BITMAPINFOHEADER);
	bmi.bmiHeader.biWidth = bitmapWidth;
	bmi.bmiHeader.biHeight = -bitmapHeight; // top-down
	bmi.bmiHeader.biPlanes = 1;
	bmi.bmiHeader.biBitCount = 32;
	bmi.bmiHeader.biCompression = BI_RGB;
	hBitmap = CreateDIBSection(GetDC(hWnd), &bmi, DIB_RGB_COLORS, &bitmap, NULL, NULL);
	if (!hBitmap)
		return false;
	memset(bitmap, 0, bitmapWidth * bitmapHeight * 4);

	RECT rectP;
	GetWindowRect(hWnd, &rectP);
	RECT rectN = {0, 0, bitmapWidth, bitmapHeight};
	AdjustWindowRect(&rectN, GetWindowLong(hWnd, GWL_STYLE), true);
	MoveWindow(hWnd, rectP.left, rectP.top, rectN.right - rectN.left, rectN.bottom - rectN.top + 40, true);

	// *** *** *** *** ***

	// !!! !!! !!!
	params = malloc(sizeof(SRenderParams<SH_degree>) * 1);
	SRenderParams<SH_degree> *params = (SRenderParams<SH_degree> *)::params;
	// !!! !!! !!!

	// !!! !!! !!!
	params_OptiX = malloc(sizeof(SOptiXRenderParams<SH_degree>) * 1);
	SOptiXRenderParams<SH_degree> *params_OptiX = (SOptiXRenderParams<SH_degree> *)::params_OptiX;
	// !!! !!! !!!

	for (int i = 0; i < 1; ++i) {
		// !!! !!! !!!
		params[i].double_tan_half_fov_x = double_tan_half_fov_x;
		params[i].double_tan_half_fov_y = double_tan_half_fov_y;
		// !!! !!! !!!
		
		params[i].bitmap = bitmap;
		params[i].w = bitmapWidth; params[i].h = bitmapHeight;
		params[i].GC = (SGaussianComponent<SH_degree> *)GC;
		params[i].numberOfGaussians = numberOfGaussians;

		params[0].bitmap_ref = (unsigned *)bitmap_ref; // !!! !!! !!!
		params[0].NUMBER_OF_POSES = NUMBER_OF_POSES; // !!! !!! !!!
	}
	// !!! !!! !!!

	//**********************************************************************************************

	bool result;

	//**********************************************************************************************

	SetConfigurationOptiX(config);
	// !!! !!! !!!
	if (config.start_epoch == 0) {
		// LOAD FROM PRETRAINED MODEL
		epochNum = 0;
		params_OptiX->epoch = epochNum; // !!! !!! !!!

		// *** *** *** *** ***

		if      constexpr (SH_degree == 0) result = InitializeOptiXRendererSH0(params[0], *params_OptiX);
		else if constexpr (SH_degree == 1) result = InitializeOptiXRendererSH1(params[0], *params_OptiX);
		else if constexpr (SH_degree == 2) result = InitializeOptiXRendererSH2(params[0], *params_OptiX);
		else if constexpr (SH_degree == 3) result = InitializeOptiXRendererSH3(params[0], *params_OptiX);
		else if constexpr (SH_degree == 4) result = InitializeOptiXRendererSH4(params[0], *params_OptiX);
		
		swprintf(consoleBuffer, 256, L"Initializing OptiX renderer: %s", (result ? L"OK... .\n" : L"Failed... .\n"));
		WriteConsole(GetStdHandle(STD_OUTPUT_HANDLE), consoleBuffer, wcslen(consoleBuffer), NULL, NULL);

		if      constexpr (SH_degree == 0) result = InitializeOptiXOptimizerSH0(params[0], *params_OptiX);
		else if constexpr (SH_degree == 1) result = InitializeOptiXOptimizerSH1(params[0], *params_OptiX);
		else if constexpr (SH_degree == 2) result = InitializeOptiXOptimizerSH2(params[0], *params_OptiX);
		else if constexpr (SH_degree == 3) result = InitializeOptiXOptimizerSH3(params[0], *params_OptiX);
		else if constexpr (SH_degree == 4) result = InitializeOptiXOptimizerSH4(params[0], *params_OptiX);

		swprintf(consoleBuffer, 256, L"Initializing OptiX optimizer: %s", (result ? L"OK... .\n" : L"Failed... .\n"));
		WriteConsole(GetStdHandle(STD_OUTPUT_HANDLE), consoleBuffer, wcslen(consoleBuffer), NULL, NULL);

		float scene_extent;
		result = GetSceneExtentOptiX(scene_extent);

		swprintf(consoleBuffer, 256, L"GetSceneExtentOptiX: %s (%f)\n", (result ? L"OK... ." : L"Failed... ."), scene_extent);
		WriteConsole(GetStdHandle(STD_OUTPUT_HANDLE), consoleBuffer, wcslen(consoleBuffer), NULL, NULL);
	} else {
		epochNum = config.start_epoch; // !!! !!! !!!
		params_OptiX->epoch = epochNum; // !!! !!! !!!

		// *** *** *** *** ***

		char buf[256];

		// *** *** *** *** ***
		
		sprintf_s(buf, "output\\%d\\checkpoints\\%d", next_available_dir_id, config.start_epoch);

		// *** *** *** *** ***

		// LOAD FROM CHECKPOINT
		if      constexpr (SH_degree == 0) result = InitializeOptiXRendererSH0(params[0], *params_OptiX, buf);
		else if constexpr (SH_degree == 1) result = InitializeOptiXRendererSH1(params[0], *params_OptiX, buf);
		else if constexpr (SH_degree == 2) result = InitializeOptiXRendererSH2(params[0], *params_OptiX, buf);
		else if constexpr (SH_degree == 3) result = InitializeOptiXRendererSH3(params[0], *params_OptiX, buf);
		else if constexpr (SH_degree == 4) result = InitializeOptiXRendererSH4(params[0], *params_OptiX, buf);

		swprintf(consoleBuffer, 256, L"Initializing OptiX renderer: %s", (result ? L"OK... .\n" : L"Failed... .\n"));
		WriteConsole(GetStdHandle(STD_OUTPUT_HANDLE), consoleBuffer, wcslen(consoleBuffer), NULL, NULL);

		if      constexpr (SH_degree == 0) result = InitializeOptiXOptimizerSH0(params[0], *params_OptiX, buf);
		else if constexpr (SH_degree == 1) result = InitializeOptiXOptimizerSH1(params[0], *params_OptiX, buf);
		else if constexpr (SH_degree == 2) result = InitializeOptiXOptimizerSH2(params[0], *params_OptiX, buf);
		else if constexpr (SH_degree == 3) result = InitializeOptiXOptimizerSH3(params[0], *params_OptiX, buf);
		else if constexpr (SH_degree == 4) result = InitializeOptiXOptimizerSH4(params[0], *params_OptiX, buf);

		float scene_extent;
		result = GetSceneExtentOptiX(scene_extent);

		swprintf(consoleBuffer, 256, L"GetSceneExtentOptiX: %s (%f)\n", (result ? L"OK... ." : L"Failed... ."), scene_extent);
		WriteConsole(GetStdHandle(STD_OUTPUT_HANDLE), consoleBuffer, wcslen(consoleBuffer), NULL, NULL);
	}

	// *********************************************************************************************

	QueryPerformanceFrequency(&lpFrequency);

	//**********************************************************************************************

	char buffer[256];

	sprintf_s(buffer, "REdiSplats - Pose: %d / %d", poseNum_rendering + 1, NUMBER_OF_POSES);
	SetWindowTextA(hWnd, buffer);

	ShowWindow(hWnd, nCmdShow);
	UpdateWindow(hWnd);

	SetTimer(hWnd, TIMER1, 0, (TIMERPROC)NULL);

	return TRUE;	
}

// *************************************************************************************************

template<int SH_degree>
LRESULT CALLBACK WndProc(HWND hWnd, UINT message, WPARAM wParam, LPARAM lParam) {
	// !!! !!! !!!
	SOptiXRenderParams<SH_degree> *params_OptiX = (SOptiXRenderParams<SH_degree> *)::params_OptiX;
	// !!! !!! !!!

	// *** *** *** *** ***

	switch (message) {
		case WM_COMMAND : {
			int wmId = LOWORD(wParam);
			switch (wmId) {
				case IDM_ABOUT : {
					DialogBox(hInst, MAKEINTRESOURCE(IDD_ABOUTBOX), hWnd, About);
					break;
				}
				case IDM_EXIT : {
					DestroyWindow(hWnd);
					break;
				}
				default : {
					return DefWindowProc(hWnd, message, wParam, lParam);
				}
			}
		}
		case WM_PAINT : {
			HDC hdc;
			PAINTSTRUCT ps;

			hdc = BeginPaint(hWnd, &ps);
			HDC hdcMem = CreateCompatibleDC(hdc);
			SelectObject(hdcMem, hBitmap);
			BitBlt(hdc, 0, 40, bitmapWidth, bitmapHeight, hdcMem, 0, 0, SRCCOPY);
			DeleteDC(hdcMem);
			ReleaseDC(hWnd, hdc);
			EndPaint(hWnd, &ps);
			break;
		}
		case WM_SETCURSOR : {
			if (flyMode) {
				WORD ht = LOWORD(lParam);
				if ((ht == HTCLIENT) && cursorShown) {
					cursorShown = false;
					ShowCursor(false);
				} else if ((ht != HTCLIENT) && (!cursorShown)) {
					cursorShown = true;
					ShowCursor(true);
				}
			} else {
				if (!cursorShown) {
					cursorShown = true;
					ShowCursor(true);
				}
			}
			break;
		}
		case WM_MOUSEMOVE : {
			if (flyMode) {
				mouseX = GET_X_LPARAM(lParam); 
				mouseY = GET_Y_LPARAM(lParam);
				int dX = mouseX - (bitmapWidth / 2);
				int dY = mouseY - 40 - (bitmapHeight / 2);
				if ((dX != 0) || (dY != 0)) {
					char buffer[256];
					if (!isfinite(FPS_inference))
						sprintf_s(buffer, "REdiSplats - Custom pose");
					else
						sprintf_s(buffer, "REdiSplats - Custom pose (FPS: %.4lf)", FPS_inference);
					SetWindowTextA(hWnd, buffer);

					float Rx_; float Ry_; float Rz_;
					float Dx_; float Dy_; float Dz_;
					float Fx_; float Fy_; float Fz_;
									
						yaw += (2.0 * M_PI) * (dX / (bitmapWidth - 1.0));
					yaw = fmod(yaw, 2.0 * M_PI);
					if (yaw < 0.0) yaw += 2.0 * M_PI;
					Rx_ = (Rx * cos(yaw)) - (Fx * sin(yaw));
					Ry_ = (Ry * cos(yaw)) - (Fy * sin(yaw));
					Rz_ = (Rz * cos(yaw)) - (Fz * sin(yaw));
					Fx_ = (Rx * sin(yaw)) + (Fx * cos(yaw));
					Fy_ = (Ry * sin(yaw)) + (Fy * cos(yaw));
					Fz_ = (Rz * sin(yaw)) + (Fz * cos(yaw));

					Rx = Rx_; Ry = Ry_; Rz = Rz_;
					Fx = Fx_; Fy = Fy_; Fz = Fz_;
						pitch += (2.0 * M_PI) * (dY / (bitmapHeight - 1.0));
					if (pitch < -M_PI / 2.0f) pitch = -M_PI / 2.0f;
					if (pitch > M_PI / 2.0f) pitch = M_PI / 2.0f;
					Fx_ = (Fx * cos(-pitch)) - (Dx * sin(-pitch));
					Fy_ = (Fy * cos(-pitch)) - (Dy * sin(-pitch));
					Fz_ = (Fz * cos(-pitch)) - (Dz * sin(-pitch));
					Dx_ = (Fx * sin(-pitch)) + (Dx * cos(-pitch));
					Dy_ = (Fy * sin(-pitch)) + (Dy * cos(-pitch));
					Dz_ = (Fz * sin(-pitch)) + (Dz * cos(-pitch));
					Fx = Fx_; Fy = Fy_; Fz = Fz_;
					Dx = Dx_; Dy = Dy_; Dz = Dz_;

					pitch = 0.0f;
					yaw = 0.0f;
					
					RECT cr;
					GetClientRect(hWnd, &cr);
					ClientToScreen(hWnd, (POINT *)&cr.left);
					ClientToScreen(hWnd, (POINT *)&cr.right);
					SetCursorPos(cr.left + (bitmapWidth / 2), cr.top + 40 + (bitmapHeight / 2));

					customPose = true;
					cameraChanged = true;
				}
			}
			break;
		}
		case WM_LBUTTONDBLCLK : {
			flyMode = !flyMode;
			if (flyMode) {
				SetFocus(hWnd);

				RECT cr;
				GetClientRect(hWnd, &cr);
				ClientToScreen(hWnd, (POINT *)&cr.left);
				ClientToScreen(hWnd, (POINT *)&cr.right);
				SetCursorPos(cr.left + (bitmapWidth / 2), cr.top + 40 + (bitmapHeight / 2));
			}
			break;
		}
		case WM_KEYUP : {
			switch (wParam) {
				case 0x44 : { D_down = false; break; }
				case 0x41 : { A_down = false; break; }
				case 0x57 : { W_down = false; break; }
				case 0x53 : { S_down = false; break; }
				case 0x43 : { C_down = false; break; }
				case 0x20 : { Space_down = false; break; }
				case 0xDB : { left_square_bracket_down = false; break; }
				case 0xDD : { right_square_bracket_down = false; break; }
			}			
			break;
		}
		case WM_KEYDOWN : {
			switch (wParam) {
				case 0x44 : { D_down = true; break; }
				case 0x41 : { A_down = true; break; }
				case 0x57 : { W_down = true; break; }
				case 0x53 : { S_down = true; break; }
				case 0x43 : { C_down = true; break; }
				case 0x20 : { Space_down = true; break; }
				case 0xDB : { left_square_bracket_down = true; break; }
				case 0xDD : { right_square_bracket_down = true; break; }
			}			
			break;
		}
		case WM_HOTKEY : {
			// !!! !!! !!!
			switch (wParam)	{
				case VK_SNAPSHOT : {
					print_screen_down = true;
					cameraChanged = true;
					break;
				}
			}
			break;
			// !!! !!! !!!
		}
		case WM_TIMER : {
			switch (wParam) {
				case TIMER1 : {
					if (left_square_bracket_down && (poseNum_rendering > 0)) {
						--poseNum_rendering;

						char buffer[256];
						if (!isfinite(FPS_inference))
							sprintf_s(buffer, "REdiSplats - Pose: %d / %d", poseNum_rendering + 1, NUMBER_OF_POSES);
						else
							sprintf_s(buffer, "REdiSplats - Pose: %d / %d (FPS: %.4lf)", poseNum_rendering + 1, NUMBER_OF_POSES, FPS_inference);
						SetWindowTextA(hWnd, buffer);

						Ox = poses[poseNum_rendering].Ox; Oy = poses[poseNum_rendering].Oy; Oz = poses[poseNum_rendering].Oz;
						Rx = poses[poseNum_rendering].Rx; Ry = poses[poseNum_rendering].Ry; Rz = poses[poseNum_rendering].Rz;
						Dx = poses[poseNum_rendering].Dx; Dy = poses[poseNum_rendering].Dy; Dz = poses[poseNum_rendering].Dz;
						Fx = poses[poseNum_rendering].Fx; Fy = poses[poseNum_rendering].Fy; Fz = poses[poseNum_rendering].Fz;

						customPose = false;
						cameraChanged = true;
					}

					if (right_square_bracket_down && (poseNum_rendering < NUMBER_OF_POSES - 1)) {
						++poseNum_rendering;

						char buffer[256];
						if (!isfinite(FPS_inference))
							sprintf_s(buffer, "REdiSplats - Pose: %d / %d", poseNum_rendering + 1, NUMBER_OF_POSES);
						else
							sprintf_s(buffer, "REdiSplats - Pose: %d / %d (FPS: %.4lf)", poseNum_rendering + 1, NUMBER_OF_POSES, FPS_inference);
						SetWindowTextA(hWnd, buffer);

						Ox = poses[poseNum_rendering].Ox; Oy = poses[poseNum_rendering].Oy; Oz = poses[poseNum_rendering].Oz;
						Rx = poses[poseNum_rendering].Rx; Ry = poses[poseNum_rendering].Ry; Rz = poses[poseNum_rendering].Rz;
						Dx = poses[poseNum_rendering].Dx; Dy = poses[poseNum_rendering].Dy; Dz = poses[poseNum_rendering].Dz;
						Fx = poses[poseNum_rendering].Fx; Fy = poses[poseNum_rendering].Fy; Fz = poses[poseNum_rendering].Fz;

						customPose = false;
						cameraChanged = true;
					}

					if (flyMode) {
						if (D_down) {
							char buffer[256];
							if (!isfinite(FPS_inference))
								sprintf_s(buffer, "REdiSplats - Custom pose");
							else
								sprintf_s(buffer, "REdiSplats - Custom pose (FPS: %.4lf)", FPS_inference);
							SetWindowTextA(hWnd, buffer);

							Ox = Ox + (Rx * 0.1f);
							Oy = Oy + (Ry * 0.1f);
							Oz = Oz + (Rz * 0.1f);

							customPose = true;
							cameraChanged = true;
						}
						if (A_down) {
							char buffer[256];
							if (!isfinite(FPS_inference))
								sprintf_s(buffer, "REdiSplats - Custom pose");
							else
								sprintf_s(buffer, "REdiSplats - Custom pose (FPS: %.4lf)", FPS_inference);
							SetWindowTextA(hWnd, buffer);

							Ox = Ox - (Rx * 0.1f);
							Oy = Oy - (Ry * 0.1f);
							Oz = Oz - (Rz * 0.1f);

							customPose = true;
							cameraChanged = true;

						}
						if (W_down) {
							char buffer[256];
							if (!isfinite(FPS_inference))
								sprintf_s(buffer, "REdiSplats - Custom pose");
							else
								sprintf_s(buffer, "REdiSplats - Custom pose (FPS: %.4lf)", FPS_inference);
							SetWindowTextA(hWnd, buffer);

							Ox = Ox + (Fx * 0.1f);
							Oy = Oy + (Fy * 0.1f);
							Oz = Oz + (Fz * 0.1f);

							customPose = true;
							cameraChanged = true;
						}
						if (S_down) {
							char buffer[256];
							if (!isfinite(FPS_inference))
								sprintf_s(buffer, "REdiSplats - Custom pose");
							else
								sprintf_s(buffer, "REdiSplats - Custom pose (FPS: %.4lf)", FPS_inference);
							SetWindowTextA(hWnd, buffer);

							Ox = Ox - (Fx * 0.1f);
							Oy = Oy - (Fy * 0.1f);
							Oz = Oz - (Fz * 0.1f);

							customPose = true;
							cameraChanged = true;
						}
						if (C_down) {
							char buffer[256];
							if (!isfinite(FPS_inference))
								sprintf_s(buffer, "REdiSplats - Custom pose");
							else
								sprintf_s(buffer, "REdiSplats - Custom pose (FPS: %.4lf)", FPS_inference);
							SetWindowTextA(hWnd, buffer);

							Ox = Ox + (Dx * 0.1f);
							Oy = Oy + (Dy * 0.1f);
							Oz = Oz + (Dz * 0.1f);

							customPose = true;
							cameraChanged = true;
						}
						if (Space_down) {
							char buffer[256];
							if (!isfinite(FPS_inference))
								sprintf_s(buffer, "REdiSplats - Custom pose");
							else
								sprintf_s(buffer, "REdiSplats - Custom pose (FPS: %.4lf)", FPS_inference);
							SetWindowTextA(hWnd, buffer);

							Ox = Ox - (Dx * 0.1f);
							Oy = Oy - (Dy * 0.1f);
							Oz = Oz - (Dz * 0.1f);

							customPose = true;
							cameraChanged = true;
						}
					}

					// *** *** *** *** ***

					if (!cameraChanged) {
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

						// *** *** *** *** ***

						if (needs_to_render_train_or_test_poses) {
							config.ray_termination_T_threshold = config.ray_termination_T_threshold_inference; // !!! !!! !!!
							SetConfigurationOptiX(config);

							// *** *** *** *** ***

							char *bitmap = NULL;

							// *** *** *** *** ***

							if (needs_to_visualize_train_or_test)
								bitmap = (char *)malloc(sizeof(char) * bitmapSize);

							// *** *** *** *** ***

							// TRAIN
							if (needs_to_render_train_poses) {
								char buf[256];
								wchar_t bufw[256];

								// *** *** *** *** ***

								if (needs_to_visualize_train) {
									swprintf(bufw, 256, L"output\\%d\\renders\\train\\%d", next_available_dir_id, params_OptiX->epoch);
									CreateDirectory(bufw, NULL);

									wcstombs_s(NULL, buf, 256, bufw, 256);
								}

								// *** *** *** *** ***

								double MSE = 0.0;
								double PSNR = 0.0;
								double FPS = 0.0;

								for (int pose = 0; pose < NUMBER_OF_POSES; ++pose) {
									params_OptiX->O.x = poses[pose].Ox; params_OptiX->O.y = poses[pose].Oy; params_OptiX->O.z = poses[pose].Oz;
									params_OptiX->R.x = poses[pose].Rx; params_OptiX->R.y = poses[pose].Ry; params_OptiX->R.z = poses[pose].Rz;
									params_OptiX->D.x = poses[pose].Dx; params_OptiX->D.y = poses[pose].Dy; params_OptiX->D.z = poses[pose].Dz;
									params_OptiX->F.x = poses[pose].Fx; params_OptiX->F.y = poses[pose].Fy; params_OptiX->F.z = poses[pose].Fz;
									params_OptiX->copyBitmapToHostMemory = true;

									// *** *** *** *** ***

									LARGE_INTEGER lpPerformanceCount1;
									LARGE_INTEGER lpPerformanceCount2;

									bool result;

									QueryPerformanceCounter(&lpPerformanceCount1);

									if      constexpr (SH_degree == 0) result = RenderOptiXSH0(*params_OptiX);
									else if constexpr (SH_degree == 1) result = RenderOptiXSH1(*params_OptiX);
									else if constexpr (SH_degree == 2) result = RenderOptiXSH2(*params_OptiX);
									else if constexpr (SH_degree == 3) result = RenderOptiXSH3(*params_OptiX);
									else if constexpr (SH_degree == 4) result = RenderOptiXSH4(*params_OptiX);

									QueryPerformanceCounter(&lpPerformanceCount2);

									swprintf(consoleBuffer, 256, L"Render OptiX: %s", (result ? L"OK... .\n" : L"Failed... .\n"));
									WriteConsole(GetStdHandle(STD_OUTPUT_HANDLE), consoleBuffer, wcslen(consoleBuffer), NULL, NULL);

									// *** *** *** *** ***

									if (needs_to_visualize_train) {
										char *fName;
										char fPath[256];

										if (strcmp(config.data_format, "colmap") != 0)
											fName = strrchr(img_names[pose], '/') + 1; // !!! !!! !!!
										else
											fName = img_names[pose];

										sprintf_s(fPath, 256, "%s\\%s_iter_%d.bmp", buf, fName, epochNum);

										// *** *** *** *** ***

										FILE *f;

										// *** *** *** *** ***

										fopen_s(&f, fPath, "wb+");
										fwrite(BMPFileHeader, 54, 1, f);

										// *** *** *** *** ***

										for (int i = 0; i < bitmapHeight; ++i) {
											for (int j = 0; j < bitmapWidth; ++j) {
												unsigned color = params_OptiX->bitmap_host[(i * bitmapWidth) + j];

												unsigned char R = color >> 16;
												unsigned char G = (color >> 8) & 255;
												unsigned char B = color & 255;

												bitmap[(((bitmapHeight - 1 - i) * scanLineSize) + (j * 3))] = B;
												bitmap[(((bitmapHeight - 1 - i) * scanLineSize) + (j * 3)) + 1] = G;
												bitmap[(((bitmapHeight - 1 - i) * scanLineSize) + (j * 3)) + 2] = R;
											}
										}

										// *** *** *** *** ***

										fwrite(bitmap, bitmapSize - 54, 1, f); // !!! !!! !!!
										fclose(f);

										// *** *** *** *** ***

										mbstowcs_s(NULL, bufw, 256, fPath, 256);  // !!! !!! !!!
										swprintf(consoleBuffer, 256, L"Saving render to file %s... .\n", bufw);
										WriteConsole(GetStdHandle(STD_OUTPUT_HANDLE), consoleBuffer, wcslen(consoleBuffer), NULL, NULL);
									}

									// *** *** *** *** ***

									if (needs_to_evaluate_train) {
										double poseMSE = 0.0;

										for (int i = 0; i < params_OptiX->height; ++i) {
											for (int j = 0; j < params_OptiX->width; ++j) {
												int color_out = params_OptiX->bitmap_host[(i * params_OptiX->width) + j];
												int R_out_i = color_out >> 16;
												int G_out_i = (color_out >> 8) & 255;
												int B_out_i = color_out & 255;
												float R_out = R_out_i / 256.0f;
												float G_out = G_out_i / 256.0f;
												float B_out = B_out_i / 256.0f;

												int color_ref = bitmap_ref[(pose * params_OptiX->width * params_OptiX->height) + ((i * params_OptiX->width) + j)];
												int R_ref_i = color_ref >> 16;
												int G_ref_i = (color_ref >> 8) & 255;
												int B_ref_i = color_ref & 255;
												float R_ref = R_ref_i / 256.0f;
												float G_ref = G_ref_i / 256.0f;
												float B_ref = B_ref_i / 256.0f;

												poseMSE += (((R_out - R_ref) * (R_out - R_ref)) + ((G_out - G_ref) * (G_out - G_ref)) + ((B_out - B_ref) * (B_out - B_ref)));
											}
										}
										poseMSE /= 3.0 * params_OptiX->width * params_OptiX->height;
										double posePSNR = -10.0 * (log(poseMSE) / log(10.0));

										double poseFPS = *((long long int *) &lpFrequency) / ((double)(*((long long int *) &lpPerformanceCount2) - *((long long int *) &lpPerformanceCount1)));
										swprintf(consoleBuffer, 256, L"TRAIN POSE: %d, PSNR: %.30lf, FPS: %.4lf;\n", pose + 1, posePSNR, poseFPS);
										WriteConsole(GetStdHandle(STD_OUTPUT_HANDLE), consoleBuffer, wcslen(consoleBuffer), NULL, NULL);

										FPS += poseFPS;
										MSE += poseMSE;
										PSNR += posePSNR;
									}
								}

								FPS /= NUMBER_OF_POSES;
								MSE /= NUMBER_OF_POSES;
								PSNR /= NUMBER_OF_POSES;

								// *** *** *** *** ***

								if (needs_to_evaluate_train) {
									char fPath[256];
									FILE *f;

									sprintf_s(fPath, 256, "output\\%d\\stats\\MSE_train.txt", next_available_dir_id);

									fopen_s(&f, fPath, "at");
									fprintf(f, "%d: %.30lf,\n", epochNum, MSE);
									fclose(f);

									sprintf_s(fPath, 256, "output\\%d\\stats\\PSNR_train.txt", next_available_dir_id);

									fopen_s(&f, fPath, "at");
									fprintf(f, "%d: %.30lf,\n", epochNum, PSNR);
									fclose(f);

									sprintf_s(fPath, 256, "output\\%d\\stats\\FPS_train.txt", next_available_dir_id);

									fopen_s(&f, fPath, "at");
									fprintf(f, "%d: %.4lf,\n", epochNum, FPS);
									fclose(f);

									// *** *** *** *** ***

									swprintf(consoleBuffer, 256, L"MSE TRAIN: %.30lf;\n", MSE);
									WriteConsole(GetStdHandle(STD_OUTPUT_HANDLE), consoleBuffer, wcslen(consoleBuffer), NULL, NULL);
									swprintf(consoleBuffer, 256, L"PSNR TRAIN: %.30lf;\n", PSNR);
									WriteConsole(GetStdHandle(STD_OUTPUT_HANDLE), consoleBuffer, wcslen(consoleBuffer), NULL, NULL);
									swprintf(consoleBuffer, 256, L"FPS TRAIN: %.4lf;\n", FPS);
									WriteConsole(GetStdHandle(STD_OUTPUT_HANDLE), consoleBuffer, wcslen(consoleBuffer), NULL, NULL);
								}
							}

							// *** *** *** *** ***

							// TEST
							if (needs_to_render_test_poses) {
								char buf[256];
								wchar_t bufw[256];

								// *** *** *** *** ***

								if (needs_to_visualize_test) {
									swprintf(bufw, 256, L"output\\%d\\renders\\test\\%d", next_available_dir_id, params_OptiX->epoch);
									CreateDirectory(bufw, NULL);

									wcstombs_s(NULL, buf, 256, bufw, 256);

									bitmap = (char *)malloc(sizeof(char) * bitmapSize);
								}

								// *** *** *** *** ***

								double MSE = 0.0;
								double PSNR = 0.0;
								double FPS = 0.0;

								for (int pose = 0; pose < NUMBER_OF_POSES_TEST; ++pose) {
									params_OptiX->O.x = poses_test[pose].Ox; params_OptiX->O.y = poses_test[pose].Oy; params_OptiX->O.z = poses_test[pose].Oz;
									params_OptiX->R.x = poses_test[pose].Rx; params_OptiX->R.y = poses_test[pose].Ry; params_OptiX->R.z = poses_test[pose].Rz;
									params_OptiX->D.x = poses_test[pose].Dx; params_OptiX->D.y = poses_test[pose].Dy; params_OptiX->D.z = poses_test[pose].Dz;
									params_OptiX->F.x = poses_test[pose].Fx; params_OptiX->F.y = poses_test[pose].Fy; params_OptiX->F.z = poses_test[pose].Fz;
									params_OptiX->copyBitmapToHostMemory = true;

									// *** *** *** *** ***

									LARGE_INTEGER lpPerformanceCount1;
									LARGE_INTEGER lpPerformanceCount2;

									bool result;

									QueryPerformanceCounter(&lpPerformanceCount1);

									if      constexpr (SH_degree == 0) result = RenderOptiXSH0(*params_OptiX);
									else if constexpr (SH_degree == 1) result = RenderOptiXSH1(*params_OptiX);
									else if constexpr (SH_degree == 2) result = RenderOptiXSH2(*params_OptiX);
									else if constexpr (SH_degree == 3) result = RenderOptiXSH3(*params_OptiX);
									else if constexpr (SH_degree == 4) result = RenderOptiXSH4(*params_OptiX);

									QueryPerformanceCounter(&lpPerformanceCount2);

									swprintf(consoleBuffer, 256, L"Render OptiX: %s", (result ? L"OK... .\n" : L"Failed... .\n"));
									WriteConsole(GetStdHandle(STD_OUTPUT_HANDLE), consoleBuffer, wcslen(consoleBuffer), NULL, NULL);

									// *** *** *** *** ***

									if (needs_to_visualize_test) {
										char *fName;
										char fPath[256];

										if (strcmp(config.data_format, "colmap") != 0)
											fName = strrchr(img_names_test[pose], '/') + 1; // !!! !!! !!!
										else
											fName = img_names_test[pose];

										sprintf_s(fPath, 256, "%s\\%s_iter_%d.bmp", buf, fName, epochNum);

										// *** *** *** *** ***

										FILE *f;

										// *** *** *** *** ***

										fopen_s(&f, fPath, "wb+");
										fwrite(BMPFileHeader, 54, 1, f);

										// *** *** *** *** ***

										for (int i = 0; i < bitmapHeight; ++i) {
											for (int j = 0; j < bitmapWidth; ++j) {
												unsigned color = params_OptiX->bitmap_host[(i * bitmapWidth) + j];

												unsigned char R = color >> 16;
												unsigned char G = (color >> 8) & 255;
												unsigned char B = color & 255;

												bitmap[(((bitmapHeight - 1 - i) * scanLineSize) + (j * 3))] = B;
												bitmap[(((bitmapHeight - 1 - i) * scanLineSize) + (j * 3)) + 1] = G;
												bitmap[(((bitmapHeight - 1 - i) * scanLineSize) + (j * 3)) + 2] = R;
											}
										}

										// *** *** *** *** ***

										fwrite(bitmap, bitmapSize - 54, 1, f); // !!! !!! !!!
										fclose(f);

										// *** *** *** *** ***

										mbstowcs_s(NULL, bufw, 256, fPath, 256); // !!! !!! !!!
										swprintf(consoleBuffer, 256, L"Saving render to file %s... .\n", bufw);
										WriteConsole(GetStdHandle(STD_OUTPUT_HANDLE), consoleBuffer, wcslen(consoleBuffer), NULL, NULL);
									}

									// *** *** *** *** ***

									if (needs_to_evaluate_test) {
										double poseMSE = 0.0;

										for (int i = 0; i < params_OptiX->height; ++i) {
											for (int j = 0; j < params_OptiX->width; ++j) {
												int color_out = params_OptiX->bitmap_host[(i * params_OptiX->width) + j];
												int R_out_i = color_out >> 16;
												int G_out_i = (color_out >> 8) & 255;
												int B_out_i = color_out & 255;
												float R_out = R_out_i / 256.0f;
												float G_out = G_out_i / 256.0f;
												float B_out = B_out_i / 256.0f;

												int color_ref = bitmap_ref_test[(pose * params_OptiX->width * params_OptiX->height) + ((i * params_OptiX->width) + j)];
												int R_ref_i = color_ref >> 16;
												int G_ref_i = (color_ref >> 8) & 255;
												int B_ref_i = color_ref & 255;
												float R_ref = R_ref_i / 256.0f;
												float G_ref = G_ref_i / 256.0f;
												float B_ref = B_ref_i / 256.0f;

												poseMSE += (((R_out - R_ref) * (R_out - R_ref)) + ((G_out - G_ref) * (G_out - G_ref)) + ((B_out - B_ref) * (B_out - B_ref)));
											}
										}
										poseMSE /= 3.0 * params_OptiX->width * params_OptiX->height;
										double posePSNR = -10.0 * (log(poseMSE) / log(10.0));

										double poseFPS = *((long long int *) &lpFrequency) / ((double)(*((long long int *) &lpPerformanceCount2) - *((long long int *) &lpPerformanceCount1)));
										swprintf(consoleBuffer, 256, L"TEST POSE: %d, PSNR: %.30lf, FPS: %.4lf;\n", pose + 1, posePSNR, poseFPS);
										WriteConsole(GetStdHandle(STD_OUTPUT_HANDLE), consoleBuffer, wcslen(consoleBuffer), NULL, NULL);

										FPS += poseFPS;
										MSE += poseMSE;
										PSNR += posePSNR;
									}
								}

								FPS /= NUMBER_OF_POSES_TEST;
								MSE /= NUMBER_OF_POSES_TEST;
								PSNR /= NUMBER_OF_POSES_TEST;

								// *** *** *** *** ***

								if (needs_to_evaluate_test) {
									char fPath[256];
									FILE *f;

									sprintf_s(fPath, 256, "output\\%d\\stats\\MSE_test.txt", next_available_dir_id);

									fopen_s(&f, fPath, "at");
									fprintf(f, "%d: %.30lf,\n", epochNum, MSE);
									fclose(f);

									sprintf_s(fPath, 256, "output\\%d\\stats\\PSNR_test.txt", next_available_dir_id);

									fopen_s(&f, fPath, "at");
									fprintf(f, "%d: %.30lf,\n", epochNum, PSNR);
									fclose(f);

									sprintf_s(fPath, 256, "output\\%d\\stats\\FPS_test.txt", next_available_dir_id);

									fopen_s(&f, fPath, "at");
									fprintf(f, "%d: %.4lf,\n", epochNum, FPS);
									fclose(f);

									// *** *** *** *** ***

									swprintf(consoleBuffer, 256, L"MSE TEST: %.30lf;\n", MSE);
									WriteConsole(GetStdHandle(STD_OUTPUT_HANDLE), consoleBuffer, wcslen(consoleBuffer), NULL, NULL);
									swprintf(consoleBuffer, 256, L"PSNR TEST: %.30lf;\n", PSNR);
									WriteConsole(GetStdHandle(STD_OUTPUT_HANDLE), consoleBuffer, wcslen(consoleBuffer), NULL, NULL);
									swprintf(consoleBuffer, 256, L"FPS TEST: %.4lf;\n", FPS);
									WriteConsole(GetStdHandle(STD_OUTPUT_HANDLE), consoleBuffer, wcslen(consoleBuffer), NULL, NULL);
								}
							}

							// *** *** *** *** ***

							if (needs_to_visualize_train_or_test)
								free(bitmap);

							// *** *** *** *** ***

							config.ray_termination_T_threshold = ray_termination_T_threshold_training; // !!! !!! !!!
							SetConfigurationOptiX(config);
						}
						
						// *** *** *** *** ***

						// PROGRAM TERMINATION
						++epochNum;

						// !!! !!! !!!
						if (epochNum > config.end_epoch)
							PostQuitMessage(0);
						// !!! !!! !!!

						// !!! !!! !!!
						if (params_OptiX->epoch % 10 == 0) cameraChanged = true;
						// !!! !!! !!!

						// *** *** *** *** ***

						LARGE_INTEGER lpPerformanceCount1;
						LARGE_INTEGER lpPerformanceCount2;

						// *** *** *** *** ***

						bool result;

						// !!! !!! !!!
						QueryPerformanceCounter(&lpPerformanceCount1);
						// !!! !!! !!!

						if      constexpr (SH_degree == 0) result = ZeroGradientOptiXSH0(*params_OptiX);
						else if constexpr (SH_degree == 1) result = ZeroGradientOptiXSH1(*params_OptiX);
						else if constexpr (SH_degree == 2) result = ZeroGradientOptiXSH2(*params_OptiX);
						else if constexpr (SH_degree == 3) result = ZeroGradientOptiXSH3(*params_OptiX);
						else if constexpr (SH_degree == 4) result = ZeroGradientOptiXSH4(*params_OptiX);

						// !!! !!! !!!
						QueryPerformanceCounter(&lpPerformanceCount2);
						training_time += (((double)(*((long long int *) &lpPerformanceCount2) - *((long long int *) &lpPerformanceCount1))) / *((long long int *) &lpFrequency));
						// !!! !!! !!!

						swprintf(consoleBuffer, 256, L"Zero OptiX gradient: %s", (result ? L"OK... .\n" : L"Failed... .\n"));
						WriteConsole(GetStdHandle(STD_OUTPUT_HANDLE), consoleBuffer, wcslen(consoleBuffer), NULL, NULL);

						// *** *** *** *** ***

						// !!! !!! !!!
						QueryPerformanceCounter(&lpPerformanceCount1);
						// !!! !!! !!!

						params_OptiX->O.x = poses[poseInd_training].Ox; params_OptiX->O.y = poses[poseInd_training].Oy; params_OptiX->O.z = poses[poseInd_training].Oz;
						params_OptiX->R.x = poses[poseInd_training].Rx; params_OptiX->R.y = poses[poseInd_training].Ry; params_OptiX->R.z = poses[poseInd_training].Rz;
						params_OptiX->D.x = poses[poseInd_training].Dx; params_OptiX->D.y = poses[poseInd_training].Dy; params_OptiX->D.z = poses[poseInd_training].Dz;
						params_OptiX->F.x = poses[poseInd_training].Fx; params_OptiX->F.y = poses[poseInd_training].Fy; params_OptiX->F.z = poses[poseInd_training].Fz;
		
						params_OptiX->poseNum = poseInd_training;
						params_OptiX->epoch = epochNum;
						params_OptiX->copyBitmapToHostMemory = false;

						if      constexpr (SH_degree == 0) result = RenderOptiXSH0(*params_OptiX, false);
						else if constexpr (SH_degree == 1) result = RenderOptiXSH1(*params_OptiX, false);
						else if constexpr (SH_degree == 2) result = RenderOptiXSH2(*params_OptiX, false);
						else if constexpr (SH_degree == 3) result = RenderOptiXSH3(*params_OptiX, false);
						else if constexpr (SH_degree == 4) result = RenderOptiXSH4(*params_OptiX, false);
								
						// !!! !!! !!!
						QueryPerformanceCounter(&lpPerformanceCount2);
						training_time += (((double)(*((long long int *) &lpPerformanceCount2) - *((long long int *) &lpPerformanceCount1))) / *((long long int *) &lpFrequency));
						// !!! !!! !!!

						swprintf(consoleBuffer, 256, L"Render OptiX: %s", (result ? L"OK... .\n" : L"Failed... .\n"));
						WriteConsole(GetStdHandle(STD_OUTPUT_HANDLE), consoleBuffer, wcslen(consoleBuffer), NULL, NULL);

						// *** *** *** *** ***

						float scene_extent;
						GetSceneExtentOptiX(scene_extent);
						swprintf(consoleBuffer, 256, L"Scene extent: %f;\n", scene_extent);
						WriteConsole(GetStdHandle(STD_OUTPUT_HANDLE), consoleBuffer, wcslen(consoleBuffer), NULL, NULL);

						// *** *** *** *** ***

						if (epochNum > 1) {
							swprintf(
								text,
								128,
								L"iteration: %d, pose: %d / %d (%d), MSE: %lf, PSNR: %f;",
								epochNum,
								poseNum_training + 1,
								NUMBER_OF_POSES,
								poseInd_training + 1,
								params_OptiX->loss_host / (3.0 * bitmapWidth * bitmapHeight),
								-10.0f * (logf(params_OptiX->loss_host / (3.0f * bitmapWidth * bitmapHeight)) / logf(10.0f))
							);
							SendMessage(GetDlgItem(hWnd, LABEL1), WM_SETTEXT, 0, (LPARAM)text);
						} else {
							swprintf(
								text,
								128,
								L"epoch: %d, pose: %d / %d (%d);",
								epochNum,
								poseNum_training + 1,
								NUMBER_OF_POSES,
								poseInd_training + 1
							);
							SendMessage(GetDlgItem(hWnd, LABEL1), WM_SETTEXT, 0, (LPARAM)text);
						}

						// *** *** *** *** ***

						// !!! !!! !!!
						QueryPerformanceCounter(&lpPerformanceCount1);
						// !!! !!! !!!

						if      constexpr (SH_degree == 0) result = UpdateGradientOptiXSH0(*params_OptiX);
						else if constexpr (SH_degree == 1) result = UpdateGradientOptiXSH1(*params_OptiX);
						else if constexpr (SH_degree == 2) result = UpdateGradientOptiXSH2(*params_OptiX);
						else if constexpr (SH_degree == 3) result = UpdateGradientOptiXSH3(*params_OptiX);
						else if constexpr (SH_degree == 4) result = UpdateGradientOptiXSH4(*params_OptiX);

						// !!! !!! !!!
						QueryPerformanceCounter(&lpPerformanceCount2);
						training_time += (((double)(*((long long int *) &lpPerformanceCount2) - *((long long int *) &lpPerformanceCount1))) / *((long long int *) &lpFrequency));
						// !!! !!! !!!

						swprintf(consoleBuffer, 256, L"Update gradient OptiX: %s", (result ? L"OK... .\n" : L"Failed... .\n"));
						WriteConsole(GetStdHandle(STD_OUTPUT_HANDLE), consoleBuffer, wcslen(consoleBuffer), NULL, NULL);

						swprintf(consoleBuffer, 256, L"EPOCH: %d, GAUSSIANS: %d, MSE: %.20lf\n", epochNum, params_OptiX->numberOfGaussians, params_OptiX->loss_host / (3.0 * bitmapWidth * bitmapHeight * 1));
						WriteConsole(GetStdHandle(STD_OUTPUT_HANDLE), consoleBuffer, wcslen(consoleBuffer), NULL, NULL);

						// *** *** *** *** ***
							
						// !!! !!! !!!
						QueryPerformanceCounter(&lpPerformanceCount1);
						// !!! !!! !!!

						if (poseNum_training < NUMBER_OF_POSES - 1) {
							++poseNum_training;
						} else {
							// !!! !!! !!!
							seed_dword_prev = seed_dword;
							// !!! !!! !!!

							for (int i = 0; i < NUMBER_OF_POSES; i++) poses_indices[i] = i;
							for (int i = 0; i < NUMBER_OF_POSES - 1; i++) {
								int index = i + (RandomInteger() % (NUMBER_OF_POSES - i));
								if (index != i) {
									poses_indices[i] ^= poses_indices[index];
									poses_indices[index] ^= poses_indices[i];
									poses_indices[i] ^= poses_indices[index];
								}
							}

							poseNum_training = 0;
						}

						poseInd_training = poses_indices[poseNum_training]; // !!! !!! !!!

						// !!! !!! !!!
						QueryPerformanceCounter(&lpPerformanceCount2);
						training_time += (((double)(*((long long int *) &lpPerformanceCount2) - *((long long int *) &lpPerformanceCount1))) / *((long long int *) &lpFrequency));
						// !!! !!! !!!

						// *** *** *** *** ***

						// SAVE CHECKPOINT
						if (
							(params_OptiX->epoch % config.saving_frequency == config.saving_iter) ||
							(params_OptiX->epoch == config.end_epoch)
						) {
							char buf[256];
							wchar_t bufw[256];

							swprintf(bufw, 256, L"output\\%d\\checkpoints\\%d", next_available_dir_id, params_OptiX->epoch);
							CreateDirectory(bufw, NULL);

							wcstombs_s(NULL, buf, 256, bufw, 256);

							// *** *** *** *** ***

							if      constexpr (SH_degree == 0) result = DumpParametersOptiXSH0(*params_OptiX, buf);
							else if constexpr (SH_degree == 1) result = DumpParametersOptiXSH1(*params_OptiX, buf);
							else if constexpr (SH_degree == 2) result = DumpParametersOptiXSH2(*params_OptiX, buf);
							else if constexpr (SH_degree == 3) result = DumpParametersOptiXSH3(*params_OptiX, buf);
							else if constexpr (SH_degree == 4) result = DumpParametersOptiXSH4(*params_OptiX, buf);

							// *** *** *** *** ***

							// !!! !!! !!!
							// Save the seed state before computing the training poses indices permutation
							char fPath[256];
							FILE *f;

							sprintf_s(fPath, "%s\\seed_iter_%d.checkpoint", buf, params_OptiX->epoch);

							fopen_s(&f, fPath, "wb");
							fwrite(&seed_dword_prev, sizeof(unsigned) * 1, 1, f);
							fclose(f);
							// !!! !!! !!!
						}

						// *** *** *** *** ***
							
						// SAVE TO *.PLY FILE
						if (
							(params_OptiX->epoch % config.saving_frequency_PLY == config.saving_iter_PLY) ||
							(params_OptiX->epoch == config.end_epoch)
						) {
							char buf[256];
							wchar_t bufw[256];

							swprintf(bufw, 256, L"output\\%d\\PLY files\\%d", next_available_dir_id, params_OptiX->epoch);
							CreateDirectory(bufw, NULL);

							wcstombs_s(NULL, buf, 256, bufw, 256);

							if      constexpr (SH_degree == 0) result = DumpParametersToPLYFileOptiXSH0(*params_OptiX, buf);
							else if constexpr (SH_degree == 1) result = DumpParametersToPLYFileOptiXSH1(*params_OptiX, buf);
							else if constexpr (SH_degree == 2) result = DumpParametersToPLYFileOptiXSH2(*params_OptiX, buf);
							else if constexpr (SH_degree == 3) result = DumpParametersToPLYFileOptiXSH3(*params_OptiX, buf);
							else if constexpr (SH_degree == 4) result = DumpParametersToPLYFileOptiXSH4(*params_OptiX, buf);
						}

						// *** *** *** *** ***

						if (
							(params_OptiX->epoch % config.evaluation_frequency_train == config.evaluation_iter_train) ||
							(params_OptiX->epoch % config.evaluation_frequency_test == config.evaluation_iter_test) ||
							(params_OptiX->epoch == config.end_epoch)
						) {
							char fPath[256];
							FILE *f;

							sprintf_s(fPath, 256, "output\\%d\\stats\\training_time.txt", next_available_dir_id);

							fopen_s(&f, fPath, "at");
							fprintf(f, "%d: %.2lf,\n", params_OptiX->epoch, training_time);
							fclose(f);
						}
					} else {
						params_OptiX->O.x = Ox; params_OptiX->O.y = Oy; params_OptiX->O.z = Oz;
						params_OptiX->R.x = Rx; params_OptiX->R.y = Ry; params_OptiX->R.z = Rz;
						params_OptiX->D.x = Dx; params_OptiX->D.y = Dy; params_OptiX->D.z = Dz;
						params_OptiX->F.x = Fx; params_OptiX->F.y = Fy; params_OptiX->F.z = Fz;
						params_OptiX->copyBitmapToHostMemory = true;

						// *** *** *** *** ***

						config.ray_termination_T_threshold = config.ray_termination_T_threshold_inference; // !!! !!! !!!
						SetConfigurationOptiX(config);

						// *** *** *** *** ***

						LARGE_INTEGER lpPerformanceCount1;
						LARGE_INTEGER lpPerformanceCount2;

						bool result;

						// *** *** *** *** ***

						QueryPerformanceCounter(&lpPerformanceCount1);

						if      constexpr (SH_degree == 0) result = RenderOptiXSH0(*params_OptiX);
						else if constexpr (SH_degree == 1) result = RenderOptiXSH1(*params_OptiX);
						else if constexpr (SH_degree == 2) result = RenderOptiXSH2(*params_OptiX);
						else if constexpr (SH_degree == 3) result = RenderOptiXSH3(*params_OptiX);
						else if constexpr (SH_degree == 4) result = RenderOptiXSH4(*params_OptiX);

						QueryPerformanceCounter(&lpPerformanceCount2);

						// *** *** *** *** ***
		
						swprintf(consoleBuffer, 256, L"Render OptiX: %s", (result ? L"OK... .\n" : L"Failed... .\n"));
						WriteConsole(GetStdHandle(STD_OUTPUT_HANDLE), consoleBuffer, wcslen(consoleBuffer), NULL, NULL);

						// *** *** *** *** ***

						config.ray_termination_T_threshold = ray_termination_T_threshold_training;
						SetConfigurationOptiX(config);

						// *** *** *** *** ***

						// PRINT SCREEN
						// !!! !!! !!!
						if (print_screen_down) {
							char fPath[256];

							// *** *** *** *** ***

							sprintf_s(fPath, 256, "output\\%d\\screenshots\\%d.bmp", next_available_dir_id, next_available_screenshot_id);

							// *** *** *** *** ***

							FILE *f;

							// *** *** *** *** ***

							fopen_s(&f, fPath, "wb+");
							fwrite(BMPFileHeader, 54, 1, f);

							// *** *** *** *** ***

							char *bitmap = (char *)malloc(sizeof(char) * bitmapSize);

							// *** *** *** *** ***

							for (int i = 0; i < bitmapHeight; ++i) {
								for (int j = 0; j < bitmapWidth; ++j) {
									unsigned color = params_OptiX->bitmap_host[(i * bitmapWidth) + j];

									unsigned char R = color >> 16;
									unsigned char G = (color >> 8) & 255;
									unsigned char B = color & 255;

									bitmap[(((bitmapHeight - 1 - i) * scanLineSize) + (j * 3))] = B;
									bitmap[(((bitmapHeight - 1 - i) * scanLineSize) + (j * 3)) + 1] = G;
									bitmap[(((bitmapHeight - 1 - i) * scanLineSize) + (j * 3)) + 2] = R;
								}
							}

							// *** *** *** *** ***

							fwrite(bitmap, bitmapSize - 54, 1, f); // !!! !!! !!!
							fclose(f);

							// *** *** *** *** ***

							free(bitmap);

							// *** *** *** *** ***

							wchar_t bufw[256];

							mbstowcs_s(NULL, bufw, 256, fPath, 256);  // !!! !!! !!!
							swprintf(consoleBuffer, 256, L"Saving screenshot to file %s... .\n", bufw);
							WriteConsole(GetStdHandle(STD_OUTPUT_HANDLE), consoleBuffer, wcslen(consoleBuffer), NULL, NULL);

							// *** *** *** *** ***

							++next_available_screenshot_id; // !!! !!! !!!
							print_screen_down = false;
						}
						// !!! !!! !!!

						// *** *** *** *** ***

						char buffer[256];

						FPS_inference = *((long long int *) &lpFrequency) / ((double)(*((long long int *) &lpPerformanceCount2) - *((long long int *) &lpPerformanceCount1)));

						if (customPose)
							sprintf_s(buffer, "REdiSplats - Custom pose (FPS: %.4lf)", FPS_inference);
						else
							sprintf_s(buffer, "REdiSplats - Pose: %d / %d (FPS: %.4lf)", poseNum_rendering + 1, NUMBER_OF_POSES, FPS_inference);
						SetWindowTextA(hWnd, buffer);

						// *** *** *** *** ***

						cameraChanged = false;
						RedrawWindow(hWnd, NULL, NULL, RDW_INVALIDATE);
					}
				}
			}
			break;
		}
		case WM_DESTROY : {
			PostQuitMessage(0);
			break;
		}
		default : {
			return DefWindowProc(hWnd, message, wParam, lParam);
		}
	}
    return 0;
}

// *************************************************************************************************

INT_PTR CALLBACK About(HWND hDlg, UINT message, WPARAM wParam, LPARAM lParam) {
    UNREFERENCED_PARAMETER(lParam);

    switch (message) {
		case WM_INITDIALOG : {
			return (INT_PTR)TRUE;
		}
		case WM_COMMAND : {
			if ((LOWORD(wParam) == IDOK) || (LOWORD(wParam) == IDCANCEL)) {
				EndDialog(hDlg, LOWORD(wParam));
				return (INT_PTR)TRUE;
			}
			break;
		}
    }
    return (INT_PTR)FALSE;
}