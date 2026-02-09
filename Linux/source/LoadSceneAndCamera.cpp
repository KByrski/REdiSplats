#include "Header.cuh"

// *** *** *** *** ***

void LoadSceneAndCamera(
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
) {
	FILE *f;

	// *********************************************************************************************

	char filePath[256];
	strcpy(filePath, dataPath);
	strcat(filePath, "/");
	strcat(filePath, jsonFileName);
	f = fopen(filePath, "rt");
		
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
	sscanf(buf, " \"camera_angle_x\": %f", &FOV);

	while (fgets(buf, 256, f) != NULL) {
		char *str = strstr(buf, "file_path");
		if (str != NULL) {
			char fileName[256];
			sscanf(str, "file_path\": \"%s", fileName);

			char *tmp = strtok(fileName, "\"");
			img_names[poseNum] = (char *)malloc(sizeof(char) * (strlen(tmp) + 1));
			strcpy(img_names[poseNum], tmp);

			FILE *f_bitmap;
			strcpy(filePath, dataPath);
			strcat(filePath, "/");
			strcat(filePath, tmp);
			strcat(filePath, ".bmp");

			f_bitmap = fopen(filePath, "rb");

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
			sscanf(buf, "%f", &poses[poseNum].Rx);
			fgets(buf, 256, f);
			sscanf(buf, "%f", &poses[poseNum].Dx);
			fgets(buf, 256, f);
			sscanf(buf, "%f", &poses[poseNum].Fx);
			fgets(buf, 256, f);
			sscanf(buf, "%f", &poses[poseNum].Ox);

			fgets(buf, 256, f);
			fgets(buf, 256, f);

			fgets(buf, 256, f);
			sscanf(buf, "%f", &poses[poseNum].Ry);
			fgets(buf, 256, f);
			sscanf(buf, "%f", &poses[poseNum].Dy);
			fgets(buf, 256, f);
			sscanf(buf, "%f", &poses[poseNum].Fy);
			fgets(buf, 256, f);
			sscanf(buf, "%f", &poses[poseNum].Oy);

			fgets(buf, 256, f);
			fgets(buf, 256, f);

			fgets(buf, 256, f);
			sscanf(buf, "%f", &poses[poseNum].Rz);
			fgets(buf, 256, f);
			sscanf(buf, "%f", &poses[poseNum].Dz);
			fgets(buf, 256, f);
			sscanf(buf, "%f", &poses[poseNum].Fz);
			fgets(buf, 256, f);
			sscanf(buf, "%f", &poses[poseNum].Oz);

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