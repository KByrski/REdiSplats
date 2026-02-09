#include "Header.cuh"

// *** *** *** *** ***

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
	char **&img_names_train,
	char **&img_names_test,
	int &bitmapSize,
	int &scanLineSize,
	float &double_tan_half_fov_x, float &double_tan_half_fov_y
) {
	FILE *f;
	
	char filePath[256];
	strcpy(filePath, dataPath);
	strcat(filePath, "/");
	strcat(filePath, jsonFileName);

	f = fopen(filePath, "rb");
	fseek(f, 0, SEEK_END);
	int fSize = ftell(f);
	fclose(f);

	char *buf = (char *)malloc(sizeof(char) * (fSize + 1));

	f = fopen(filePath, "rt");
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
	
	img_names_train = (char **)malloc(sizeof(char *) * numberOfPoses_train); // !!! !!! !!!
	img_names_test = (char **)malloc(sizeof(char *) * numberOfPoses_test); // !!! !!! !!!

	void *bitmap_tmp = NULL;

	tmp1 = buf;
	for (int poseNum = 0; poseNum < numberOfPoses; ++poseNum) {
		tmp2 = strstr(tmp1, "\"img_name\"");		
		char tmp3[256];
		sscanf(tmp2, "\"img_name\": \"%s", tmp3);
		char *fName = strtok(tmp3, "\"");
		tmp1 = tmp2 + strlen("\"img_name\":");

		// *****************************************************************************************

		FILE *f_bitmap;
		strcpy(filePath, dataPath);
		strcat(filePath, "/images/");
		strcat(filePath, fName);
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
			bitmap_train = (int *)malloc(sizeof(int) * bitmapWidth * bitmapHeight * numberOfPoses_train);
			bitmap_test = (int *)malloc(sizeof(int) * bitmapWidth * bitmapHeight * numberOfPoses_test);
			bitmap_tmp = malloc(scanLineSize * bitmapHeight);

			numberOfPoses_train = 0; // !!! !!! !!!
			numberOfPoses_test = 0; // !!! !!! !!!
		}

		if ((poseNum & 7) != 0) {
			img_names_train[numberOfPoses_train] = (char *)malloc(sizeof(char) * (strlen(fName) + 1));
			strcpy(img_names_train[numberOfPoses_train], fName);
		} else {
			img_names_test[numberOfPoses_test] = (char *)malloc(sizeof(char) * (strlen(fName) + 1));
			strcpy(img_names_test[numberOfPoses_test], fName);
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

		// *****************************************************************************************
		
		int width;
		tmp2 = strstr(tmp1, "\"width\"");
		sscanf(tmp2, "\"width\": %d", &width);
		tmp1 = tmp2 + strlen("\"width\":");

		int height;
		tmp2 = strstr(tmp1, "\"height\"");
		sscanf(tmp2, "\"height\": %d", &height);
		tmp1 = tmp2 + strlen("\"height\":");
		
		tmp2 = strstr(tmp1, "\"position\"");
		if ((poseNum & 7) != 0)
			sscanf(
				tmp2,
				"\"position\": [%f, %f, %f]",
				&poses_train[numberOfPoses_train].Ox, &poses_train[numberOfPoses_train].Oy, &poses_train[numberOfPoses_train].Oz
			);
		else
			sscanf(
				tmp2,
				"\"position\": [%f, %f, %f]",
				&poses_test[numberOfPoses_test].Ox, &poses_test[numberOfPoses_test].Oy, &poses_test[numberOfPoses_test].Oz
			);
		tmp1 = tmp2 + strlen("\"position\":");

		tmp2 = strstr(tmp1, "\"rotation\"");
		if ((poseNum & 7) != 0)
			sscanf(
				tmp2,
				"\"rotation\": [[%f, %f, %f], [%f, %f, %f], [%f, %f, %f]]",
				&poses_train[numberOfPoses_train].Rx, &poses_train[numberOfPoses_train].Dx, &poses_train[numberOfPoses_train].Fx,
				&poses_train[numberOfPoses_train].Ry, &poses_train[numberOfPoses_train].Dy, &poses_train[numberOfPoses_train].Fy,
				&poses_train[numberOfPoses_train].Rz, &poses_train[numberOfPoses_train].Dz, &poses_train[numberOfPoses_train].Fz
			);
		else
			sscanf(
				tmp2,
				"\"rotation\": [[%f, %f, %f], [%f, %f, %f], [%f, %f, %f]]",
				&poses_test[numberOfPoses_test].Rx, &poses_test[numberOfPoses_test].Dx, &poses_test[numberOfPoses_test].Fx,
				&poses_test[numberOfPoses_test].Ry, &poses_test[numberOfPoses_test].Dy, &poses_test[numberOfPoses_test].Fy,
				&poses_test[numberOfPoses_test].Rz, &poses_test[numberOfPoses_test].Dz, &poses_test[numberOfPoses_test].Fz
			);
		tmp1 = tmp2 + strlen("\"rotation\":");

		float fy;
		tmp2 = strstr(tmp1, "\"fy\"");
		sscanf(tmp2, "\"fy\": %f", &fy);
		double_tan_half_fov_y = height / fy;
		tmp1 = tmp2 + strlen("\"fy\":");

		float fx;
		tmp2 = strstr(tmp1, "\"fx\"");
		sscanf(tmp2, "\"fx\": %f", &fx);
		double_tan_half_fov_x = width / fx;
		tmp1 = tmp2 + strlen("\"fx\":");

		if ((poseNum & 7) != 0)
			++numberOfPoses_train;
		else
			++numberOfPoses_test;
	}
}