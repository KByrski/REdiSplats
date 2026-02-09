#include "Utils.h"

// *** *** *** *** ***

unsigned seed_dword = 0;

// *** *** *** *** ***

void CopyFile(const char *fNameDest, const char *fNameSrc) {
	FILE *f;

	f = fopen(fNameSrc, "rb");
	fseek(f, 0, SEEK_END);
	int size = ftell(f);
	fseek(f, 0, SEEK_SET);
	void *buf = malloc(size);
	fread(buf, size, 1, f);
	fclose(f);
	
	f = fopen(fNameDest, "wb");
	fwrite(buf, size, 1, f);
	fclose(f);
	
	free(buf);
}

// *** *** *** *** ***

unsigned RandomInteger() {
	unsigned result;

	result = seed_dword;
	seed_dword = (1664525 * seed_dword) + 1013904223;
	return result;
}

// *** *** *** *** ***

void SaveBMPFile(const char *fPath, void *BMPFileHeader, int bitmapWidth, int bitmapHeight, int bitmapSize, int scanLineSize, unsigned *bitmap, char *buffer) { 
	FILE *f;

	// *********************************************************************************************

	f = fopen(fPath, "wb+");
	fwrite(BMPFileHeader, 54, 1, f);

	// *********************************************************************************************

	for (int i = 0; i < bitmapHeight; ++i) {
		for (int j = 0; j < bitmapWidth; ++j) {
			unsigned color = bitmap[(i * bitmapWidth) + j];

			unsigned char R = color >> 16;
			unsigned char G = (color >> 8) & 255;
			unsigned char B = color & 255;

			buffer[(((bitmapHeight - 1 - i) * scanLineSize) + (j * 3))] = B;
			buffer[(((bitmapHeight - 1 - i) * scanLineSize) + (j * 3)) + 1] = G;
			buffer[(((bitmapHeight - 1 - i) * scanLineSize) + (j * 3)) + 2] = R;
		}
	}

	// *** *** *** *** ***

	fwrite(buffer, bitmapSize - 54, 1, f); // !!! !!! !!!
	fclose(f);
}