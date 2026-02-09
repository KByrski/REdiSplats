#include <stdio.h>
#include <stdlib.h>

// *** *** *** *** ***

extern unsigned seed_dword;

// *** *** *** *** ***

void CopyFile(const char *fNameDest, const char *fNameSrc);
unsigned RandomInteger();
void SaveBMPFile(const char *fPath, void *BMPFileHeader, int bitmapWidth, int bitmapHeight, int bitmapSize, int scanLineSize, unsigned *bitmap, char *buffer);