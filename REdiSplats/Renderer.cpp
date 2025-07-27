#define _USE_MATH_DEFINES
#include <conio.h>
#include <intrin.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "Renderer.h"

// *** *** *** *** ***

unsigned seed_dword = 0;

float RandomFloat() {
	float result;

	*((unsigned*)&result) = (127 << 23) | (seed_dword & ((1 << 23) - 1));
	seed_dword = (1664525 * seed_dword) + 1013904223;
	return result - 1.0f;
}

// *** *** *** *** ***

unsigned RandomInteger() {
	unsigned result;

	result = seed_dword;
	seed_dword = (1664525 * seed_dword) + 1013904223;
	return result;
}

// *** *** *** *** ***

unsigned long long seed_qword = 0;

double RandomDouble() {
	double result;

	*((unsigned long long*) & result) = (1023ULL << 52) | (seed_qword & ((1ULL << 52) - 1ULL));
	seed_qword = (6364136223846793005ULL * seed_qword) + 1442695040888963407ULL;
	return result - 1.0;
}