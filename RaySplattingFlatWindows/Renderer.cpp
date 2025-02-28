#define _USE_MATH_DEFINES
#include <conio.h>
#include <intrin.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "Renderer.h"

// *** *** *** *** ***

unsigned seed_float = 0;

float RandomFloat() {
	float result;

	*((unsigned*)&result) = (127 << 23) | (seed_float & ((1 << 23) - 1));
	seed_float = (1664525 * seed_float) + 1013904223;
	return result - 1.0f;
}

// *** *** *** *** ***

unsigned RandomInteger() {
	unsigned result;

	result = seed_float;
	seed_float = (1664525 * seed_float) + 1013904223;
	return result;
}

// *** *** *** *** ***

unsigned long long seed_double = 0;

double RandomDouble() {
	double result;

	*((unsigned long long*) & result) = (1023ULL << 52) | (seed_double & ((1ULL << 52) - 1ULL));
	seed_double = (6364136223846793005ULL * seed_double) + 1442695040888963407ULL;
	return result - 1.0;
}

// *** *** *** *** ***

void RandomNormalFloat(float& Z1, float& Z2) {
	float U1 = RandomFloat();
	float U2 = RandomFloat();
	float tmp1 = sqrt(-2.0f * log(U1));
	float tmp2 = 2.0f * M_PI * U2;
	Z1 = tmp1 * cos(tmp2);
	Z2 = tmp1 * sin(tmp2);
}

// *** *** *** *** ***

void RandomNormalDouble(double& Z1, double& Z2) {
	double U1;
	double tmp1;
	do {
		U1 = RandomDouble();
		tmp1 = log(U1);
	} while (!(isfinite(tmp1)));
	double U2 = RandomDouble();
	tmp1 = sqrt(-2.0 * tmp1);
	double tmp2 = 2.0 * M_PI * U2;
	Z1 = tmp1 * cos(tmp2);
	Z2 = tmp1 * sin(tmp2);
}