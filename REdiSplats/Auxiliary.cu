#include "Header.cuh"

// *************************************************************************************************

void LoadFromFile(const char *dirPath, int epochNum, const char *fName, void *buf, int size) {
	FILE *f;

	char fPath[256];
	sprintf_s(fPath, "%s\\%s_iter_%d.checkpoint", dirPath, fName, epochNum);

	fopen_s(&f, fPath, "rb");
	fread(buf, size, 1, f);
	fclose(f);
}

// *************************************************************************************************

__global__ void GenerateInstances(
	float4 *GC_part_1, float4 *GC_part_2, float4 *GC_part_3, float2 *GC_part_4,
	int numberOfGaussians,
	OptixTraversableHandle GAS,
	float *instances
) {
	extern __shared__ float tmp[];

	int tid = (blockIdx.x * blockDim.x) + threadIdx.x;
	int wid = tid >> 5;
	int number_of_warps = numberOfGaussians >> 5;

	// *** *** *** *** ***

	if (wid <= number_of_warps) {
		int index = ((tid < numberOfGaussians) ? tid : (numberOfGaussians - 1));

		// *** *** *** *** ***

		float4 GC_1 = GC_part_1[index];
		float4 GC_2 = GC_part_2[index];
		float4 GC_3 = GC_part_3[index];
		float2 GC_4 = GC_part_4[index];

		// *** *** *** *** ***

		float aa = GC_3.z * GC_3.z;
		float bb = GC_3.w * GC_3.w;
		float cc = GC_4.x * GC_4.x;
		float dd = GC_4.y * GC_4.y;
		float s = 2.0f / (aa + bb + cc + dd);

		float bs = GC_3.w * s;  float cs = GC_4.x * s;  float ds = GC_4.y * s;
		float ab = GC_3.z * bs; float ac = GC_3.z * cs; float ad = GC_3.z * ds;
		bb = bb * s;			float bc = GC_3.w * cs; float bd = GC_3.w * ds;
		cc = cc * s;			float cd = GC_4.x * ds;       dd = dd * s;

		// !!! !!! !!!
		float radius_squared = chi_square_squared_radius - (2.0f * logf(1 + expf(-GC_1.w)));
		radius_squared = (radius_squared < 0.0f) ? 0.0f : radius_squared;

		float radius = sqrtf(radius_squared);
		// !!! !!! !!!

		float sX = expf(GC_2.w) * radius;
		float sY = expf(GC_3.x) * radius;
		float sZ = expf(GC_3.y) * radius;

		float Q11 = sX * (1.0f - cc - dd);
		float Q12 = sY * (bc - ad);
		float Q13 = sZ * (bd + ac);

		float Q21 = sX * (bc + ad);
		float Q22 = sY * (1.0f - bb - dd);
		float Q23 = sZ * (cd - ab);

		float Q31 = sX * (bd - ac);
		float Q32 = sY * (cd + ab);
		float Q33 = sZ * (1.0f - bb - cc);

		// *** *** *** *** ***

		float *base_address = &tmp[(threadIdx.x * 20) + (threadIdx.x >> 3)];

		// transform
		base_address[0] = Q11;
		base_address[1] = Q12;
		base_address[2] = Q13;
		base_address[3] = GC_2.x;

		base_address[4] = Q21;
		base_address[5] = Q22;
		base_address[6] = Q23;
		base_address[7] = GC_2.y;

		base_address[8] = Q31;
		base_address[9] = Q32;
		base_address[10] = Q33;
		base_address[11] = GC_2.z;

		// instanceId
		base_address[12] = 0.0f;

		// sbtOffset
		base_address[13] = 0.0f;

		// visibilityMask
		base_address[14] = __uint_as_float(255);

		// flags
		base_address[15] = __uint_as_float(OPTIX_INSTANCE_FLAG_NONE);

		// traversableHandle
		base_address[16] = __uint_as_float(GAS);
		base_address[17] = __uint_as_float(GAS >> 32);

		// pad
		base_address[18] = 0.0f;
		base_address[19] = 0.0f;
	}

	// *** *** *** *** ***

	__syncthreads();

	// *** *** *** *** ***

	if (wid <= number_of_warps) {
		int lane_id = threadIdx.x & 31;

		float *base_address_1 = &instances[(tid & -32) * 20];
		float *base_address_2 = &tmp[((threadIdx.x & -32) * 20) + ((threadIdx.x & -32) >> 3)];

		base_address_1[lane_id      ] = base_address_2[lane_id      ];
		base_address_1[lane_id + 32 ] = base_address_2[lane_id + 32 ];
		base_address_1[lane_id + 64 ] = base_address_2[lane_id + 64 ];
		base_address_1[lane_id + 96 ] = base_address_2[lane_id + 96 ];
		base_address_1[lane_id + 128] = base_address_2[lane_id + 128];

		base_address_1[lane_id + 160] = base_address_2[lane_id + 160 + 1];
		base_address_1[lane_id + 192] = base_address_2[lane_id + 192 + 1];
		base_address_1[lane_id + 224] = base_address_2[lane_id + 224 + 1];
		base_address_1[lane_id + 256] = base_address_2[lane_id + 256 + 1];
		base_address_1[lane_id + 288] = base_address_2[lane_id + 288 + 1];

		base_address_1[lane_id + 320] = base_address_2[lane_id + 320 + 2];
		base_address_1[lane_id + 352] = base_address_2[lane_id + 352 + 2];
		base_address_1[lane_id + 384] = base_address_2[lane_id + 384 + 2];
		base_address_1[lane_id + 416] = base_address_2[lane_id + 416 + 2];
		base_address_1[lane_id + 448] = base_address_2[lane_id + 448 + 2];

		base_address_1[lane_id + 480] = base_address_2[lane_id + 480 + 3];
		base_address_1[lane_id + 512] = base_address_2[lane_id + 512 + 3];
		base_address_1[lane_id + 544] = base_address_2[lane_id + 544 + 3];
		base_address_1[lane_id + 576] = base_address_2[lane_id + 576 + 3];
		base_address_1[lane_id + 608] = base_address_2[lane_id + 608 + 3];
	}
}

// *************************************************************************************************

__global__ void ComputeAABBs(
	float4 *GC_part_1, float4 *GC_part_2, float4 *GC_part_3, float2 *GC_part_4,
	int numberOfGaussians,
	float *AABBs
) {
	extern __shared__ float tmp[];

	int tid = (blockIdx.x * blockDim.x) + threadIdx.x;
	int wid = tid >> 5;
	int number_of_warps = numberOfGaussians >> 5;

	// *** *** *** *** ***

	if (wid <= number_of_warps) {
		int index = ((tid < numberOfGaussians) ? tid : (numberOfGaussians - 1));

		// *** *** *** *** ***

		float4 GC_1 = GC_part_1[index];
		float4 GC_2 = GC_part_2[index];
		float4 GC_3 = GC_part_3[index];
		float2 GC_4 = GC_part_4[index];

		// *** *** *** *** ***

		float aa = GC_3.z * GC_3.z;
		float bb = GC_3.w * GC_3.w;
		float cc = GC_4.x * GC_4.x;
		float dd = GC_4.y * GC_4.y;
		float s  = 2.0f / (aa + bb + cc + dd);

		float bs = GC_3.w * s;  float cs = GC_4.x * s;  float ds = GC_4.y * s;
		float ab = GC_3.z * bs; float ac = GC_3.z * cs; float ad = GC_3.z * ds;
		bb = bb * s;			float bc = GC_3.w * cs; float bd = GC_3.w * ds;
		cc = cc * s;			float cd = GC_4.x * ds;       dd = dd * s;

		float Q11 = 1.0f - cc - dd;
		float Q12 = bc - ad;
		float Q13 = bd + ac;

		float Q21 = bc + ad;
		float Q22 = 1.0f - bb - dd;
		float Q23 = cd - ab;

		float Q31 = bd - ac;
		float Q32 = cd + ab;
		float Q33 = 1.0f - bb - cc;

		float sX = expf(GC_2.w);
		float sY = expf(GC_3.x);
		float sZ = expf(GC_3.y);

		float sX_squared = sX * sX;
		float sY_squared = sY * sY;
		float sZ_squared = sZ * sZ;

		// !!! !!! !!!
		float radius_squared = chi_square_squared_radius - (2.0f * logf(1 + expf(-GC_1.w)));
		radius_squared = (radius_squared < 0.0f) ? 0.0f : radius_squared;
		// !!! !!! !!!

		float tmpX = __fsqrt_ru(radius_squared * __fmaf_rn(sX_squared, Q11 * Q11, __fmaf_rn(sY_squared, Q12 * Q12, sZ_squared * Q13 * Q13)));
		float tmpY = __fsqrt_ru(radius_squared * __fmaf_rn(sX_squared, Q21 * Q21, __fmaf_rn(sY_squared, Q22 * Q22, sZ_squared * Q23 * Q23)));
		float tmpZ = __fsqrt_ru(radius_squared * __fmaf_rn(sX_squared, Q31 * Q31, __fmaf_rn(sY_squared, Q32 * Q32, sZ_squared * Q33 * Q33)));

		// *** *** *** *** ***

		float *base_address = &tmp[(threadIdx.x * 6) + (threadIdx.x >> 4)];

		// minX, minY, minZ
		base_address[0] = GC_2.x - tmpX;
		base_address[1] = GC_2.y - tmpY;
		base_address[2] = GC_2.z - tmpZ;

		// maxX, maxY, maxZ
		base_address[3] = GC_2.x + tmpX;
		base_address[4] = GC_2.y + tmpY;
		base_address[5] = GC_2.z + tmpZ;
	}

	// *** *** *** *** ***

	__syncthreads();

	// *** *** *** *** ***

	if (wid <= number_of_warps) {
		int lane_id = threadIdx.x & 31;

		float *base_address_1 = &AABBs[(tid & -32) * 6];
		float *base_address_2 = &tmp  [((threadIdx.x & -32) * 6) + ((threadIdx.x & -32) >> 4)];

		base_address_1[lane_id      ] = base_address_2[lane_id      ];
		base_address_1[lane_id + 32 ] = base_address_2[lane_id + 32 ];
		base_address_1[lane_id + 64 ] = base_address_2[lane_id + 64 ];

		base_address_1[lane_id + 96 ] = base_address_2[lane_id + 96  + 1];
		base_address_1[lane_id + 128] = base_address_2[lane_id + 128 + 1];
		base_address_1[lane_id + 160] = base_address_2[lane_id + 160 + 1];
	}
}

// *************************************************************************************************

__device__ OptixAabb SReductionOperator_OptixAabb::operator()(const OptixAabb &a, const OptixAabb &b) const {
	OptixAabb result;
		
	result.minX = (a.minX <= b.minX) ? a.minX : b.minX;
	result.minY = (a.minY <= b.minY) ? a.minY : b.minY;
	result.minZ = (a.minZ <= b.minZ) ? a.minZ : b.minZ;

	result.maxX = (a.maxX >= b.maxX) ? a.maxX : b.maxX;
	result.maxY = (a.maxY >= b.maxY) ? a.maxY : b.maxY;
	result.maxZ = (a.maxZ >= b.maxZ) ? a.maxZ : b.maxZ;

	return result;
};

// *************************************************************************************************

__global__ void MultiplyPointwiseReal(REAL_G *arr1_in, REAL_G *arr2_in, REAL_G *arr_out, int size) {
	int tid = (blockIdx.x * blockDim.x) + threadIdx.x;
	if (tid < size) {
		REAL_G a = arr1_in[tid];
		REAL_G b = arr2_in[tid];
		REAL_G c = a * b;
		arr_out[tid] = c;
	}
}

// *************************************************************************************************

__global__ void MultiplyPointwiseComplex(COMPLEX_G *arr1_in, COMPLEX_G *arr2_in, COMPLEX_G *arr_out, int size) {
	int tid = (blockIdx.x * blockDim.x) + threadIdx.x;
	if (tid < size) {
		COMPLEX_G c;
		COMPLEX_G a = arr1_in[tid];
		COMPLEX_G b = arr2_in[tid];
		c.x = MAD_G(-a.y, b.y, a.x * b.x);
		c.y = MAD_G( a.x, b.y, a.y * b.x);
		arr_out[tid] = c;
	}
}

// *************************************************************************************************

__device__ float RandomFloat(unsigned n) {
	const unsigned a = 1664525;
	const unsigned c = 1013904223;

	unsigned tmp1 = 1;
	unsigned tmp2 = a;
	unsigned tmp3 = 0;
	while (n != 0) {
		if ((n & 1) != 0) tmp3 = (tmp2 * tmp3) + tmp1;
		tmp1 = (tmp2 * tmp1) + tmp1;
		tmp2 = tmp2 * tmp2;
		n >>= 1;
	}
	float result = __uint_as_float(1065353216 | ((tmp3 * c) & 8388607)) - 1.0f;
	return result;
}

// *************************************************************************************************

__device__ void RandomNormalFloat(unsigned n, float& Z1, float& Z2) {
	float U1 = RandomFloat(n);
	float U2 = RandomFloat(n + 1);

	float tmp1 = sqrtf(-2.0f * __logf(1.0f - U1)); // !!! !!! !!!
	float tmp2 = 2.0f * M_PI * U2;

	float sine;
	float cosine;
	__sincosf(tmp2, &sine, &cosine);

	Z1 = tmp1 * cosine;
	Z2 = tmp1 * sine;
}

// *************************************************************************************************

__device__ void ComputeRotationMatrix(
	float4 q,

	float &R11, float &R12, float &R13,
	float &R21, float &R22, float &R23,
	float &R31, float &R32, float &R33
) {
	float aa = q.x * q.x;
	float bb = q.y * q.y;
	float cc = q.z * q.z;
	float dd = q.w * q.w;

	float s = __fdividef(2.0f, aa + bb + cc + dd);

	float bs = q.y * s;  float cs = q.z * s;  float ds = q.w * s;
	float ab = q.x * bs; float ac = q.x * cs; float ad = q.x * ds;
	bb = bb * s;         float bc = q.y * cs; float bd = q.y * ds;
	cc = cc * s;         float cd = q.z * ds;       dd = dd * s;

	R11 = 1.0f - cc - dd;
	R12 = bc - ad;
	R13 = bd + ac;

	R21 = bc + ad;
	R22 = 1.0f - bb - dd;
	R23 = cd - ab;

	R31 = bd - ac;
	R32 = cd + ab;
	R33 = 1.0f - bb - cc;
}

// *************************************************************************************************

/*
Z1 ~ N(0, 1)
Z2 ~ N(0, 1)
Z3 ~ N(0, 1)
*/
__device__ float3 RandomMultinormalFloat(
	float3 m,
	float3 scale,

	float R11, float R12, float R13,
	float R21, float R22, float R23,
	float R31, float R32, float R33,

	float Z1, float Z2, float Z3
) {
	Z1 *= scale.x;
	Z2 *= scale.y;
	Z3 *= scale.z;

	return make_float3(
		m.x + __fmaf_rn(R11, Z1, __fmaf_rn(R12, Z2, R13 * Z3)),
		m.y + __fmaf_rn(R21, Z1, __fmaf_rn(R22, Z2, R23 * Z3)),
		m.z + __fmaf_rn(R31, Z1, __fmaf_rn(R32, Z2, R33 * Z3))
	);
}