#include <optix_device.h>

#include "Header.cuh"







// *************************************************************************************************

struct SLight {
	float Ox, Oy, Oz;
	float R, G, B;
};

struct STriangleComponent {
	float3 N1, N2, N3;
	int material_index;
};

struct SMaterial {
	float Ra, Ga, Ba;
	float Rd, Gd, Bd;
	float shininess;
	float kt;
	float n;
};

struct LaunchParamsMesh {
	unsigned *bitmap;
	unsigned width;
	unsigned height;

	float3 O;
	float3 R, D, F;
	float double_tan_half_fov_x;
	float double_tan_half_fov_y;

	OptixTraversableHandle traversable;

	float4 *GC_part_1;
	float4 *GC_part_2;
	float4 *GC_part_3;
	float2 *GC_part_4;

	int *Gaussians_indices;
	REAL_G *bitmap_out_R, *bitmap_out_G, *bitmap_out_B;

	float ray_termination_T_threshold;
	float last_significant_Gauss_alpha_gradient_precision;
	float chi_square_squared_radius;
	int max_Gaussians_per_ray;

	SLight *light;
	STriangleComponent *TC;
	SMaterial *materials;
	OptixTraversableHandle traversable_tri;
};

// *************************************************************************************************

extern "C" __constant__ LaunchParamsMesh optixLaunchParams;

// *************************************************************************************************

// !!! !!! !!!

// !!! !!! !!!
extern "C" __global__ void __raygen__renderFrame() {
	int x = optixGetLaunchIndex().x;
	int y = optixGetLaunchIndex().y;
	int pixel_ind = (y * optixLaunchParams.width) + x;

	float3 d = make_float3(
		(-0.5f + ((x + 0.5f) / optixLaunchParams.width)) * optixLaunchParams.double_tan_half_fov_x,
		(-0.5f + ((y + 0.5f) / optixLaunchParams.height)) * optixLaunchParams.double_tan_half_fov_y,
		1.0f
	);
	/*float3 d = make_float3(
		-0.5f + ((x + 0.5f) / optixLaunchParams.width),
		-0.5f + ((y + 0.5f) / optixLaunchParams.height),
		0.5f / tanf(0.5f * optixLaunchParams.FOV)
	);*/
	float3 v = make_float3(
		__fmaf_rn(optixLaunchParams.R.x, d.x, __fmaf_rn(optixLaunchParams.D.x, d.y, optixLaunchParams.F.x * d.z)),
		__fmaf_rn(optixLaunchParams.R.y, d.x, __fmaf_rn(optixLaunchParams.D.y, d.y, optixLaunchParams.F.y * d.z)),
		__fmaf_rn(optixLaunchParams.R.z, d.x, __fmaf_rn(optixLaunchParams.D.z, d.y, optixLaunchParams.F.z * d.z))
	);

	float v_norm_inv = __frsqrt_rn(__fmaf_rn(v.x, v.x, __fmaf_rn(v.y, v.y, v.z * v.z)));
	v.x *= v_norm_inv; v.y *= v_norm_inv; v.z *= v_norm_inv;
		
	float tMin = 0.0f;

	unsigned t_as_uint;

	unsigned R_as_uint = __float_as_uint(0.0f);
	unsigned G_as_uint = __float_as_uint(0.0f);
	unsigned B_as_uint = __float_as_uint(0.0f);
	
	unsigned alpha_as_uint;
	
	unsigned run_from_CH_for_Gaussians = 0;

	unsigned recursion_depth = 1;

	optixTrace(
		optixLaunchParams.traversable,
		optixLaunchParams.O,
		v,
		tMin, // tmin
		INFINITY, // tmax
		0.0f, // rayTime
		OptixVisibilityMask(255),
		OPTIX_RAY_FLAG_NONE,
		0,
		1,
		0,

		t_as_uint,
		R_as_uint, G_as_uint, B_as_uint,
		alpha_as_uint,
		run_from_CH_for_Gaussians,
		recursion_depth
	);

	float R = __saturatef(__uint_as_float(R_as_uint));
	float G = __saturatef(__uint_as_float(G_as_uint));
	float B = __saturatef(__uint_as_float(B_as_uint));
	int Ri = __roundf(R * 255.0f);
	int Gi = __roundf(G * 255.0f);
	int Bi = __roundf(B * 255.0f);
	optixLaunchParams.bitmap[pixel_ind] = (Ri << 16) + (Gi << 8) + Bi;
}

// *********************************************************************************************

extern "C" __global__ void __closesthit__radiance() {
	unsigned recursion_depth = optixGetPayload_6();

	if (recursion_depth == 16) return; // !!! !!! !!!

	++recursion_depth;

	// *********************************************************************************************

	unsigned Gauss_ind = optixGetPrimitiveIndex();
		
	// *********************************************************************************************

	float4 GC_1 = optixLaunchParams.GC_part_1[Gauss_ind];
	float4 GC_2 = optixLaunchParams.GC_part_2[Gauss_ind];
	float4 GC_3 = optixLaunchParams.GC_part_3[Gauss_ind];
	float2 GC_4 = optixLaunchParams.GC_part_4[Gauss_ind];

	// *********************************************************************************************

	float aa = GC_3.z * GC_3.z;
	float bb = GC_3.w * GC_3.w;
	float cc = GC_4.x * GC_4.x;
	float dd = GC_4.y * GC_4.y;
	float s = 0.5f * (aa + bb + cc + dd);

	float ab = GC_3.z * GC_3.w; float ac = GC_3.z * GC_4.x; float ad = GC_3.z * GC_4.y;
								float bc = GC_3.w * GC_4.x; float bd = GC_3.w * GC_4.y;
														    float cd = GC_4.x * GC_4.y;

	float R11 = s - cc - dd;
	float R12 = bc - ad;
	float R13 = bd + ac;

	float R21 = bc + ad;
	float R22 = s - bb - dd;
	float R23 = cd - ab;

	float R31 = bd - ac;
	float R32 = cd + ab;
	float R33 = s - bb - cc;

	// *********************************************************************************************

	float3 O = optixGetWorldRayOrigin();
	float3 v = optixGetWorldRayDirection();

	float t = optixGetRayTmax();

	O.x = O.x - GC_2.x;
	O.y = O.y - GC_2.y;
	O.z = O.z - GC_2.z;

	float3 O_prim;
	float3 v_prim;

	float sXInv = 1.0f + expf(-GC_2.w);
	O_prim.x = __fmaf_rn(R11, O.x, __fmaf_rn(R21, O.y, R31 * O.z)) * sXInv;
	v_prim.x = __fmaf_rn(R11, v.x, __fmaf_rn(R21, v.y, R31 * v.z)) * sXInv;

	float sYInv = 1.0f + expf(-GC_3.x);
	O_prim.y = __fmaf_rn(R12, O.x, __fmaf_rn(R22, O.y, R32 * O.z)) * sYInv;
	v_prim.y = __fmaf_rn(R12, v.x, __fmaf_rn(R22, v.y, R32 * v.z)) * sYInv;

	float sZInv = 1.0f + expf(-GC_3.y);
	O_prim.z = __fmaf_rn(R13, O.x, __fmaf_rn(R23, O.y, R33 * O.z)) * sZInv;
	v_prim.z = __fmaf_rn(R13, v.x, __fmaf_rn(R23, v.y, R33 * v.z)) * sZInv;

	// *********************************************************************************************

	float v_dot_v = __fmaf_rn(v_prim.x, v_prim.x, __fmaf_rn(v_prim.y, v_prim.y, v_prim.z * v_prim.z));
	float v_dot_O = __fmaf_rn(v_prim.x, O_prim.x, __fmaf_rn(v_prim.y, O_prim.y, v_prim.z * O_prim.z));
	float tmp = v_dot_O / v_dot_v;

	float O_perp_x = __fmaf_rn(-v_prim.x, tmp, O_prim.x);
	float O_perp_y = __fmaf_rn(-v_prim.y, tmp, O_prim.y);
	float O_perp_z = __fmaf_rn(-v_prim.z, tmp, O_prim.z);

	float alpha = __saturatef(expf(-0.5f * (__fmaf_rn(O_perp_x, O_perp_x, __fmaf_rn(O_perp_y, O_perp_y, O_perp_z * O_perp_z)) / (s * s))) / (1.0f + expf(-GC_1.w)));

	// *********************************************************************************************

	// !!! !!! !!!
	/*const int lightsNum = 1;
	const float shininess = 128.0f;

	struct SLight {
		float3 O;
		float R, G, B;
	} lights[lightsNum] = { make_float3(200.0f, 0.0f, -200.0f), 10000.0f, 10000.0f, 10000.0f };*/
	// !!! !!! !!!

	// *********************************************************************************************

	if (!optixGetPayload_5()) {
		SLight light = optixLaunchParams.light[0];
		
		O = optixGetWorldRayOrigin();

		float t = nextafter(optixGetRayTmax(), INFINITY);

		// *** *** *** *** ***

		float v_prim_norm_inv;

		unsigned t_as_uint;
		unsigned R_as_uint;
		unsigned G_as_uint;
		unsigned B_as_uint;
		unsigned alpha_as_uint;
		unsigned run_from_CH_for_Gaussians;

		float tmp;

		float R;
		float G;
		float B;
		float T = 1 - alpha;

		// *** *** *** *** ***

		O_prim = make_float3(
			__fmaf_rn(v.x, t, O.x),
			__fmaf_rn(v.y, t, O.y),
			__fmaf_rn(v.z, t, O.z)
		);
		v_prim = make_float3(
			light.Ox - O_prim.x,
			light.Oy - O_prim.y,
			light.Oz - O_prim.z
		);
		v_prim_norm_inv = __frsqrt_rn(__fmaf_rn(v_prim.x, v_prim.x, __fmaf_rn(v_prim.y, v_prim.y, v_prim.z * v_prim.z)));
		v_prim.x *= v_prim_norm_inv; v_prim.y *= v_prim_norm_inv; v_prim.z *= v_prim_norm_inv;

		t_as_uint = __float_as_uint(0.0f);
		
		optixTrace(
			optixLaunchParams.traversable_tri,
			O_prim,
			v_prim,
			0.0001f, // tmin
			__frcp_rn(v_prim_norm_inv), // tmax
			0.0f, // rayTime
			OptixVisibilityMask(255),
			OPTIX_RAY_FLAG_TERMINATE_ON_FIRST_HIT | OPTIX_RAY_FLAG_DISABLE_CLOSESTHIT,
			0,
			1,
			0,

			t_as_uint,
			R_as_uint, G_as_uint, B_as_uint,
			alpha_as_uint,
			run_from_CH_for_Gaussians,
			recursion_depth
		);
		
		tmp = ((__uint_as_float(t_as_uint) != INFINITY) ? (0.25f * alpha) : alpha);
		R = GC_1.x * tmp;
		G = GC_1.y * tmp;
		B = GC_1.z * tmp;
		
		// *** *** *** *** ***

		for (int i = 1; i < 128; ++i) {
			R_as_uint = __float_as_uint(0.0f);
			G_as_uint = __float_as_uint(0.0f);
			B_as_uint = __float_as_uint(0.0f);
			alpha_as_uint = __float_as_uint(1.0f);
			run_from_CH_for_Gaussians = 1;

			t_as_uint = __float_as_uint(INFINITY);

			optixTrace(
				optixLaunchParams.traversable,
				optixGetWorldRayOrigin(),
				v,
				t, // tmin
				INFINITY, // tmax
				0.0f, // rayTime
				OptixVisibilityMask(255),
				OPTIX_RAY_FLAG_NONE,
				0,
				1,
				0,

				t_as_uint,
				R_as_uint, G_as_uint, B_as_uint,
				alpha_as_uint,
				run_from_CH_for_Gaussians,
				recursion_depth
			);

			t = __uint_as_float(t_as_uint);
			alpha = __uint_as_float(alpha_as_uint);

			// *** *** *** *** ***

			if (t != INFINITY) {
				unsigned R_as_uint_tmp = R_as_uint; // !!! !!! !!!
				unsigned G_as_uint_tmp = G_as_uint; // !!! !!! !!!
				unsigned B_as_uint_tmp = B_as_uint; // !!! !!! !!!

				O_prim = make_float3(
					__fmaf_rn(v.x, t, O.x),
					__fmaf_rn(v.y, t, O.y),
					__fmaf_rn(v.z, t, O.z)
				);
				v_prim = make_float3(
					light.Ox - O_prim.x,
					light.Oy - O_prim.y,
					light.Oz - O_prim.z
				);
				v_prim_norm_inv = __frsqrt_rn(__fmaf_rn(v_prim.x, v_prim.x, __fmaf_rn(v_prim.y, v_prim.y, v_prim.z * v_prim.z)));
				v_prim.x *= v_prim_norm_inv; v_prim.y *= v_prim_norm_inv; v_prim.z *= v_prim_norm_inv;
			
				t_as_uint = __float_as_uint(0.0f);

				optixTrace(
					optixLaunchParams.traversable_tri,
					O_prim,
					v_prim,
					0.0001f, // tmin
					__frcp_rn(v_prim_norm_inv), // tmax
					0.0f, // rayTime
					OptixVisibilityMask(255),
					OPTIX_RAY_FLAG_TERMINATE_ON_FIRST_HIT | OPTIX_RAY_FLAG_DISABLE_CLOSESTHIT,
					0,
					1,
					0,

					t_as_uint,
					R_as_uint, G_as_uint, B_as_uint,
					alpha_as_uint,
					run_from_CH_for_Gaussians,
					recursion_depth
				);
			
				tmp = ((__uint_as_float(t_as_uint) != INFINITY) ? (0.25f * T) : T);
				R = __fmaf_rn(tmp, __uint_as_float(R_as_uint_tmp), R);
				G = __fmaf_rn(tmp, __uint_as_float(G_as_uint_tmp), G);
				B = __fmaf_rn(tmp, __uint_as_float(B_as_uint_tmp), B);
			} else {
				R = __fmaf_rn(T, __uint_as_float(R_as_uint), R);
				G = __fmaf_rn(T, __uint_as_float(G_as_uint), G);
				B = __fmaf_rn(T, __uint_as_float(B_as_uint), B);
			}
			
			// *** *** *** *** ***

			if ((t == INFINITY) || ((T * (1.0f - alpha)) < (0.1f / 255.0f))) break;
			t = nextafter(t, INFINITY);
			
			T = T * (1.0f - alpha);
		}

		R_as_uint = __float_as_uint(R);
		G_as_uint = __float_as_uint(G);
		B_as_uint = __float_as_uint(B);

		optixSetPayload_1(R_as_uint);
		optixSetPayload_2(G_as_uint);
		optixSetPayload_3(B_as_uint);
	} else {
		optixSetPayload_0(__float_as_uint(optixGetRayTmax()));
		optixSetPayload_1(__float_as_uint(GC_1.x * alpha));
		optixSetPayload_2(__float_as_uint(GC_1.y * alpha));
		optixSetPayload_3(__float_as_uint(GC_1.z * alpha));
		optixSetPayload_4(__float_as_uint(alpha));
	}
}

// *********************************************************************************************

__device__ float3 Reflect(float3 N, float3 v, float cos_theta_1) {
	float tmp = 2.0f * cos_theta_1;
	return make_float3(
		__fmaf_rn(-N.x, tmp, v.x),
		__fmaf_rn(-N.y, tmp, v.y),
		__fmaf_rn(-N.z, tmp, v.z)
	);
}

// *********************************************************************************************

__device__ bool GetCosTheta2ForRefraction(float cos_theta_1, float n_inv, float &cos_theta_2) {
	float cos_theta_2_squared = __fmaf_rn(
		-n_inv * n_inv,
		((__fmaf_rn(-cos_theta_1, cos_theta_1, 1.0f) < 0.0f) ? 0.0f : __fmaf_rn(-cos_theta_1, cos_theta_1, 1.0f)),
		1.0f
	);
	if (cos_theta_2_squared < 0.0f) return true;
	else {
		cos_theta_2 = __fsqrt_rn(cos_theta_2_squared);
		return false;
	}
}

// *********************************************************************************************

__device__ float GetFresnelFactor(float cos_theta_1, float n, float cos_theta_2) {
	float tmp1 = (1.0f - n) / (1.0f + n);
	float R0 = tmp1 * tmp1;
	float tmp2 = 1.0f - ((n <= 1.0f) ? cos_theta_2 : cos_theta_1);
	float tmp3 = tmp2 * tmp2;
	tmp3 = tmp3 * tmp3;
	tmp3 = tmp2 * tmp3;
	return __fmaf_rn(1.0f - R0, tmp3, R0);
}

// *********************************************************************************************

__device__ float3 Refract(
	float3 N,
	float3 v,
	float n_inv,
	float cos_theta_1, float cos_theta_2
) {
	float tmp = __fmaf_rn(-cos_theta_1, n_inv, copysignf(cos_theta_2, cos_theta_1));
	return make_float3(
		(v.x * n_inv) + (N.x * tmp),
		(v.y * n_inv) + (N.y * tmp),
		(v.z * n_inv) + (N.z * tmp)
	);
}

// *************************************************************************************************

extern "C" __global__ void __intersection__is() {
	int index = optixGetPrimitiveIndex();

	float4 GC_1 = optixLaunchParams.GC_part_1[index];
	float4 GC_2 = optixLaunchParams.GC_part_2[index];
	float4 GC_3 = optixLaunchParams.GC_part_3[index];
	float2 GC_4 = optixLaunchParams.GC_part_4[index];

	// *********************************************************************************************

	float aa = GC_3.z * GC_3.z;
	float bb = GC_3.w * GC_3.w;
	float cc = GC_4.x * GC_4.x;
	float dd = GC_4.y * GC_4.y;
	float s = 0.5f * (aa + bb + cc + dd);

	float ab = GC_3.z * GC_3.w; float ac = GC_3.z * GC_4.x; float ad = GC_3.z * GC_4.y;
								float bc = GC_3.w * GC_4.x; float bd = GC_3.w * GC_4.y;
															float cd = GC_4.x * GC_4.y;

	float R11 = s - cc - dd;
	float R12 = bc - ad;
	float R13 = bd + ac;

	float R21 = bc + ad;
	float R22 = s - bb - dd;
	float R23 = cd - ab;

	float R31 = bd - ac;
	float R32 = cd + ab;
	float R33 = s - bb - cc;

	// *********************************************************************************************

	float3 O = optixGetObjectRayOrigin();
	float3 v = optixGetObjectRayDirection();

	O.x = O.x - GC_2.x;
	O.y = O.y - GC_2.y;
	O.z = O.z - GC_2.z;

	float sXInv = 1.0f + expf(-GC_2.w);
	float Ox_prim = __fmaf_rn(R11, O.x, __fmaf_rn(R21, O.y, R31 * O.z)) * sXInv;
	float vx_prim = __fmaf_rn(R11, v.x, __fmaf_rn(R21, v.y, R31 * v.z)) * sXInv;

	float sYInv = 1.0f + expf(-GC_3.x);
	float Oy_prim = __fmaf_rn(R12, O.x, __fmaf_rn(R22, O.y, R32 * O.z)) * sYInv;
	float vy_prim = __fmaf_rn(R12, v.x, __fmaf_rn(R22, v.y, R32 * v.z)) * sYInv;

	float sZInv = 1.0f + expf(-GC_3.y);
	float Oz_prim = __fmaf_rn(R13, O.x, __fmaf_rn(R23, O.y, R33 * O.z)) * sZInv;
	float vz_prim = __fmaf_rn(R13, v.x, __fmaf_rn(R23, v.y, R33 * v.z)) * sZInv;

	// *********************************************************************************************

	float v_dot_v = __fmaf_rn(vx_prim, vx_prim, __fmaf_rn(vy_prim, vy_prim, vz_prim * vz_prim));
	float O_dot_v = __fmaf_rn(Ox_prim, vx_prim, __fmaf_rn(Oy_prim, vy_prim, Oz_prim * vz_prim));
	float O_dot_O = __fmaf_rn(Ox_prim, Ox_prim, __fmaf_rn(Oy_prim, Oy_prim, Oz_prim * Oz_prim));

	float tmp1 = 1.0f / v_dot_v;
	float tmp2 = O_dot_v * tmp1;

	float O_perp_x = __fmaf_rn(-vx_prim, tmp2, Ox_prim);
	float O_perp_y = __fmaf_rn(-vy_prim, tmp2, Oy_prim);
	float O_perp_z = __fmaf_rn(-vz_prim, tmp2, Oz_prim);

	float delta = -__fmaf_rn(O_perp_x, O_perp_x, __fmaf_rn(O_perp_y, O_perp_y, __fmaf_rn(O_perp_z, O_perp_z, -optixLaunchParams.chi_square_squared_radius * s * s))); // !!! !!! !!!
	if (delta >= 0.0f) {
		float t = -(O_dot_v + copysignf(__fsqrt_rn(v_dot_v * delta), O_dot_v)) * tmp1;
		float t1 = ((O_dot_v <= 0.0f) ? (O_dot_O - (optixLaunchParams.chi_square_squared_radius * s * s)) / (v_dot_v * t) : t);
		optixReportIntersection(
			t1,
			0
		);
	}
}

// *********************************************************************************************

extern "C" __global__ void __closesthit__radiance2() {
	float3 O = optixGetWorldRayOrigin();
	float3 v = optixGetWorldRayDirection();
	float t = optixGetRayTmax();

	O.x = __fmaf_rn(v.x, t, O.x);
	O.y = __fmaf_rn(v.y, t, O.y);
	O.z = __fmaf_rn(v.z, t, O.z);

	unsigned tri_ind = optixGetPrimitiveIndex();
	STriangleComponent TC = optixLaunchParams.TC[tri_ind];

	float2 texCoords = optixGetTriangleBarycentrics();
	float3 N1 = TC.N1;
	float3 N2 = TC.N2;
	float3 N3 = TC.N3;
	float3 N = make_float3(
		__fmaf_rn(N2.x - N1.x, texCoords.x, __fmaf_rn(N3.x - N1.x, texCoords.y, N1.x)),
		__fmaf_rn(N2.y - N1.y, texCoords.x, __fmaf_rn(N3.y - N1.y, texCoords.y, N1.y)),
		__fmaf_rn(N2.z - N1.z, texCoords.x, __fmaf_rn(N3.z - N1.z, texCoords.y, N1.z))
	);
	float N_norm_inv = __frsqrt_rn(__fmaf_rn(N.x, N.x, __fmaf_rn(N.y, N.y, N.z * N.z)));
	N.x *= N_norm_inv; N.y *= N_norm_inv; N.z *= N_norm_inv;

	SMaterial material = optixLaunchParams.materials[TC.material_index];

	float R = 0.0f;
	float G = 0.0f;
	float B = 0.0f;

	// *********************************************************************************************

	float cos_theta_1 = __fmaf_rn(N.x, v.x, __fmaf_rn(N.y, v.y, N.z * v.z));
	float n = ((cos_theta_1 <= 0.0f) ? material.n : (1.0f / material.n));
	float n_inv = ((cos_theta_1 <= 0.0f) ? (1.0f / material.n) : material.n);
	float cos_theta_2;
	float F1;
	bool TIR = GetCosTheta2ForRefraction(cos_theta_1, n_inv, cos_theta_2);
	if (!TIR)
		F1 = GetFresnelFactor(fabsf(cos_theta_1), n, cos_theta_2);
	else
		F1 = 1.0f;

	// *********************************************************************************************

	unsigned t_as_uint;
	unsigned R_as_uint;
	unsigned G_as_uint;
	unsigned B_as_uint;
	unsigned alpha_as_uint;
	unsigned run_from_CH_for_Gaussians = 0;
	unsigned recursion_depth = optixGetPayload_6() + 1;

	// *********************************************************************************************

	for (int i = 0; i < 1; ++i) {
		SLight light = optixLaunchParams.light[0];

		float3 L = make_float3(light.Ox - O.x, light.Oy - O.y, light.Oz - O.z);
		float L_norm_squared = (L.x * L.x) + (L.y * L.y) + (L.z * L.z);
		float L_norm = __fsqrt_rn(L_norm_squared);
		L.x /= L_norm; L.y /= L_norm; L.z /= L_norm;

		// *** *** *** *** ***

		t_as_uint = __float_as_uint(0.0f);

		optixTrace(
			optixLaunchParams.traversable_tri,
			O,
			L,
			0.0001f, // tmin
			L_norm, // tmax
			0.0f, // rayTime
			OptixVisibilityMask(255),
			OPTIX_RAY_FLAG_TERMINATE_ON_FIRST_HIT | OPTIX_RAY_FLAG_DISABLE_CLOSESTHIT,
			0,
			1,
			0,

			t_as_uint,
			R_as_uint, G_as_uint, B_as_uint,
			alpha_as_uint,
			run_from_CH_for_Gaussians,
			recursion_depth
		);

		if (__uint_as_float(t_as_uint) == INFINITY) {
			float3 H = make_float3(L.x - v.x, L.y - v.y, L.z - v.z);
			float H_norm = __fsqrt_rn((H.x * H.x) + (H.y * H.y) + (H.z * H.z));
			H.x /= H_norm; H.y /= H_norm; H.z /= H_norm;

			float N_dot_L = (N.x * L.x) + (N.y * L.y) + (N.z * L.z);
			float n = ((-N_dot_L <= 0.0f) ? material.n : (1.0f / material.n));
			float n_inv = ((-N_dot_L <= 0.0f) ? (1.0f / material.n) : material.n);
			float cos_theta_2;
			float F2;
		
			if ((N_dot_L >= 0.0f) && (cos_theta_1 <= 0.0f)) {
				bool TIR = GetCosTheta2ForRefraction(-N_dot_L, n_inv, cos_theta_2);
				if (!TIR)
					F2 = GetFresnelFactor(N_dot_L, n, cos_theta_2);
				else
					F2 = 1.0f;

				float tmp1 = ((1.0f - F2) * N_dot_L) / L_norm_squared;

				float N_dot_H = fabsf((N.x * H.x) + (N.y * H.y) + (N.z * H.z));
				float tmp2 = (F2 * __saturatef(N_dot_H / (material.shininess - (material.shininess * N_dot_H) + N_dot_H))) / L_norm_squared;

				R = R + (light.R * material.Rd * (1.0f - material.kt) * tmp1);
				G = G + (light.R * material.Gd * (1.0f - material.kt) * tmp1);
				B = B + (light.R * material.Bd * (1.0f - material.kt) * tmp1);

				R = R + (light.R * tmp2);
				G = G + (light.G * tmp2);
				B = B + (light.B * tmp2);
			}
		}
	}

	// *********************************************************************************************
	
	float3 v_refl = Reflect(N, v, cos_theta_1);

	if (recursion_depth < 4) {
		R_as_uint = __float_as_uint(0.0f);
		G_as_uint = __float_as_uint(0.0f);
		B_as_uint = __float_as_uint(0.0f);
		
		optixTrace(
			optixLaunchParams.traversable,
			O,
			v_refl,
			0.0001f, // tmin
			INFINITY, // tmax
			0.0f, // rayTime
			OptixVisibilityMask(255),
			OPTIX_RAY_FLAG_NONE,
			0,
			1,
			0,

			t_as_uint,
			R_as_uint, G_as_uint, B_as_uint,
			alpha_as_uint,
			run_from_CH_for_Gaussians,
			recursion_depth
		);
	
		R = __fmaf_rn(F1, __uint_as_float(R_as_uint), R);
		G = __fmaf_rn(F1, __uint_as_float(G_as_uint), G);
		B = __fmaf_rn(F1, __uint_as_float(B_as_uint), B);
	}

	// *********************************************************************************************

	if (F1 < 1.0f) {
		float3 v_refr = Refract(N, v, n_inv, cos_theta_1, cos_theta_2);

		if (recursion_depth < 16) {
			R_as_uint = __float_as_uint(0.0f);
			G_as_uint = __float_as_uint(0.0f);
			B_as_uint = __float_as_uint(0.0f);
			
			optixTrace(
				optixLaunchParams.traversable,
				O,
				v_refr,
				0.0001f, // tmin
				INFINITY, // tmax
				0.0f, // rayTime
				OptixVisibilityMask(255),
				OPTIX_RAY_FLAG_NONE,
				0,
				1,
				0,

				t_as_uint,
				R_as_uint, G_as_uint, B_as_uint,
				alpha_as_uint,
				run_from_CH_for_Gaussians,
				recursion_depth
			);

			R = __fmaf_rn((1.0f - 0.125f) * (1.0f - F1), __uint_as_float(R_as_uint), R);
			G = __fmaf_rn((1.0f - 0.125f) * (1.0f - F1), __uint_as_float(G_as_uint), G);
			B = __fmaf_rn((1.0f - 0.125f) * (1.0f - F1), __uint_as_float(B_as_uint), B);
		}
	}

	// *********************************************************************************************

	R_as_uint = __float_as_uint(R);
	G_as_uint = __float_as_uint(G);
	B_as_uint = __float_as_uint(B);

	optixSetPayload_1(R_as_uint);
	optixSetPayload_2(G_as_uint);
	optixSetPayload_3(B_as_uint);
}

// *************************************************************************************************

extern "C" __global__ void __miss__radiance() {
	optixSetPayload_0(__float_as_uint(INFINITY));
}