#include <optix_device.h>

#include "Header.cuh"

// *************************************************************************************************

struct LaunchParams {
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
};

// *************************************************************************************************

extern "C" __constant__ LaunchParams optixLaunchParams;

// *************************************************************************************************

extern "C" __global__ void __raygen__renderFrame() {
	int x = optixGetLaunchIndex().x;
	int y = optixGetLaunchIndex().y;
	int pixel_ind = (y * optixLaunchParams.width) + x;
	
	REAL3_R d = make_REAL3_R(
		(((REAL_R)-0.5) + ((x + ((REAL_R)0.5)) / optixLaunchParams.width)) * optixLaunchParams.double_tan_half_fov_x,
		(((REAL_R)-0.5) + ((y + ((REAL_R)0.5)) / optixLaunchParams.height)) * optixLaunchParams.double_tan_half_fov_y,
		1,
	);
	REAL3_R v = make_REAL3_R(
		MAD_R(optixLaunchParams.R.x, d.x, MAD_R(optixLaunchParams.D.x, d.y, optixLaunchParams.F.x * d.z)),
		MAD_R(optixLaunchParams.R.y, d.x, MAD_R(optixLaunchParams.D.y, d.y, optixLaunchParams.F.y * d.z)),
		MAD_R(optixLaunchParams.R.z, d.x, MAD_R(optixLaunchParams.D.z, d.y, optixLaunchParams.F.z * d.z))
	);

	float tMin = 0.0f;

	REAL_R R = 0;
	REAL_R G = 0;
	REAL_R B = 0;

	REAL_R T = 1;
	REAL_R TLastGrad = 1;

	bool belowThreshold = false;

	// !!! !!! !!!
	//int i = 0;
	//for (i = 0; i < optixLaunchParams.max_Gaussians_per_ray; ++i) {
	// !!! !!! !!!

	for (int i = 0; i < optixLaunchParams.max_Gaussians_per_ray; ++i) {
		unsigned Gauss_ind = -1;
		unsigned t_as_uint;
				
		optixTrace(
			optixLaunchParams.traversable,
			optixLaunchParams.O,

			#ifndef RENDERER_OPTIX_USE_DOUBLE_PRECISION
				v,
			#else
				make_float3(v.x, v.y, v.z),
			#endif

			tMin,
			INFINITY,
			0.0f,
			OptixVisibilityMask(255),
			OPTIX_RAY_FLAG_NONE,
			0,
			1,
			0,

			Gauss_ind,
			t_as_uint
		);
		
		optixLaunchParams.Gaussians_indices[(i * optixLaunchParams.width * optixLaunchParams.height) + pixel_ind] = Gauss_ind;
		if (Gauss_ind != -1) {
			float4 GC_1 = optixLaunchParams.GC_part_1[Gauss_ind];
			float4 GC_2 = optixLaunchParams.GC_part_2[Gauss_ind];
			float4 GC_3 = optixLaunchParams.GC_part_3[Gauss_ind];
			float2 GC_4 = optixLaunchParams.GC_part_4[Gauss_ind];

			// *************************************************************************************

			REAL_R aa = ((REAL_R)GC_3.z) * GC_3.z;
			REAL_R bb = ((REAL_R)GC_3.w) * GC_3.w;
			REAL_R cc = ((REAL_R)GC_4.x) * GC_4.x;
			REAL_R dd = ((REAL_R)GC_4.y) * GC_4.y;
			REAL_R s = ((REAL_R)0.5) * (aa + bb + cc + dd);

			REAL_R ab = GC_3.z * GC_3.w; REAL_R ac = GC_3.z * GC_4.x; REAL_R ad = GC_3.z * GC_4.y;
										 REAL_R bc = GC_3.w * GC_4.x; REAL_R bd = GC_3.w * GC_4.y;
																	  REAL_R cd = GC_4.x * GC_4.y;

			REAL_R R11 = s - cc - dd;
			REAL_R R12 = bc - ad;
			REAL_R R13 = bd + ac;

			REAL_R R21 = bc + ad;
			REAL_R R22 = s - bb - dd;
			REAL_R R23 = cd - ab;

			REAL_R R31 = bd - ac;
			REAL_R R32 = cd + ab;
			REAL_R R33 = s - bb - cc;

			// *************************************************************************************

			REAL3_R O = make_REAL3_R(
				optixLaunchParams.O.x - GC_2.x,
				optixLaunchParams.O.y - GC_2.y,
				optixLaunchParams.O.z - GC_2.z
			);

			REAL3_R O_prim;
			REAL3_R v_prim;

			REAL_R sXInv = EXP_R(-GC_2.w);
			O_prim.x = MAD_R(R11, O.x, MAD_R(R21, O.y, R31 * O.z)) * sXInv;
			v_prim.x = MAD_R(R11, v.x, MAD_R(R21, v.y, R31 * v.z)) * sXInv;

			REAL_R sYInv = EXP_R(-GC_3.x);
			O_prim.y = MAD_R(R12, O.x, MAD_R(R22, O.y, R32 * O.z)) * sYInv;
			v_prim.y = MAD_R(R12, v.x, MAD_R(R22, v.y, R32 * v.z)) * sYInv;

			REAL_R sZInv = EXP_R(-GC_3.y);
			O_prim.z = MAD_R(R13, O.x, MAD_R(R23, O.y, R33 * O.z)) * sZInv;
			v_prim.z = MAD_R(R13, v.x, MAD_R(R23, v.y, R33 * v.z)) * sZInv;

			// *************************************************************************************

			REAL_R v_dot_v = MAD_R(v_prim.x, v_prim.x, MAD_R(v_prim.y, v_prim.y, v_prim.z * v_prim.z));
			REAL_R O_dot_O = MAD_R(O_prim.x, O_prim.x, MAD_R(O_prim.y, O_prim.y, O_prim.z * O_prim.z));
			REAL_R v_dot_O = MAD_R(v_prim.x, O_prim.x, MAD_R(v_prim.y, O_prim.y, v_prim.z * O_prim.z));
			
			REAL_R tmp = v_dot_O / v_dot_v;
			REAL_R O_perp_x = MAD_R(-v_prim.x, tmp, O_prim.x); // !!! !!! !!!
			REAL_R O_perp_y = MAD_R(-v_prim.y, tmp, O_prim.y); // !!! !!! !!!
			REAL_R O_perp_z = MAD_R(-v_prim.z, tmp, O_prim.z); // !!! !!! !!!
			REAL_R alpha = EXP_R(-((REAL_R)0.5) * (MAD_R(O_perp_x, O_perp_x, MAD_R(O_perp_y, O_perp_y, O_perp_z * O_perp_z)) / (s * s)));
			#ifndef RENDERER_OPTIX_USE_DOUBLE_PRECISION
				alpha = __saturatef(alpha) / (((REAL_R)1.0) + EXP_R(-GC_1.w));
			#else
				alpha = (alpha < 0) ? 0 : alpha;
				alpha = (alpha > 1) ? 1 : alpha;
				alpha = alpha / (((REAL_R)1.0) + EXP_R(-GC_1.w));
			#endif
			
			tmp = T * alpha;

			R = R + (GC_1.x * tmp);
			G = G + (GC_1.y * tmp);
			B = B + (GC_1.z * tmp);
			T = T - tmp;

			if (T < ((REAL_R)optixLaunchParams.ray_termination_T_threshold)) {
				if (!belowThreshold) belowThreshold = true;
				else
					TLastGrad = TLastGrad * (1 - alpha);
			}
			if (TLastGrad >= ((REAL_R)optixLaunchParams.last_significant_Gauss_alpha_gradient_precision)) {					
				float t = __uint_as_float(t_as_uint);
				tMin = nextafter(t, INFINITY);
			} else {
				if (i < optixLaunchParams.max_Gaussians_per_ray - 1)
					optixLaunchParams.Gaussians_indices[((i + 1) * optixLaunchParams.width * optixLaunchParams.height) + pixel_ind] = -1;
				break;
			}
		} else
			break;
	}

	// !!! !!! !!!
	//R = ((REAL_R)i) / optixLaunchParams.max_Gaussians_per_ray;
	//G = ((REAL_R)i) / optixLaunchParams.max_Gaussians_per_ray;
	//B = ((REAL_R)i) / optixLaunchParams.max_Gaussians_per_ray;
	// !!! !!! !!!

	#ifndef RENDERER_OPTIX_USE_DOUBLE_PRECISION
		R = __saturatef(R);
		G = __saturatef(G);
		B = __saturatef(B);
	#else
		// Cannot use the code below due to the error in OptiX:
		//component = MIN_R(MAX_R(((REAL_R)component), ((REAL_R)0)), ((REAL_R)1));
		
		R = (R < 0.0) ? 0.0 : R;
		R = (R > 1.0) ? 1.0 : R;
		
		G = (G < 0.0) ? 0.0 : G;
		G = (G > 1.0) ? 1.0 : G;
		
		B = (B < 0.0) ? 0.0 : B;
		B = (B > 1.0) ? 1.0 : B;
	#endif
	int Ri = RINT_R(R * 255);
	int Gi = RINT_R(G * 255);
	int Bi = RINT_R(B * 255);

	optixLaunchParams.bitmap[pixel_ind] = (Ri << 16) + (Gi << 8) + Bi;
	optixLaunchParams.bitmap_out_R[(y * (optixLaunchParams.width + 11 - 1)) + x] = R; // !!! !!! !!!
	optixLaunchParams.bitmap_out_G[(y * (optixLaunchParams.width + 11 - 1)) + x] = G; // !!! !!! !!!
	optixLaunchParams.bitmap_out_B[(y * (optixLaunchParams.width + 11 - 1)) + x] = B; // !!! !!! !!!
}

// *************************************************************************************************

extern "C" __global__ void __anyhit__radiance() {
}

// *************************************************************************************************

extern "C" __global__ void __closesthit__radiance() {
	unsigned Gauss_ind = optixGetPrimitiveIndex();
	Gauss_ind /= (NUMBER_OF_SIDES - 2);
	optixSetPayload_0(Gauss_ind);
	optixSetPayload_1(__float_as_uint(optixGetRayTmax()));
}

// *************************************************************************************************

extern "C" __global__ void __intersection__is() {
	int index = optixGetPrimitiveIndex();

	float4 GC_2 = optixLaunchParams.GC_part_2[index];
	float4 GC_3 = optixLaunchParams.GC_part_3[index];
	float2 GC_4 = optixLaunchParams.GC_part_4[index];

	// *********************************************************************************************

	REAL_R aa = ((REAL_R)GC_3.z) * GC_3.z;
	REAL_R bb = ((REAL_R)GC_3.w) * GC_3.w;
	REAL_R cc = ((REAL_R)GC_4.x) * GC_4.x;
	REAL_R dd = ((REAL_R)GC_4.y) * GC_4.y;
	REAL_R s = ((REAL_R)0.5) * (aa + bb + cc + dd);

	REAL_R ab = ((REAL_R)GC_3.z) * GC_3.w; REAL_R ac = ((REAL_R)GC_3.z) * GC_4.x; REAL_R ad = ((REAL_R)GC_3.z) * GC_4.y;
										   REAL_R bc = ((REAL_R)GC_3.w) * GC_4.x; REAL_R bd = ((REAL_R)GC_3.w) * GC_4.y;
																				  REAL_R cd = ((REAL_R)GC_4.x) * GC_4.y;

	REAL_R R11 = s - cc - dd;
	REAL_R R12 = bc - ad;
	REAL_R R13 = bd + ac;

	REAL_R R21 = bc + ad;
	REAL_R R22 = s - bb - dd;
	REAL_R R23 = cd - ab;

	REAL_R R31 = bd - ac;
	REAL_R R32 = cd + ab;
	REAL_R R33 = s - bb - cc;

	// *********************************************************************************************

	float3 O = optixGetObjectRayOrigin();
	float3 v = optixGetObjectRayDirection();

	REAL_R Ox = ((REAL_R)O.x) - GC_2.x;
	REAL_R Oy = ((REAL_R)O.y) - GC_2.y;
	REAL_R Oz = ((REAL_R)O.z) - GC_2.z;

	// OLD INVERSE SIGMOID ACTIVATION FUNCTION FOR SCALE PARAMETERS
	/*REAL_R sXInv = 1 + EXP_R(-GC_2.w);
	REAL_R Ox_prim = MAD_R(R11, Ox, MAD_R(R21, Oy, R31 * Oz)) * sXInv;
	REAL_R vx_prim = MAD_R(R11, v.x, MAD_R(R21, v.y, R31 * v.z)) * sXInv;

	REAL_R sYInv = 1 + EXP_R(-GC_3.x);
	REAL_R Oy_prim = MAD_R(R12, Ox, MAD_R(R22, Oy, R32 * Oz)) * sYInv;
	REAL_R vy_prim = MAD_R(R12, v.x, MAD_R(R22, v.y, R32 * v.z)) * sYInv;

	REAL_R sZInv = 1 + EXP_R(-GC_3.y);
	REAL_R Oz_prim = MAD_R(R13, Ox, MAD_R(R23, Oy, R33 * Oz)) * sZInv;
	REAL_R vz_prim = MAD_R(R13, v.x, MAD_R(R23, v.y, R33 * v.z)) * sZInv;*/

	// NEW EXPONENTIAL ACTIVATION FUNCTION FOR SCALE PARAMETERS
	REAL_R sXInv = EXP_R(-GC_2.w);
	REAL_R Ox_prim = MAD_R(R11, Ox, MAD_R(R21, Oy, R31 * Oz)) * sXInv;
	REAL_R vx_prim = MAD_R(R11, v.x, MAD_R(R21, v.y, R31 * v.z)) * sXInv;

	REAL_R sYInv = EXP_R(-GC_3.x);
	REAL_R Oy_prim = MAD_R(R12, Ox, MAD_R(R22, Oy, R32 * Oz)) * sYInv;
	REAL_R vy_prim = MAD_R(R12, v.x, MAD_R(R22, v.y, R32 * v.z)) * sYInv;

	REAL_R sZInv = EXP_R(-GC_3.y);
	REAL_R Oz_prim = MAD_R(R13, Ox, MAD_R(R23, Oy, R33 * Oz)) * sZInv;
	REAL_R vz_prim = MAD_R(R13, v.x, MAD_R(R23, v.y, R33 * v.z)) * sZInv;
	
	// *********************************************************************************************

	REAL_R v_dot_v = MAD_R(vx_prim, vx_prim, MAD_R(vy_prim, vy_prim, vz_prim * vz_prim));
	REAL_R O_dot_v = MAD_R(Ox_prim, vx_prim, MAD_R(Oy_prim, vy_prim, Oz_prim * vz_prim));
	REAL_R O_dot_O = MAD_R(Ox_prim, Ox_prim, MAD_R(Oy_prim, Oy_prim, Oz_prim * Oz_prim));

	REAL_R tmp1 = 1 / v_dot_v;
	REAL_R tmp2 = O_dot_v * tmp1;

	REAL_R O_perp_x = MAD_R(-vx_prim, tmp2, Ox_prim);
	REAL_R O_perp_y = MAD_R(-vy_prim, tmp2, Oy_prim);
	REAL_R O_perp_z = MAD_R(-vz_prim, tmp2, Oz_prim);

	s = s * s; // !!! !!! !!!

	tmp2 = optixLaunchParams.chi_square_squared_radius * s;
	REAL_R O_perp_squared_norm = MAD_R(O_perp_x, O_perp_x, MAD_R(O_perp_y, O_perp_y, O_perp_z * O_perp_z));
	REAL_R delta = tmp2 - O_perp_squared_norm;
	
	if (delta >= 0) {
		REAL_R t = -(O_dot_v + COPYSIGN_R(SQRT_R(v_dot_v * delta), O_dot_v)) * tmp1;
		REAL_R t1 = ((O_dot_v <= 0) ? (O_dot_O - tmp2) / (v_dot_v * t) : t);
		optixReportIntersection(
			((float)t1),
			0,
			#ifndef RENDERER_OPTIX_USE_DOUBLE_PRECISION
				__float_as_uint(O_perp_squared_norm),
				__float_as_uint(s)
			#else
				(unsigned)__double2loint(O_perp_squared_norm),
				(unsigned)__double2hiint(O_perp_squared_norm),
				(unsigned)__double2loint(s),
				(unsigned)__double2hiint(s)
			#endif
			
		);
	}
}
