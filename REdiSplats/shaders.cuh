#include "Header.cuh"

// *************************************************************************************************

const int BUFFER_SIZE = 16;

// *************************************************************************************************

struct SRayPayload {
	int Gauss_num;
	float2 data[BUFFER_SIZE];
};

// *************************************************************************************************

template<int SH_degree>
__device__ void __raygen__() {
	int x = optixGetLaunchIndex().x;
	int y = optixGetLaunchIndex().y;
	int pixel_ind = (y * optixLaunchParams.width) + x;

	REAL3_R d = make_REAL3_R(
		(((REAL_R)-0.5) + __fdividef(x + ((REAL_R)0.5), optixLaunchParams.width)) * optixLaunchParams.double_tan_half_fov_x,
		(((REAL_R)-0.5) + __fdividef(y + ((REAL_R)0.5), optixLaunchParams.height)) * optixLaunchParams.double_tan_half_fov_y,
		1
	);
	REAL3_R v = make_REAL3_R(
		MAD_R(optixLaunchParams.R.x, d.x, MAD_R(optixLaunchParams.D.x, d.y, optixLaunchParams.F.x * d.z)),
		MAD_R(optixLaunchParams.R.y, d.x, MAD_R(optixLaunchParams.D.y, d.y, optixLaunchParams.F.y * d.z)),
		MAD_R(optixLaunchParams.R.z, d.x, MAD_R(optixLaunchParams.D.z, d.y, optixLaunchParams.F.z * d.z))
	);

	// *** *** *** *** ***

	SRayPayload rp;

	unsigned long long rp_addr = ((unsigned long long)&rp);
	unsigned rp_addr_lo = rp_addr;
	unsigned rp_addr_hi = rp_addr >> 32;

	// *** *** *** *** ***

	float tMin = 0.0f;

	// !!! !!! !!!
	float max_RSH = -INFINITY;
	float max_GSH = -INFINITY;
	float max_BSH = -INFINITY;
	// !!! !!! !!!

	float R = 0.0f;
	float G = 0.0f;
	float B = 0.0f;
	float T = 1.0f;

	int number_of_Gaussians = 0;

	int i;
	bool condition;
	do {
		rp.Gauss_num = 0;

		// *** *** *** *** ***

		optixTrace(
			optixLaunchParams.traversable,
			optixLaunchParams.O,
			v,
			tMin, // !!! !!! !!!
			INFINITY,
			0.0f,
			OptixVisibilityMask(255),
			OPTIX_RAY_FLAG_DISABLE_CLOSESTHIT,
			0,
			1,
			0,

			rp_addr_lo,
			rp_addr_hi
		);

		// *** *** *** *** ***

		for (i = 0; i < rp.Gauss_num; ++i) {
			float2 data = rp.data[i];

			int ind = __float_as_uint(data.y);
			tMin = data.x;

			// *** *** *** *** ***

			float4 GC_1 = optixLaunchParams.GC_part_1[ind];
			float4 GC_2 = optixLaunchParams.GC_part_2[ind];
			float4 GC_3 = optixLaunchParams.GC_part_3[ind];
			float2 GC_4 = optixLaunchParams.GC_part_4[ind];

			// *** *** *** *** ***

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

			// *** *** *** *** ***

			float3 O = make_float3(
				optixLaunchParams.O.x - GC_2.x,
				optixLaunchParams.O.y - GC_2.y,
				optixLaunchParams.O.z - GC_2.z
			);

			float3 O_prim;
			float3 v_prim;

			float sXInv = __expf(-GC_2.w);
			O_prim.x = __fmaf_rn(R11, O.x, __fmaf_rn(R21, O.y, R31 * O.z)) * sXInv;
			v_prim.x = __fmaf_rn(R11, v.x, __fmaf_rn(R21, v.y, R31 * v.z)) * sXInv;

			float sYInv = __expf(-GC_3.x);
			O_prim.y = __fmaf_rn(R12, O.x, __fmaf_rn(R22, O.y, R32 * O.z)) * sYInv;
			v_prim.y = __fmaf_rn(R12, v.x, __fmaf_rn(R22, v.y, R32 * v.z)) * sYInv;

			float sZInv = __expf(-GC_3.y);
			O_prim.z = __fmaf_rn(R13, O.x, __fmaf_rn(R23, O.y, R33 * O.z)) * sZInv;
			v_prim.z = __fmaf_rn(R13, v.x, __fmaf_rn(R23, v.y, R33 * v.z)) * sZInv;

			// *** *** *** *** ***

			float PHitX = __fmaf_rn(v_prim.x, tMin, O_prim.x); // !!! !!! !!!
			float PHitY = __fmaf_rn(v_prim.y, tMin, O_prim.y); // !!! !!! !!!
			float PHitZ = __fmaf_rn(v_prim.z, tMin, O_prim.z); // !!! !!! !!!

			float alpha = __fdividef(__expf(-0.5f * __fdividef(__fmaf_rn(PHitX, PHitX, __fmaf_rn(PHitY, PHitY, PHitZ * PHitZ)), (s * s))), 1.0f + __expf(-GC_1.w));

			// *** *** *** *** ***

			REAL_R RSH = ((REAL_R)0.28209479177387814) * GC_1.x;
			REAL_R GSH = ((REAL_R)0.28209479177387814) * GC_1.y;
			REAL_R BSH = ((REAL_R)0.28209479177387814) * GC_1.z;

			// Spherical harmonics
			if constexpr (SH_degree >= 1) {
				float4 GC_2 = optixLaunchParams.GC_part_2[ind];
				float4 GC_SH_1 = optixLaunchParams.GC_SH_1[ind];
				float4 GC_SH_2 = optixLaunchParams.GC_SH_2[ind];

				// !!! !!! !!!
				// It will be negated later in the subsequent block of code
				REAL3_R vSH = make_REAL3_R(O.x, O.y, O.z);
				// !!! !!! !!!

				REAL_R vSH_norm = MAD_R(vSH.x, vSH.x, MAD_R(vSH.y, vSH.y, vSH.z * vSH.z));
				REAL_R vSH_norm_inv;

				// !!! !!! !!!
				// Here vSH will be negated and normalized at the same time
				asm volatile (
					"rsqrt.approx.f32 %0, %1;\n"
					"neg.f32 %0, %0;"
					: "=f"(vSH_norm_inv)
					: "f"(vSH_norm)
				);
				// !!! !!! !!!

				vSH.x *= vSH_norm_inv;
				vSH.y *= vSH_norm_inv;
				vSH.z *= vSH_norm_inv;

				// *** *** *** *** ***

				REAL_R tmp;

				tmp = ((REAL_R)-0.4886025119029199) * vSH.y;
				RSH = MAD_R(tmp, GC_SH_1.x, RSH);
				GSH = MAD_R(tmp, GC_SH_1.y, GSH);
				BSH = MAD_R(tmp, GC_SH_1.z, BSH);

				tmp = ((REAL_R)0.4886025119029199) * vSH.z;
				RSH = MAD_R(tmp, GC_SH_1.w, RSH);
				GSH = MAD_R(tmp, GC_SH_2.x, GSH);
				BSH = MAD_R(tmp, GC_SH_2.y, BSH);

				tmp = ((REAL_R)-0.4886025119029199) * vSH.x;
				RSH = MAD_R(tmp, GC_SH_2.z, RSH);
				GSH = MAD_R(tmp, GC_SH_2.w, GSH);

				if constexpr (SH_degree >= 2) {
					float4 GC_SH_3 = optixLaunchParams.GC_SH_3[ind];
					float4 GC_SH_4 = optixLaunchParams.GC_SH_4[ind];
					float4 GC_SH_5 = optixLaunchParams.GC_SH_5[ind];
					float4 GC_SH_6 = optixLaunchParams.GC_SH_6[ind];

					BSH = MAD_R(tmp, GC_SH_3.x, BSH);

					REAL_R xx = vSH.x * vSH.x, yy = vSH.y * vSH.y, zz = vSH.z * vSH.z;
					REAL_R xy = vSH.x * vSH.y, yz = vSH.y * vSH.z, xz = vSH.x * vSH.z;

					tmp = ((REAL_R)1.0925484305920792) * xy;
					RSH = MAD_R(tmp, GC_SH_3.y, RSH);
					GSH = MAD_R(tmp, GC_SH_3.z, GSH);
					BSH = MAD_R(tmp, GC_SH_3.w, BSH);

					tmp = ((REAL_R)-1.0925484305920792) * yz;
					RSH = MAD_R(tmp, GC_SH_4.x, RSH);
					GSH = MAD_R(tmp, GC_SH_4.y, GSH);
					BSH = MAD_R(tmp, GC_SH_4.z, BSH);

					tmp = ((REAL_R)0.31539156525252005) * (3 * zz - 1);
					RSH = MAD_R(tmp, GC_SH_4.w, RSH);
					GSH = MAD_R(tmp, GC_SH_5.x, GSH);
					BSH = MAD_R(tmp, GC_SH_5.y, BSH);

					tmp = ((REAL_R)-1.0925484305920792) * xz;
					RSH = MAD_R(tmp, GC_SH_5.z, RSH);
					GSH = MAD_R(tmp, GC_SH_5.w, GSH);
					BSH = MAD_R(tmp, GC_SH_6.x, BSH);

					tmp = ((REAL_R)0.5462742152960396) * (xx - yy);
					RSH = MAD_R(tmp, GC_SH_6.y, RSH);
					GSH = MAD_R(tmp, GC_SH_6.z, GSH);
					BSH = MAD_R(tmp, GC_SH_6.w, BSH);

					if constexpr (SH_degree >= 3) {
						float4 GC_SH_7 = optixLaunchParams.GC_SH_7[ind];
						float4 GC_SH_8 = optixLaunchParams.GC_SH_8[ind];
						float4 GC_SH_9 = optixLaunchParams.GC_SH_9[ind];
						float4 GC_SH_10 = optixLaunchParams.GC_SH_10[ind];
						float4 GC_SH_11 = optixLaunchParams.GC_SH_11[ind];

						tmp = ((REAL_R)-0.5900435899266435) * vSH.y * (3 * xx - yy);
						RSH = MAD_R(tmp, GC_SH_7.x, RSH);
						GSH = MAD_R(tmp, GC_SH_7.y, GSH);
						BSH = MAD_R(tmp, GC_SH_7.z, BSH);

						tmp = ((REAL_R)2.890611442640554) * xy * vSH.z;
						RSH = MAD_R(tmp, GC_SH_7.w, RSH);
						GSH = MAD_R(tmp, GC_SH_8.x, GSH);
						BSH = MAD_R(tmp, GC_SH_8.y, BSH);

						tmp = ((REAL_R)-0.4570457994644658) * vSH.y * (5 * zz - 1);
						RSH = MAD_R(tmp, GC_SH_8.z, RSH);
						GSH = MAD_R(tmp, GC_SH_8.w, GSH);
						BSH = MAD_R(tmp, GC_SH_9.x, BSH);

						tmp = ((REAL_R)0.3731763325901154) * vSH.z * (5 * zz - 3);
						RSH = MAD_R(tmp, GC_SH_9.y, RSH);
						GSH = MAD_R(tmp, GC_SH_9.z, GSH);
						BSH = MAD_R(tmp, GC_SH_9.w, BSH);

						tmp = ((REAL_R)-0.4570457994644658) * vSH.x * (5 * zz - 1);
						RSH = MAD_R(tmp, GC_SH_10.x, RSH);
						GSH = MAD_R(tmp, GC_SH_10.y, GSH);
						BSH = MAD_R(tmp, GC_SH_10.z, BSH);

						tmp = ((REAL_R)1.445305721320277) * (xx - yy) * vSH.z;
						RSH = MAD_R(tmp, GC_SH_10.w, RSH);
						GSH = MAD_R(tmp, GC_SH_11.x, GSH);
						BSH = MAD_R(tmp, GC_SH_11.y, BSH);

						tmp = ((REAL_R)-0.5900435899266435) * vSH.x * (xx - 3 * yy);
						RSH = MAD_R(tmp, GC_SH_11.z, RSH);
						GSH = MAD_R(tmp, GC_SH_11.w, GSH);

						if constexpr (SH_degree >= 4) {
							float4 GC_SH_12 = optixLaunchParams.GC_SH_12[ind];
							float4 GC_SH_13 = optixLaunchParams.GC_SH_13[ind];
							float4 GC_SH_14 = optixLaunchParams.GC_SH_14[ind];
							float4 GC_SH_15 = optixLaunchParams.GC_SH_15[ind];
							float4 GC_SH_16 = optixLaunchParams.GC_SH_16[ind];
							float4 GC_SH_17 = optixLaunchParams.GC_SH_17[ind];
							float4 GC_SH_18 = optixLaunchParams.GC_SH_18[ind];

							BSH = MAD_R(tmp, GC_SH_12.x, BSH);

							tmp = ((REAL_R)2.50334294179670454) * xy * (xx - yy);
							RSH = MAD_R(tmp, GC_SH_12.y, RSH);
							GSH = MAD_R(tmp, GC_SH_12.z, GSH);
							BSH = MAD_R(tmp, GC_SH_12.w, BSH);

							tmp = ((REAL_R)-1.77013076977993053) * yz * (3 * xx - yy);
							RSH = MAD_R(tmp, GC_SH_13.x, RSH);
							GSH = MAD_R(tmp, GC_SH_13.y, GSH);
							BSH = MAD_R(tmp, GC_SH_13.z, BSH);

							tmp = ((REAL_R)0.94617469575756002) * xy * (7 * zz - 1);
							RSH = MAD_R(tmp, GC_SH_13.w, RSH);
							GSH = MAD_R(tmp, GC_SH_14.x, GSH);
							BSH = MAD_R(tmp, GC_SH_14.y, BSH);

							tmp = ((REAL_R)-0.66904654355728917) * yz * (7 * zz - 3);
							RSH = MAD_R(tmp, GC_SH_14.z, RSH);
							GSH = MAD_R(tmp, GC_SH_14.w, GSH);
							BSH = MAD_R(tmp, GC_SH_15.x, BSH);

							tmp = ((REAL_R)0.10578554691520430) * ((zz * (35 * zz - 30)) + 3);
							RSH = MAD_R(tmp, GC_SH_15.y, RSH);
							GSH = MAD_R(tmp, GC_SH_15.z, GSH);
							BSH = MAD_R(tmp, GC_SH_15.w, BSH);

							tmp = ((REAL_R)-0.66904654355728917) * xz * (7 * zz - 3);
							RSH = MAD_R(tmp, GC_SH_16.x, RSH);
							GSH = MAD_R(tmp, GC_SH_16.y, GSH);
							BSH = MAD_R(tmp, GC_SH_16.z, BSH);

							tmp = ((REAL_R)0.47308734787878001) * (xx - yy) * (7 * zz - 1);
							RSH = MAD_R(tmp, GC_SH_16.w, RSH);
							GSH = MAD_R(tmp, GC_SH_17.x, GSH);
							BSH = MAD_R(tmp, GC_SH_17.y, BSH);

							tmp = ((REAL_R)-1.77013076977993053) * xz * (xx - 3 * yy);
							RSH = MAD_R(tmp, GC_SH_17.z, RSH);
							GSH = MAD_R(tmp, GC_SH_17.w, GSH);
							BSH = MAD_R(tmp, GC_SH_18.x, BSH);

							tmp = ((REAL_R)0.62583573544917613) * ((xx * (xx - 3 * yy)) - (yy * (3 * xx - yy)));
							RSH = MAD_R(tmp, GC_SH_18.y, RSH);
							GSH = MAD_R(tmp, GC_SH_18.z, GSH);
							BSH = MAD_R(tmp, GC_SH_18.w, BSH);
						} else {
							float GC_SH_12 = optixLaunchParams.GC_SH_12[ind];
							BSH = MAD_R(tmp, GC_SH_12, BSH);
						}
					}
				} else {
					float GC_SH_3 = optixLaunchParams.GC_SH_3[ind];
					BSH = MAD_R(tmp, GC_SH_3, BSH);
				}
			}

			RSH = RSH + ((REAL_R)0.5);
			GSH = GSH + ((REAL_R)0.5);
			BSH = BSH + ((REAL_R)0.5);

			// *** *** *** *** ***

			REAL_R RSH_clamped = ((RSH < 0) ? 0 : RSH);
			REAL_R GSH_clamped = ((GSH < 0) ? 0 : GSH);
			REAL_R BSH_clamped = ((BSH < 0) ? 0 : BSH);

			if (!optixLaunchParams.inference) {
				if (RSH_clamped == RSH) ind |= 536870912;
				if (GSH_clamped == GSH) ind |= 1073741824;
				if (BSH_clamped == BSH) ind |= 2147483648;

				optixLaunchParams.Gaussians_indices[(number_of_Gaussians * optixLaunchParams.width * optixLaunchParams.height) + pixel_ind] = ind;
			
				// !!! !!! !!!
				max_RSH = ((RSH_clamped > max_RSH) ? RSH_clamped : max_RSH);
				max_GSH = ((GSH_clamped > max_GSH) ? GSH_clamped : max_GSH);
				max_BSH = ((BSH_clamped > max_BSH) ? BSH_clamped : max_BSH);
				// !!! !!! !!!
			}

			// *** *** *** *** ***

			REAL_R tmp = T * alpha;
			R = MAD_R(RSH_clamped, tmp, R);
			G = MAD_R(GSH_clamped, tmp, G);
			B = MAD_R(BSH_clamped, tmp, B);
			T = T - tmp;

			// *** *** *** *** ***

			++number_of_Gaussians;
			condition = (number_of_Gaussians == optixLaunchParams.max_Gaussians_per_ray) ||	(T < optixLaunchParams.ray_termination_T_threshold);
			if (condition)
				break;
		}
		tMin = nextafter(tMin, INFINITY);
	} while((i > 0) && (!condition));

	// *** *** *** *** ***

	R = MAD_R(T, optixLaunchParams.bg_color_R, R);
	G = MAD_R(T, optixLaunchParams.bg_color_G, G);
	B = MAD_R(T, optixLaunchParams.bg_color_B, B);

	// *** *** *** *** ***

	REAL_R R_clamped = __saturatef(R);
	REAL_R G_clamped = __saturatef(G);
	REAL_R B_clamped = __saturatef(B);

	int Ri = floorf(R_clamped * 255.99999f);
	int Gi = floorf(G_clamped * 255.99999f);
	int Bi = floorf(B_clamped * 255.99999f);

	optixLaunchParams.bitmap[pixel_ind] = (Ri << 16) + (Gi << 8) + Bi;

	// *** *** *** *** ***
	
	if (optixLaunchParams.inference)
		optixLaunchParams.bitmap[pixel_ind] = (Ri << 16) + (Gi << 8) + Bi;
	else {
		if (number_of_Gaussians < optixLaunchParams.max_Gaussians_per_ray)
			optixLaunchParams.Gaussians_indices[(number_of_Gaussians * optixLaunchParams.width * optixLaunchParams.height) + pixel_ind] = -1;

		// *** *** *** *** ***

		optixLaunchParams.bitmap_out_R[(y * (optixLaunchParams.width + 11 - 1)) + x] = R_clamped; // !!! !!! !!!
		optixLaunchParams.bitmap_out_G[(y * (optixLaunchParams.width + 11 - 1)) + x] = G_clamped; // !!! !!! !!!
		optixLaunchParams.bitmap_out_B[(y * (optixLaunchParams.width + 11 - 1)) + x] = B_clamped; // !!! !!! !!!

		// *** *** *** *** ***

		max_RSH = ((optixLaunchParams.bg_color_R > max_RSH) ? optixLaunchParams.bg_color_R : max_RSH);
		max_GSH = ((optixLaunchParams.bg_color_G > max_GSH) ? optixLaunchParams.bg_color_G : max_GSH);
		max_BSH = ((optixLaunchParams.bg_color_B > max_BSH) ? optixLaunchParams.bg_color_B : max_BSH);

		// !!! !!! !!!
		max_RSH = ((R_clamped != R) ? -max_RSH : max_RSH);
		max_GSH = ((G_clamped != G) ? -max_GSH : max_GSH);
		max_BSH = ((B_clamped != B) ? -max_BSH : max_BSH);
		// !!! !!! !!!

		optixLaunchParams.max_RSH[pixel_ind] = max_RSH; // !!! !!! !!!
		optixLaunchParams.max_GSH[pixel_ind] = max_GSH; // !!! !!! !!!
		optixLaunchParams.max_BSH[pixel_ind] = max_BSH; // !!! !!! !!!
	}
}

// *************************************************************************************************

template<int SH_degree>
__device__ void __anyhit__() {
	SRayPayload *rp;

	unsigned long long rp_addr_lo = optixGetPayload_0();
	unsigned long long rp_addr_hi = optixGetPayload_1();
	*((unsigned long long *)&rp) = rp_addr_lo + (rp_addr_hi << 32);

	// *** *** *** *** ***

	unsigned Gauss_ind = optixGetInstanceIndex();
	float tHit = optixGetRayTmax();
	int Gauss_num = rp->Gauss_num;

	// *** *** *** *** ***

	float2 tmp1 = make_float2(tHit, __uint_as_float(Gauss_ind));
	float2 tmp2;

	for (int i = 0; i < Gauss_num; ++i) {
		tmp2 = rp->data[i];

		if (tmp1.x < tmp2.x) {
			rp->data[i] = tmp1;
			tmp1 = tmp2;
		}
	}

	if (Gauss_num < BUFFER_SIZE) {
		rp->data[Gauss_num++] = tmp1;
		rp->Gauss_num = Gauss_num;

		optixIgnoreIntersection();
	} else {
		if (tHit <= tmp2.x) optixIgnoreIntersection();
	}
}