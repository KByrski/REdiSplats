#include "Header.cuh"

// *************************************************************************************************

template<int SH_degree>
__global__ void ComputeGradient(SOptiXRenderParams<SH_degree> params_OptiX) {
	int tid = (blockIdx.x * blockDim.x) + threadIdx.x;

	unsigned i = tid / params_OptiX.width;
	unsigned j = tid % params_OptiX.width;

	if ((j < params_OptiX.width) && (i < params_OptiX.height)) {
		REAL3_R d = make_REAL3_R(
			(((REAL_R)-0.5) + ((j + ((REAL_R)0.5)) / params_OptiX.width)) * params_OptiX.double_tan_half_fov_x,
			(((REAL_R)-0.5) + ((i + ((REAL_R)0.5)) / params_OptiX.height)) * params_OptiX.double_tan_half_fov_y,
			1
		);
		REAL3_G v = make_REAL3_G(
			MAD_G(params_OptiX.R.x, d.x, MAD_G(params_OptiX.D.x, d.y, params_OptiX.F.x * d.z)),
			MAD_G(params_OptiX.R.y, d.x, MAD_G(params_OptiX.D.y, d.y, params_OptiX.F.y * d.z)),
			MAD_G(params_OptiX.R.z, d.x, MAD_G(params_OptiX.D.z, d.y, params_OptiX.F.z * d.z))
		);

		//******************************************************************************************

		REAL_G R = params_OptiX.bitmap_out_R[(i * (params_OptiX.width + 11 - 1)) + j]; // !!! !!! !!!
		REAL_G G = params_OptiX.bitmap_out_G[(i * (params_OptiX.width + 11 - 1)) + j]; // !!! !!! !!!
		REAL_G B = params_OptiX.bitmap_out_B[(i * (params_OptiX.width + 11 - 1)) + j]; // !!! !!! !!!

		const REAL_G tmp0 = 1 / ((REAL_G)256.0); // !!! !!! !!!
		int color_ref =  params_OptiX.bitmap_ref[(params_OptiX.poseNum * (params_OptiX.width * params_OptiX.height)) + ((i * params_OptiX.width) + j)];
		REAL_G B_ref = ((color_ref & 255) + ((REAL_G)0.5)) * tmp0;
		color_ref = color_ref >> 8;
		REAL_G G_ref = ((color_ref & 255) + ((REAL_G)0.5)) * tmp0;
		color_ref = color_ref >> 8;
		REAL_G R_ref = (color_ref + ((REAL_G)0.5)) * tmp0;

		// Compute loss
		atomicAdd(
			params_OptiX.loss_device,
			MAD_G(R - R_ref, R - R_ref, MAD_G(G - G_ref, G - G_ref, (B - B_ref) * (B - B_ref)))
		);

		REAL_G dI_dalpha = 0;

		REAL_G d_dR_dalpha = MAD_G(
			1 - ((REAL_G)lambda),
			(2 * (R - R_ref)) / (3 * params_OptiX.width * params_OptiX.height),
			((REAL_G)lambda) * params_OptiX.mu_bitmap_out_R[((i + 10) * (params_OptiX.width + 11 - 1)) + (j + 10)]
		);
		REAL_G d_dG_dalpha = MAD_G(
			1 - ((REAL_G)lambda),
			(2 * (G - G_ref)) / (3 * params_OptiX.width * params_OptiX.height),
			((REAL_G)lambda) * params_OptiX.mu_bitmap_out_G[((i + 10) * (params_OptiX.width + 11 - 1)) + (j + 10)]
		);
		REAL_G d_dB_dalpha = MAD_G(
			1 - ((REAL_G)lambda),
			(2 * (B - B_ref)) / (3 * params_OptiX.width * params_OptiX.height),
			((REAL_G)lambda) * params_OptiX.mu_bitmap_out_B[((i + 10) * (params_OptiX.width + 11 - 1)) + (j + 10)]
		);

		REAL_G max_RSH = params_OptiX.max_RSH[(i * params_OptiX.width) + j];
		REAL_G max_GSH = params_OptiX.max_GSH[(i * params_OptiX.width) + j];
		REAL_G max_BSH = params_OptiX.max_BSH[(i * params_OptiX.width) + j];
		if (max_RSH < 0) d_dR_dalpha = 0;
		if (max_GSH < 0) d_dG_dalpha = 0;
		if (max_BSH < 0) d_dB_dalpha = 0;
		REAL_G dI_dalpha_max = 2 * MAD_G(ABS_G(d_dR_dalpha), max_RSH, MAD_G(ABS_G(d_dG_dalpha), max_GSH, ABS_G(d_dB_dalpha) * max_BSH)); // !!! !!! !!!
		
		//REAL_G dI_dalpha_max = ABS_G(d_dR_dalpha) + ABS_G(d_dG_dalpha) + ABS_G(d_dB_dalpha); // !!! !!! !!!

		REAL_G alpha_prev = 0; // !!! !!! !!!
		REAL_G alpha_next;

		REAL_G d_dI_dalpha_prev = MAD_G(R, d_dR_dalpha, MAD_G(G, d_dG_dalpha, B * d_dB_dalpha)); // !!! !!! !!!
		
		//******************************************************************************************

		int k = 0;

		// *****************************************************************************************

		REAL3_G dL_dm;

		// *****************************************************************************************

		int GaussInd_packed;
		int GaussInd;
		while (k < max_Gaussians_per_ray) {
			GaussInd_packed = params_OptiX.Gaussians_indices[(k * params_OptiX.width * params_OptiX.height) + ((i * params_OptiX.width) + j)]; // !!! !!! !!!
			GaussInd = GaussInd_packed & 536870911;

			bool R_channel = ((GaussInd_packed & 536870912) != 0);
			bool G_channel = ((GaussInd_packed & 1073741824) != 0);
			bool B_channel = ((GaussInd_packed & 2147483648) != 0);
			
			if (GaussInd_packed == -1) return; // !!! !!! !!!

			++k;

			//**************************************************************************************

			float4 GC_1 = params_OptiX.GC_part_1_1[GaussInd];
			float4 GC_2 = params_OptiX.GC_part_2_1[GaussInd];
			float4 GC_3 = params_OptiX.GC_part_3_1[GaussInd];
			float2 GC_4 = params_OptiX.GC_part_4_1[GaussInd];

			REAL_G aa = ((REAL_G)GC_3.z) * GC_3.z;
			REAL_G bb = ((REAL_G)GC_3.w) * GC_3.w;
			REAL_G cc = ((REAL_G)GC_4.x) * GC_4.x;
			REAL_G dd = ((REAL_G)GC_4.y) * GC_4.y;
			REAL_G s = ((REAL_G)0.5) * (aa + bb + cc + dd);

			REAL_G ab = ((REAL)GC_3.z) * GC_3.w; REAL_G ac = ((REAL_G)GC_3.z) * GC_4.x; REAL_G ad = ((REAL_G)GC_3.z) * GC_4.y;
			REAL_G bc = ((REAL_G)GC_3.w) * GC_4.x; REAL_G bd = ((REAL_G)GC_3.w) * GC_4.y;
			REAL_G cd = ((REAL_G)GC_4.x) * GC_4.y;       

			REAL_G R11 = s - cc - dd;
			REAL_G R12 = bc - ad;
			REAL_G R13 = bd + ac;

			REAL_G R21 = bc + ad;
			REAL_G R22 = s - bb - dd;
			REAL_G R23 = cd - ab;

			REAL_G R31 = bd - ac;
			REAL_G R32 = cd + ab;
			REAL_G R33 = s - bb - cc;

			REAL_G Ox = ((REAL_G)params_OptiX.O.x) - GC_2.x;
			REAL_G Oy = ((REAL_G)params_OptiX.O.y) - GC_2.y;
			REAL_G Oz = ((REAL_G)params_OptiX.O.z) - GC_2.z;

			// NEW EXPONENTIAL ACTIVATION FUNCTION FOR SCALE PARAMETERS
			REAL_G sXInv = EXP_G(-GC_2.w);
			REAL_G Ox_prim = MAD_G(R11, Ox, MAD_G(R21, Oy, R31 * Oz)) * sXInv;
			REAL_G vx_prim = MAD_G(R11, v.x, MAD_G(R21, v.y, R31 * v.z)) * sXInv;

			REAL_G sYInv = EXP_G(-GC_3.x);
			REAL_G Oy_prim = MAD_G(R12, Ox, MAD_G(R22, Oy, R32 * Oz)) * sYInv;
			REAL_G vy_prim = MAD_G(R12, v.x, MAD_G(R22, v.y, R32 * v.z)) * sYInv;

			REAL_G sZInv = EXP_G(-GC_3.y);
			REAL_G Oz_prim = MAD_G(R13, Ox, MAD_G(R23, Oy, R33 * Oz)) * sZInv;
			REAL_G vz_prim = MAD_G(R13, v.x, MAD_G(R23, v.y, R33 * v.z)) * sZInv;

			// NEW (MORE NUMERICALLY STABLE)
			REAL_G v_dot_v = MAD_G(vx_prim, vx_prim, MAD_G(vy_prim, vy_prim, vz_prim * vz_prim));
			REAL_G O_dot_O = MAD_G(Ox_prim, Ox_prim, MAD_G(Oy_prim, Oy_prim, Oz_prim * Oz_prim));
			REAL_G v_dot_O = MAD_G(vx_prim, Ox_prim, MAD_G(vy_prim, Oy_prim, vz_prim * Oz_prim));
			REAL_G tmp1 = v_dot_O / v_dot_v;
			REAL_G tmp2 = 1 / (1 + EXP_G(-GC_1.w));

			REAL_G vecx_tmp = MAD_G(-vx_prim, tmp1, Ox_prim); // !!! !!! !!!
			REAL_G vecy_tmp = MAD_G(-vy_prim, tmp1, Oy_prim); // !!! !!! !!!
			REAL_G vecz_tmp = MAD_G(-vz_prim, tmp1, Oz_prim); // !!! !!! !!!

#ifndef GRADIENT_OPTIX_USE_DOUBLE_PRECISION
			alpha_next = tmp2 * __saturatef(expf(-0.5f * (MAD_G(vecx_tmp, vecx_tmp, MAD_G(vecy_tmp, vecy_tmp, vecz_tmp * vecz_tmp)) / (s * s)))); // !!! !!! !!!
#else
			alpha_next = exp(-0.5 * (MAD_G(vecx_tmp, vecx_tmp, MAD_G(vecy_tmp, vecy_tmp, vecz_tmp * vecz_tmp)) / (s * s)));
			alpha_next = (alpha_next < 0) ? 0 : alpha_next;
			alpha_next = (alpha_next > 1) ? 1 : alpha_next;
			alpha_next = tmp2 * alpha_next; // !!! !!! !!!
#endif

			// *************************************************************************************

			REAL_G tmp3 = (1 - alpha_prev); // !!! !!! !!!

			d_dR_dalpha = d_dR_dalpha * tmp3;
			d_dG_dalpha = d_dG_dalpha * tmp3;
			d_dB_dalpha = d_dB_dalpha * tmp3;
			d_dI_dalpha_prev = d_dI_dalpha_prev * tmp3;
			dI_dalpha_max = dI_dalpha_max * tmp3; // !!! !!! !!!
			
			// *************************************************************************************

			REAL_G dL_dparam;

			// *************************************************************************************

			// dL_d[R, G, B]
			REAL_G RSH = ((REAL_G)0.28209479177387814) * GC_1.x;
			if (R_channel) {
				dL_dparam = ((REAL_G)0.28209479177387814) * d_dR_dalpha * alpha_next;
				atomicAdd(((REAL_G *)params_OptiX.dL_dparams_1) + (GaussInd * 4), dL_dparam);
			}

			REAL_G GSH = ((REAL_G)0.28209479177387814) * GC_1.y;
			if (G_channel) {
				dL_dparam = ((REAL_G)0.28209479177387814) * d_dG_dalpha * alpha_next;
				atomicAdd(((REAL_G *)params_OptiX.dL_dparams_1) + (GaussInd * 4) + 1, dL_dparam);
			}

			REAL_G BSH = ((REAL_G)0.28209479177387814) * GC_1.z;
			if (B_channel) {
				dL_dparam = ((REAL_G)0.28209479177387814) * d_dB_dalpha * alpha_next;
				atomicAdd(((REAL_G *)params_OptiX.dL_dparams_1) + (GaussInd * 4) + 2, dL_dparam);
			}

			// *************************************************************************************

			dL_dm = make_REAL3_G(0, 0, 0);

			// *************************************************************************************

			// Spherical harmonics
			if constexpr (SH_degree >= 1) {
				if (params_OptiX.epoch >= SH1_activation_iter) {
					float4 GC_SH_1 = params_OptiX.GC_SH_1[GaussInd];
					float4 GC_SH_2 = params_OptiX.GC_SH_2[GaussInd];

					// *** *** *** *** ***
				
					REAL3_G vSH = make_REAL3_G(-Ox, -Oy, -Oz);

					REAL_R vSH_norm = MAD_G(vSH.x, vSH.x, MAD_G(vSH.y, vSH.y, vSH.z * vSH.z));
					REAL_R vSH_norm_inv;
					#ifndef GRADIENT_USE_DOUBLE_PRECISION
						asm volatile ("rsqrt.approx.f32 %0, %1;" : "=f"(vSH_norm_inv) : "f"(vSH_norm));
					#else
						vSH_norm_inv = 1 / sqrt(x);
					#endif

					vSH.x *= vSH_norm_inv;
					vSH.y *= vSH_norm_inv;
					vSH.z *= vSH_norm_inv;

					// *** *** *** *** ***
					
					REAL_G RSH_band = 0;
					REAL_G GSH_band = 0;
					REAL_G BSH_band = 0;

					// *** *** *** *** ***

					REAL_G tmp;
					REAL_G tmp2;
					REAL_G tmp2_x, tmp2_y, tmp2_z;

					// *** *** *** *** ***

					tmp = ((REAL_G)-0.4886025119029199) * vSH.y;
					dL_dparam = tmp * alpha_next;
					tmp2 = 0;
				
					// dL_dR_1_(-1)
					RSH_band = MAD_G(tmp, GC_SH_1.x, RSH_band);
					if (R_channel) {
						atomicAdd(((REAL_G *)params_OptiX.dL_dparams_SH_1) + (GaussInd * 4)    , dL_dparam * d_dR_dalpha);
						tmp2 = MAD_G(GC_SH_1.x, d_dR_dalpha, tmp2);
					}

					// dL_dG_1_(-1)
					GSH_band = MAD_G(tmp, GC_SH_1.y, GSH_band);
					if (G_channel) {
						atomicAdd(((REAL_G *)params_OptiX.dL_dparams_SH_1) + (GaussInd * 4) + 1, dL_dparam * d_dG_dalpha);
						tmp2 = MAD_G(GC_SH_1.y, d_dG_dalpha, tmp2);
					}

					// dL_dB_1_(-1)
					BSH_band = MAD_G(tmp, GC_SH_1.z, BSH_band);
					if (B_channel) {
						atomicAdd(((REAL_G *)params_OptiX.dL_dparams_SH_1) + (GaussInd * 4) + 2, dL_dparam * d_dB_dalpha);
						tmp2 = MAD_G(GC_SH_1.z, d_dB_dalpha, tmp2);
					}

					dL_dm.y = MAD_G(((REAL_G)-0.4886025119029199), tmp2, dL_dm.y);

					// *** *** *** *** ***

					tmp = ((REAL_G)0.4886025119029199) * vSH.z;
					dL_dparam = tmp * alpha_next;
					tmp2 = 0;

					// dL_dR_1_0
					RSH_band = MAD_G(tmp, GC_SH_1.w, RSH_band);
					if (R_channel) {
						atomicAdd(((REAL_G *)params_OptiX.dL_dparams_SH_1) + (GaussInd * 4) + 3, dL_dparam * d_dR_dalpha);
						tmp2 = MAD_G(GC_SH_1.w, d_dR_dalpha, tmp2);
					}

					// dL_dG_1_0
					GSH_band = MAD_G(tmp, GC_SH_2.x, GSH_band);
					if (G_channel) {
						atomicAdd(((REAL_G *)params_OptiX.dL_dparams_SH_2) + (GaussInd * 4)    , dL_dparam * d_dG_dalpha);
						tmp2 = MAD_G(GC_SH_2.x, d_dG_dalpha, tmp2);
					}

					// dL_dB_1_0
					BSH_band = MAD_G(tmp, GC_SH_2.y, BSH_band);
					if (B_channel) {
						atomicAdd(((REAL_G *)params_OptiX.dL_dparams_SH_2) + (GaussInd * 4) + 1, dL_dparam * d_dB_dalpha);
						tmp2 = MAD_G(GC_SH_2.y, d_dB_dalpha, tmp2);
					}

					dL_dm.z = MAD_G(((REAL_G)0.4886025119029199), tmp2, dL_dm.z);

					// *** *** *** *** ***

					tmp = ((REAL_G)-0.4886025119029199) * vSH.x;
					dL_dparam = tmp * alpha_next;
					tmp2 = 0;

					// dL_dR_1_1
					RSH_band = MAD_G(tmp, GC_SH_2.z, RSH_band);
					if (R_channel) {
						atomicAdd(((REAL_G *)params_OptiX.dL_dparams_SH_2) + (GaussInd * 4) + 2, dL_dparam * d_dR_dalpha);
						tmp2 = MAD_G(GC_SH_2.z, d_dR_dalpha, tmp2);
					}
				
					// dL_dG_1_1
					GSH_band = MAD_G(tmp, GC_SH_2.w, GSH_band);
					if (G_channel) {
						atomicAdd(((REAL_G *)params_OptiX.dL_dparams_SH_2) + (GaussInd * 4) + 3, dL_dparam * d_dG_dalpha);
						tmp2 = MAD_G(GC_SH_2.w, d_dG_dalpha, tmp2);
					}

					// *** *** *** *** ***

					if constexpr (SH_degree >= 2) {
						float4 GC_SH_3 = params_OptiX.GC_SH_3[GaussInd];
						float4 GC_SH_4 = params_OptiX.GC_SH_4[GaussInd];
						float4 GC_SH_5 = params_OptiX.GC_SH_5[GaussInd];
						float4 GC_SH_6 = params_OptiX.GC_SH_6[GaussInd];

						// *** *** *** *** ***

						// dL_dB_1_1
						BSH_band = MAD_G(tmp, GC_SH_3.x, BSH_band);
						if (B_channel) {
							atomicAdd(((REAL_G *)params_OptiX.dL_dparams_SH_3) + (GaussInd * 4), dL_dparam * d_dB_dalpha);
							tmp2 = MAD_G(GC_SH_3.x, d_dB_dalpha, tmp2);
						}

						dL_dm.x = MAD_G(((REAL_G)-0.4886025119029199), tmp2, dL_dm.x);

						// *** *** *** *** ***

						RSH += RSH_band;
						GSH += GSH_band;
						BSH += BSH_band;

						tmp2 = 0;
						if (R_channel) tmp2 = MAD_G(RSH_band, d_dR_dalpha, tmp2);
						if (G_channel) tmp2 = MAD_G(GSH_band, d_dG_dalpha, tmp2);
						if (B_channel) tmp2 = MAD_G(BSH_band, d_dB_dalpha, tmp2);
						tmp2 = -tmp2; // !!! !!! !!!

						dL_dm.x = MAD_G(vSH.x, tmp2, dL_dm.x);
						dL_dm.y = MAD_G(vSH.y, tmp2, dL_dm.y);
						dL_dm.z = MAD_G(vSH.z, tmp2, dL_dm.z);
					
						RSH_band = 0;
						GSH_band = 0;
						BSH_band = 0;

						// *** *** *** *** ***

						if (params_OptiX.epoch >= SH2_activation_iter) {
							REAL_G xx = vSH.x * vSH.x, yy = vSH.y * vSH.y, zz = vSH.z * vSH.z;
							REAL_G xy = vSH.x * vSH.y, yz = vSH.y * vSH.z, xz = vSH.x * vSH.z;

							// *** *** *** *** ***

							tmp = ((REAL_G)1.0925484305920792) * xy;
							dL_dparam = tmp * alpha_next;
							tmp2 = 0;

							// dL_dR_2_(-2)
							RSH_band = MAD_G(tmp, GC_SH_3.y, RSH_band);
							if (R_channel) {
								atomicAdd(((REAL_G *)params_OptiX.dL_dparams_SH_3) + (GaussInd * 4) + 1, dL_dparam * d_dR_dalpha);
								tmp2 = MAD_G(GC_SH_3.y, d_dR_dalpha, tmp2);
							}

							// dL_dG_2_(-2)
							GSH_band = MAD_G(tmp, GC_SH_3.z, GSH_band);
							if (G_channel) {
								atomicAdd(((REAL_G *)params_OptiX.dL_dparams_SH_3) + (GaussInd * 4) + 2, dL_dparam * d_dG_dalpha);
								tmp2 = MAD_G(GC_SH_3.z, d_dG_dalpha, tmp2);
							}

							// dL_dB_2_(-2)
							BSH_band = MAD_G(tmp, GC_SH_3.w, BSH_band);
							if (B_channel) {
								atomicAdd(((REAL_G *)params_OptiX.dL_dparams_SH_3) + (GaussInd * 4) + 3, dL_dparam * d_dB_dalpha);
								tmp2 = MAD_G(GC_SH_3.w, d_dB_dalpha, tmp2);
							}

							// xy
							// dmx: -2kx(xy) + k(y)
							// dmy: -2ky(xy) + k(x)
							// dmz: -2kz(xy)
							dL_dm.x = MAD_G(((REAL_G)1.0925484305920792) * vSH.y, tmp2, dL_dm.x);
							dL_dm.y = MAD_G(((REAL_G)1.0925484305920792) * vSH.x, tmp2, dL_dm.y);

							// *** *** *** *** ***

							tmp = ((REAL_G)-1.0925484305920792) * yz;
							dL_dparam = tmp * alpha_next;
							tmp2 = 0;

							// dL_dR_2_(-1)
							RSH_band = MAD_G(tmp, GC_SH_4.x, RSH_band);
							if (R_channel) {
								atomicAdd(((REAL_G *)params_OptiX.dL_dparams_SH_4) + (GaussInd * 4)    , dL_dparam * d_dR_dalpha);
								tmp2 = MAD_G(GC_SH_4.x, d_dR_dalpha, tmp2);
							}

							// dL_dG_2_(-1)
							GSH_band = MAD_G(tmp, GC_SH_4.y, GSH_band);
							if (G_channel) {
								atomicAdd(((REAL_G *)params_OptiX.dL_dparams_SH_4) + (GaussInd * 4) + 1, dL_dparam * d_dG_dalpha);
								tmp2 = MAD_G(GC_SH_4.y, d_dG_dalpha, tmp2);
							}

							// dL_dB_2_(-1)
							BSH_band = MAD_G(tmp, GC_SH_4.z, BSH_band);
							if (B_channel) {
								atomicAdd(((REAL_G *)params_OptiX.dL_dparams_SH_4) + (GaussInd * 4) + 2, dL_dparam * d_dB_dalpha);
								tmp2 = MAD_G(GC_SH_4.z, d_dB_dalpha, tmp2);
							}

							// yz
							// dmx: -2kx(yz)   
							// dmy: -2ky(yz) + k(z)
							// dmz: -2kz(yz) + k(y)
							dL_dm.y = MAD_G(((REAL_G)-1.0925484305920792) * vSH.z, tmp2, dL_dm.y);
							dL_dm.z = MAD_G(((REAL_G)-1.0925484305920792) * vSH.y, tmp2, dL_dm.z);

							// *** *** *** *** ***

							tmp = ((REAL_G)0.31539156525252005) * (3 * zz - 1);
							dL_dparam = tmp * alpha_next;
							tmp2 = 0;

							// dL_dR_2_0
							RSH_band = MAD_G(tmp, GC_SH_4.w, RSH_band);
							if (R_channel) {
								atomicAdd(((REAL_G *)params_OptiX.dL_dparams_SH_4) + (GaussInd * 4) + 3, dL_dparam * d_dR_dalpha);
								tmp2 = MAD_G(GC_SH_4.w, d_dR_dalpha, tmp2);
							}

							// dL_dG_2_0
							GSH_band = MAD_G(tmp, GC_SH_5.x, GSH_band);
							if (G_channel) {
								atomicAdd(((REAL_G *)params_OptiX.dL_dparams_SH_5) + (GaussInd * 4)    , dL_dparam * d_dG_dalpha);
								tmp2 = MAD_G(GC_SH_5.x, d_dG_dalpha, tmp2);
							}

							// dL_dB_2_0
							BSH_band = MAD_G(tmp, GC_SH_5.y, BSH_band);
							if (B_channel) {
								atomicAdd(((REAL_G *)params_OptiX.dL_dparams_SH_5) + (GaussInd * 4) + 1, dL_dparam * d_dB_dalpha);
								tmp2 = MAD_G(GC_SH_5.y, d_dB_dalpha, tmp2);
							}

							// 3zz - 1
							// dmx: -2kx(3zz - 1)
							// dmy: -2ky(3zz - 1)
							// dmz: -2kz(3zz - 1) + k(6z)
							dL_dm.z = MAD_G(((REAL_G)0.31539156525252005) * 6 * vSH.z, tmp2, dL_dm.z);

							// *** *** *** *** ***

							tmp = ((REAL_G)-1.0925484305920792) * xz;
							dL_dparam = tmp * alpha_next;
							tmp2 = 0;

							// dL_dR_2_1
							RSH_band = MAD_G(tmp, GC_SH_5.z, RSH_band);
							if (R_channel) {
								atomicAdd(((REAL_G *)params_OptiX.dL_dparams_SH_5) + (GaussInd * 4) + 2, dL_dparam * d_dR_dalpha);
								tmp2 = MAD_G(GC_SH_5.z, d_dR_dalpha, tmp2);
							}

							// dL_dG_2_1
							GSH_band = MAD_G(tmp, GC_SH_5.w, GSH_band);
							if (G_channel) {
								atomicAdd(((REAL_G *)params_OptiX.dL_dparams_SH_5) + (GaussInd * 4) + 3, dL_dparam * d_dG_dalpha);
								tmp2 = MAD_G(GC_SH_5.w, d_dG_dalpha, tmp2);
							}

							// dL_dB_2_1
							BSH_band = MAD_G(tmp, GC_SH_6.x, BSH_band);
							if (B_channel) {
								atomicAdd(((REAL_G *)params_OptiX.dL_dparams_SH_6) + (GaussInd * 4)    , dL_dparam * d_dB_dalpha);
								tmp2 = MAD_G(GC_SH_6.x, d_dB_dalpha, tmp2);
							}

							// xz
							// dmx: -2kx(xz) + k(z)
							// dmy: -2ky(xz)  
							// dmz: -2kz(xz) + k(x)
							dL_dm.x = MAD_G(((REAL_G)-1.0925484305920792) * vSH.z, tmp2, dL_dm.x);
							dL_dm.z = MAD_G(((REAL_G)-1.0925484305920792) * vSH.x, tmp2, dL_dm.z);

							// *** *** *** *** ***

							tmp = ((REAL_G)0.5462742152960396) * (xx - yy);
							dL_dparam = tmp * alpha_next;
							tmp2 = 0;

							// dL_dR_2_2
							RSH_band = MAD_G(tmp, GC_SH_6.y, RSH_band);
							if (R_channel) {
								atomicAdd(((REAL_G *)params_OptiX.dL_dparams_SH_6) + (GaussInd * 4) + 1, dL_dparam * d_dR_dalpha);
								tmp2 = MAD_G(GC_SH_6.y, d_dR_dalpha, tmp2);
							}

							// dL_dG_2_2
							GSH_band = MAD_G(tmp, GC_SH_6.z, GSH_band);
							if (G_channel) {
								atomicAdd(((REAL_G *)params_OptiX.dL_dparams_SH_6) + (GaussInd * 4) + 2, dL_dparam * d_dG_dalpha);
								tmp2 = MAD_G(GC_SH_6.z, d_dG_dalpha, tmp2);
							}

							// dL_dB_2_2
							BSH_band = MAD_G(tmp, GC_SH_6.w, BSH_band);
							if (B_channel) {
								atomicAdd(((REAL_G *)params_OptiX.dL_dparams_SH_6) + (GaussInd * 4) + 3, dL_dparam * d_dB_dalpha);
								tmp2 = MAD_G(GC_SH_6.w, d_dB_dalpha, tmp2);
							}

							// xx - yy
							// dmx: -2kx(xx - yy) + k( 2x)
							// dmy: -2ky(xx - yy) + k(-2y)
							// dmz: -2kz(xx - yy)
							dL_dm.x = MAD_G(((REAL_G)0.5462742152960396) *  2 * vSH.x, tmp2, dL_dm.x);
							dL_dm.y = MAD_G(((REAL_G)0.5462742152960396) * -2 * vSH.y, tmp2, dL_dm.y);

							// *** *** *** *** ***

							RSH += RSH_band;
							GSH += GSH_band;
							BSH += BSH_band;

							tmp2 = 0;
							if (R_channel) tmp2 = MAD_G(RSH_band, d_dR_dalpha, tmp2);
							if (G_channel) tmp2 = MAD_G(GSH_band, d_dG_dalpha, tmp2);
							if (B_channel) tmp2 = MAD_G(BSH_band, d_dB_dalpha, tmp2);
							tmp2 *= -2; // !!! !!! !!!

							dL_dm.x = MAD_G(vSH.x, tmp2, dL_dm.x);
							dL_dm.y = MAD_G(vSH.y, tmp2, dL_dm.y);
							dL_dm.z = MAD_G(vSH.z, tmp2, dL_dm.z);
										
							// *** *** *** *** ***

							if constexpr (SH_degree >= 3) {							
								if (params_OptiX.epoch >= SH3_activation_iter) {
									float4 GC_SH_7 = params_OptiX.GC_SH_7[GaussInd];
									float4 GC_SH_8 = params_OptiX.GC_SH_8[GaussInd];
									float4 GC_SH_9 = params_OptiX.GC_SH_9[GaussInd];
									float4 GC_SH_10 = params_OptiX.GC_SH_10[GaussInd];
									float4 GC_SH_11 = params_OptiX.GC_SH_11[GaussInd];

									// *** *** *** *** ***

									RSH_band = 0;
									GSH_band = 0;
									BSH_band = 0;

									// *** *** *** *** ***

									tmp = ((REAL_G)-0.5900435899266435) * vSH.y * (3 * xx - yy);
									dL_dparam = tmp * alpha_next;
									tmp2 = 0;

									// dL_dR_3_(-3)
									RSH_band = MAD_G(tmp, GC_SH_7.x, RSH_band);
									if (R_channel) {
										atomicAdd(((REAL_G *)params_OptiX.dL_dparams_SH_7) + (GaussInd * 4)    , dL_dparam * d_dR_dalpha);
										tmp2 = MAD_G(GC_SH_7.x, d_dR_dalpha, tmp2);
									}

									// dL_dG_3_(-3)
									GSH_band = MAD_G(tmp, GC_SH_7.y, GSH_band);
									if (G_channel) {
										atomicAdd(((REAL_G *)params_OptiX.dL_dparams_SH_7) + (GaussInd * 4) + 1, dL_dparam * d_dG_dalpha);
										tmp2 = MAD_G(GC_SH_7.y, d_dG_dalpha, tmp2);
									}

									// dL_dB_3_(-3)
									BSH_band = MAD_G(tmp, GC_SH_7.z, BSH_band);
									if (B_channel) {
										atomicAdd(((REAL_G *)params_OptiX.dL_dparams_SH_7) + (GaussInd * 4) + 2, dL_dparam * d_dB_dalpha);
										tmp2 = MAD_G(GC_SH_7.z, d_dB_dalpha, tmp2);
									}

									// 3xxy - yyy
									// dmx: -3kx(3xxy - yyy) + k(6xy)
									// dmy: -3ky(3xxy - yyy) + k(3xx - 3yy)
									// dmz: -3kz(3xxy - yyy)
									dL_dm.x = MAD_G(((REAL_G)-0.5900435899266435) * (6 * xy         ), tmp2, dL_dm.x);
									dL_dm.y = MAD_G(((REAL_G)-0.5900435899266435) * (3 * xx - 3 * yy), tmp2, dL_dm.y);

									// *** *** *** *** ***

									tmp = ((REAL_G)2.890611442640554) * xy * vSH.z;
									dL_dparam = tmp * alpha_next;
									tmp2 = 0;

									// dL_dR_3_(-2)
									RSH_band = MAD_G(tmp, GC_SH_7.w, RSH_band);
									if (R_channel) {
										atomicAdd(((REAL_G *)params_OptiX.dL_dparams_SH_7) + (GaussInd * 4) + 3, dL_dparam * d_dR_dalpha);
										tmp2 = MAD_G(GC_SH_7.w, d_dR_dalpha, tmp2);
									}

									// dL_dG_3_(-2)
									GSH_band = MAD_G(tmp, GC_SH_8.x, GSH_band);
									if (G_channel) {
										atomicAdd(((REAL_G *)params_OptiX.dL_dparams_SH_8) + (GaussInd * 4)    , dL_dparam * d_dG_dalpha);
										tmp2 = MAD_G(GC_SH_8.x, d_dG_dalpha, tmp2);
									}

									// dL_dB_3_(-2)
									BSH_band = MAD_G(tmp, GC_SH_8.y, BSH_band);
									if (B_channel) {
										atomicAdd(((REAL_G *)params_OptiX.dL_dparams_SH_8) + (GaussInd * 4) + 1, dL_dparam * d_dB_dalpha);
										tmp2 = MAD_G(GC_SH_8.y, d_dB_dalpha, tmp2);
									}

									// xyz
									// dmx: -3kx(xyz) + k(yz)
									// dmy: -3ky(xyz) + k(xz)
									// dmz: -3kz(xyz) + k(xy)
									dL_dm.x = MAD_G(((REAL_G)2.890611442640554) * yz, tmp2, dL_dm.x);
									dL_dm.y = MAD_G(((REAL_G)2.890611442640554) * xz, tmp2, dL_dm.y);
									dL_dm.z = MAD_G(((REAL_G)2.890611442640554) * xy, tmp2, dL_dm.z);

									// *** *** *** *** ***

									tmp = ((REAL_G)-0.4570457994644658) * vSH.y * (5 * zz - 1);
									dL_dparam = tmp * alpha_next;
									tmp2 = 0;

									// dL_dR_3_(-1)
									RSH_band = MAD_G(tmp, GC_SH_8.z, RSH_band);
									if (R_channel) {
										atomicAdd(((REAL_G *)params_OptiX.dL_dparams_SH_8) + (GaussInd * 4) + 2, dL_dparam * d_dR_dalpha);
										tmp2 = MAD_G(GC_SH_8.z, d_dR_dalpha, tmp2);
									}

									// dL_dG_3_(-1)
									GSH_band = MAD_G(tmp, GC_SH_8.w, GSH_band);
									if (G_channel) {
										atomicAdd(((REAL_G *)params_OptiX.dL_dparams_SH_8) + (GaussInd * 4) + 3, dL_dparam * d_dG_dalpha);
										tmp2 = MAD_G(GC_SH_8.w, d_dG_dalpha, tmp2);
									}

									// dL_dB_3_(-1)
									BSH_band = MAD_G(tmp, GC_SH_9.x, BSH_band);
									if (B_channel) {
										atomicAdd(((REAL_G *)params_OptiX.dL_dparams_SH_9) + (GaussInd * 4)    , dL_dparam * d_dB_dalpha);
										tmp2 = MAD_G(GC_SH_9.x, d_dB_dalpha, tmp2);
									}

									// 5yzz - y
									// dmx: -3kx(5yzz - y)
									// dmy: -3ky(5yzz - y) + k(5zz - 1)
									// dmz: -3kz(5yzz - y) + k(10yz)
									dL_dm.y = MAD_G(((REAL_G)-0.4570457994644658) * ( 5 * zz - 1), tmp2, dL_dm.y);
									dL_dm.z = MAD_G(((REAL_G)-0.4570457994644658) * (10 * yz    ), tmp2, dL_dm.z);

									// *** *** *** *** ***

									tmp = ((REAL_G)0.3731763325901154) * vSH.z * (5 * zz - 3);
									dL_dparam = tmp * alpha_next;
									tmp2 = 0;

									// dL_dR_3_0
									RSH_band = MAD_G(tmp, GC_SH_9.y, RSH_band);
									if (R_channel) {
										atomicAdd(((REAL_G *)params_OptiX.dL_dparams_SH_9) + (GaussInd * 4) + 1, dL_dparam * d_dR_dalpha);
										tmp2 = MAD_G(GC_SH_9.y, d_dR_dalpha, tmp2);
									}

									// dL_dG_3_0
									GSH_band = MAD_G(tmp, GC_SH_9.z, GSH_band);
									if (G_channel) {
										atomicAdd(((REAL_G *)params_OptiX.dL_dparams_SH_9) + (GaussInd * 4) + 2, dL_dparam * d_dG_dalpha);
										tmp2 = MAD_G(GC_SH_9.z, d_dG_dalpha, tmp2);
									}

									// dL_dB_3_0
									BSH_band = MAD_G(tmp, GC_SH_9.w, BSH_band);
									if (B_channel) {
										atomicAdd(((REAL_G *)params_OptiX.dL_dparams_SH_9) + (GaussInd * 4) + 3, dL_dparam * d_dB_dalpha);
										tmp2 = MAD_G(GC_SH_9.w, d_dB_dalpha, tmp2);
									}

									// 5zzz - 3z
									// dmx: -3kx(5zzz - 3z)
									// dmy: -3ky(5zzz - 3z)
									// dmz: -3kz(5zzz - 3z) + k(15zz - 3)
									dL_dm.z = MAD_G(((REAL_G)0.3731763325901154) * (15 * zz - 3), tmp2, dL_dm.z);

									// *** *** *** *** ***

									tmp = ((REAL_G)-0.4570457994644658) * vSH.x * (5 * zz - 1);
									dL_dparam = tmp * alpha_next;
									tmp2 = 0;

									// dL_dR_3_1
									RSH_band = MAD_G(tmp, GC_SH_10.x, RSH_band);
									if (R_channel) {
										atomicAdd(((REAL_G *)params_OptiX.dL_dparams_SH_10) + (GaussInd * 4)    , dL_dparam * d_dR_dalpha);
										tmp2 = MAD_G(GC_SH_10.x, d_dR_dalpha, tmp2);
									}

									// dL_dG_3_1
									GSH_band = MAD_G(tmp, GC_SH_10.y, GSH_band);
									if (G_channel) {
										atomicAdd(((REAL_G *)params_OptiX.dL_dparams_SH_10) + (GaussInd * 4) + 1, dL_dparam * d_dG_dalpha);
										tmp2 = MAD_G(GC_SH_10.y, d_dG_dalpha, tmp2);
									}

									// dL_dB_3_1
									BSH_band = MAD_G(tmp, GC_SH_10.z, BSH_band);
									if (B_channel) {
										atomicAdd(((REAL_G *)params_OptiX.dL_dparams_SH_10) + (GaussInd * 4) + 2, dL_dparam * d_dB_dalpha);
										tmp2 = MAD_G(GC_SH_10.z, d_dB_dalpha, tmp2);
									}

									// 5xzz - x
									// dmx: -3kx(5xzz - x) + k(5zz - 1)
									// dmy: -3ky(5xzz - x)
									// dmz: -3kz(5xzz - x) + k(10xz)
									dL_dm.x = MAD_G(((REAL_G)-0.4570457994644658) * ( 5 * zz - 1), tmp2, dL_dm.x);
									dL_dm.z = MAD_G(((REAL_G)-0.4570457994644658) * (10 * xz    ), tmp2, dL_dm.z);

									// *** *** *** *** ***

									tmp = ((REAL_G)1.445305721320277) * (xx - yy) * vSH.z;						
									dL_dparam = tmp * alpha_next;
									tmp2 = 0;

									// dL_dR_3_2
									RSH_band = MAD_G(tmp, GC_SH_10.w, RSH_band);
									if (R_channel) {
										atomicAdd(((REAL_G *)params_OptiX.dL_dparams_SH_10) + (GaussInd * 4) + 3, dL_dparam * d_dR_dalpha);
										tmp2 = MAD_G(GC_SH_10.w, d_dR_dalpha, tmp2);
									}

									// dL_dG_3_2
									GSH_band = MAD_G(tmp, GC_SH_11.x, GSH_band);
									if (G_channel) {
										atomicAdd(((REAL_G *)params_OptiX.dL_dparams_SH_11) + (GaussInd * 4)    , dL_dparam * d_dG_dalpha);
										tmp2 = MAD_G(GC_SH_11.x, d_dG_dalpha, tmp2);
									}

									// dL_dB_3_2
									BSH_band = MAD_G(tmp, GC_SH_11.y, BSH_band);
									if (B_channel) {
										atomicAdd(((REAL_G *)params_OptiX.dL_dparams_SH_11) + (GaussInd * 4) + 1, dL_dparam * d_dB_dalpha);
										tmp2 = MAD_G(GC_SH_11.y, d_dB_dalpha, tmp2);
									}

									// xxz - yyz
									// dmx: -3kx(xxz - yyz) + k(2xz)
									// dmy: -3ky(xxz - yyz) + k(-2yz)
									// dmz: -3kz(xxz - yyz) + k(xx - yy)
									dL_dm.x = MAD_G(((REAL_G)1.445305721320277) * ( 2 * xz ), tmp2, dL_dm.x);
									dL_dm.y = MAD_G(((REAL_G)1.445305721320277) * (-2 * yz ), tmp2, dL_dm.y);
									dL_dm.z = MAD_G(((REAL_G)1.445305721320277) * ( xx - yy), tmp2, dL_dm.z);	

									// *** *** *** *** ***

									tmp = ((REAL_G)-0.5900435899266435) * vSH.x * (xx - 3 * yy);
									dL_dparam = tmp * alpha_next;
									tmp2 = 0;

									// dL_dR_3_3
									RSH_band = MAD_G(tmp, GC_SH_11.z, RSH_band);
									if (R_channel) {
										atomicAdd(((REAL_G *)params_OptiX.dL_dparams_SH_11) + (GaussInd * 4) + 2, dL_dparam * d_dR_dalpha);
										tmp2 = MAD_G(GC_SH_11.z, d_dR_dalpha, tmp2);
									}

									// dL_dG_3_3
									GSH_band = MAD_G(tmp, GC_SH_11.w, GSH_band);
									if (G_channel) {
										atomicAdd(((REAL_G *)params_OptiX.dL_dparams_SH_11) + (GaussInd * 4) + 3, dL_dparam * d_dG_dalpha);
										tmp2 = MAD_G(GC_SH_11.w, d_dG_dalpha, tmp2);
									}

									// *** *** *** *** ***

									if constexpr (SH_degree >= 4) {
										float4 GC_SH_12 = params_OptiX.GC_SH_12[GaussInd];
										float4 GC_SH_13 = params_OptiX.GC_SH_13[GaussInd];
										float4 GC_SH_14 = params_OptiX.GC_SH_14[GaussInd];
										float4 GC_SH_15 = params_OptiX.GC_SH_15[GaussInd];
										float4 GC_SH_16 = params_OptiX.GC_SH_16[GaussInd];
										float4 GC_SH_17 = params_OptiX.GC_SH_17[GaussInd];
										float4 GC_SH_18 = params_OptiX.GC_SH_18[GaussInd];

										// *** *** *** *** ***

										// !!! !!! !!!
										// dL_dB_3_3
										BSH_band = MAD_G(tmp, GC_SH_12.x, BSH_band);
										if (B_channel) {
											atomicAdd(((REAL_G *)params_OptiX.dL_dparams_SH_12) + (GaussInd * 4), dL_dparam * d_dB_dalpha);
											tmp2 = MAD_G(GC_SH_12.x, d_dB_dalpha, tmp2);
										}

										// xxx - 3xyy
										// dmx: -3kx(xxx - 3xyy) + k(3xx - 3yy)
										// dmy: -3ky(xxx - 3xyy) + k(-6xy)
										// dmz: -3kz(xxx - 3xyy)
										dL_dm.x = MAD_G(((REAL_G)-0.5900435899266435) * ( 3 * xx - 3 * yy), tmp2, dL_dm.x);
										dL_dm.y = MAD_G(((REAL_G)-0.5900435899266435) * (-6 * xy         ), tmp2, dL_dm.y);

										// *** *** *** *** ***

										RSH += RSH_band;
										GSH += GSH_band;
										BSH += BSH_band;

										tmp2 = 0;
										if (R_channel) tmp2 = MAD_G(RSH_band, d_dR_dalpha, tmp2);
										if (G_channel) tmp2 = MAD_G(GSH_band, d_dG_dalpha, tmp2);
										if (B_channel) tmp2 = MAD_G(BSH_band, d_dB_dalpha, tmp2);
										tmp2 *= -3; // !!! !!! !!!

										dL_dm.x = MAD_G(vSH.x, tmp2, dL_dm.x);
										dL_dm.y = MAD_G(vSH.y, tmp2, dL_dm.y);
										dL_dm.z = MAD_G(vSH.z, tmp2, dL_dm.z);

										RSH_band = 0;
										GSH_band = 0;
										BSH_band = 0;

										// *** *** *** *** ***

										if (params_OptiX.epoch >= SH4_activation_iter) {
											tmp = ((REAL_G)2.50334294179670454) * xy * (xx - yy);
											dL_dparam = tmp * alpha_next;
											tmp2 = 0;

											// dL_dR_4_(-4)
											RSH_band = MAD_G(tmp, GC_SH_12.y, RSH_band);
											if (R_channel) {
												atomicAdd(((REAL_G *)params_OptiX.dL_dparams_SH_12) + (GaussInd * 4) + 1, dL_dparam * d_dR_dalpha);
												tmp2 = MAD_G(GC_SH_12.y, d_dR_dalpha, tmp2);
											}

											// dL_dG_4_(-4)
											GSH_band = MAD_G(tmp, GC_SH_12.z, GSH_band);
											if (G_channel) {
												atomicAdd(((REAL_G *)params_OptiX.dL_dparams_SH_12) + (GaussInd * 4) + 2, dL_dparam * d_dG_dalpha);
												tmp2 = MAD_G(GC_SH_12.z, d_dG_dalpha, tmp2);
											}

											// dL_dB_4_(-4)
											BSH_band = MAD_G(tmp, GC_SH_12.w, BSH_band);
											if (B_channel) {
												atomicAdd(((REAL_G *)params_OptiX.dL_dparams_SH_12) + (GaussInd * 4) + 3, dL_dparam * d_dB_dalpha);
												tmp2 = MAD_G(GC_SH_12.w, d_dB_dalpha, tmp2);
											}

											// xxxy - xyyy
											// dmx: -4kx(xxxy - xyyy) + k(3xxy - yyy)
											// dmy: -4ky(xxxy - xyyy) + k(xxx - 3xyy)
											// dmz: -4kz(xxxy - xyyy)
											dL_dm.x = MAD_G(((REAL_G)2.50334294179670454) * (vSH.y * (3 * xx -     yy)), tmp2, dL_dm.x);
											dL_dm.y = MAD_G(((REAL_G)2.50334294179670454) * (vSH.x * (    xx - 3 * yy)), tmp2, dL_dm.y);

											// *** *** *** *** ***

											tmp = ((REAL_G)-1.77013076977993053) * yz * (3 * xx - yy);							
											dL_dparam = tmp * alpha_next;
											tmp2 = 0;

											// dL_dR_4_(-3)
											RSH_band = MAD_G(tmp, GC_SH_13.x, RSH_band);
											if (R_channel) {
												atomicAdd(((REAL_G *)params_OptiX.dL_dparams_SH_13) + (GaussInd * 4)    , dL_dparam * d_dR_dalpha);
												tmp2 = MAD_G(GC_SH_13.x, d_dR_dalpha, tmp2);
											}

											// dL_dG_4_(-3)
											GSH_band = MAD_G(tmp, GC_SH_13.y, GSH_band);
											if (G_channel) {
												atomicAdd(((REAL_G *)params_OptiX.dL_dparams_SH_13) + (GaussInd * 4) + 1, dL_dparam * d_dG_dalpha);
												tmp2 = MAD_G(GC_SH_13.y, d_dG_dalpha, tmp2);
											}

											// dL_dB_4_(-3)
											BSH_band = MAD_G(tmp, GC_SH_13.z, BSH_band);
											if (B_channel) {
												atomicAdd(((REAL_G *)params_OptiX.dL_dparams_SH_13) + (GaussInd * 4) + 2, dL_dparam * d_dB_dalpha);
												tmp2 = MAD_G(GC_SH_13.z, d_dB_dalpha, tmp2);
											}

											// 3xxyz - yyyz
											// dmx: -4kx(3xxyz - yyyz) + k(6xyz)
											// dmy: -4ky(3xxyz - yyyz) + k(3xxz - 3yyz)
											// dmz: -4kz(3xxyz - yyyz) + k(3xxy - yyy)
											dL_dm.x = MAD_G(((REAL_G)-1.77013076977993053) * (vSH.x * (6 * yz         )), tmp2, dL_dm.x);
											dL_dm.y = MAD_G(((REAL_G)-1.77013076977993053) * (vSH.z * (3 * xx - 3 * yy)), tmp2, dL_dm.y);
											dL_dm.z = MAD_G(((REAL_G)-1.77013076977993053) * (vSH.y * (3 * xx -     yy)), tmp2, dL_dm.z);

											// *** *** *** *** ***

											tmp = ((REAL_G)0.94617469575756002) * xy * (7 * zz - 1);
											dL_dparam = tmp * alpha_next;
											tmp2 = 0;

											// dL_dR_4_(-2)
											RSH_band = MAD_G(tmp, GC_SH_13.w, RSH_band);
											if (R_channel) {
												atomicAdd(((REAL_G *)params_OptiX.dL_dparams_SH_13) + (GaussInd * 4) + 3, dL_dparam * d_dR_dalpha);
												tmp2 = MAD_G(GC_SH_13.w, d_dR_dalpha, tmp2);
											}

											// dL_dG_4_(-2)
											GSH_band = MAD_G(tmp, GC_SH_14.x, GSH_band);
											if (G_channel) {
												atomicAdd(((REAL_G *)params_OptiX.dL_dparams_SH_14) + (GaussInd * 4)    , dL_dparam * d_dG_dalpha);
												tmp2 = MAD_G(GC_SH_14.x, d_dG_dalpha, tmp2);
											}

											// dL_dB_4_(-2)
											BSH_band = MAD_G(tmp, GC_SH_14.y, BSH_band);
											if (B_channel) {
												atomicAdd(((REAL_G *)params_OptiX.dL_dparams_SH_14) + (GaussInd * 4) + 1, dL_dparam * d_dB_dalpha);
												tmp2 = MAD_G(GC_SH_14.y, d_dB_dalpha, tmp2);
											}

											// 7xyzz - xy
											// dmx: -4kx(7xyzz - xy) + k(7yzz - y)
											// dmy: -4ky(7xyzz - xy) + k(7xzz - x)
											// dmz: -4kz(7xyzz - xy) + k(14xyz)
											dL_dm.x = MAD_G(((REAL_G)0.94617469575756002) * (vSH.y * ( 7 * zz - 1)), tmp2, dL_dm.x);
											dL_dm.y = MAD_G(((REAL_G)0.94617469575756002) * (vSH.x * ( 7 * zz - 1)), tmp2, dL_dm.y);
											dL_dm.z = MAD_G(((REAL_G)0.94617469575756002) * (vSH.x * (14 * yz    )), tmp2, dL_dm.z);

											// *** *** *** *** ***

											tmp = ((REAL_G)-0.66904654355728917) * yz * (7 * zz - 3);
											dL_dparam = tmp * alpha_next;
											tmp2 = 0;

											// dL_dR_4_(-1)
											RSH_band = MAD_G(tmp, GC_SH_14.z, RSH_band);
											if (R_channel) {
												atomicAdd(((REAL_G *)params_OptiX.dL_dparams_SH_14) + (GaussInd * 4) + 2, dL_dparam * d_dR_dalpha);
												tmp2 = MAD_G(GC_SH_14.z, d_dR_dalpha, tmp2);
											}

											// dL_dG_4_(-1)
											GSH_band = MAD_G(tmp, GC_SH_14.w, GSH_band);
											if (G_channel) {
												atomicAdd(((REAL_G *)params_OptiX.dL_dparams_SH_14) + (GaussInd * 4) + 3, dL_dparam * d_dG_dalpha);
												tmp2 = MAD_G(GC_SH_14.w, d_dG_dalpha, tmp2);
											}

											// dL_dB_4_(-1)
											BSH_band = MAD_G(tmp, GC_SH_15.x, BSH_band);
											if (B_channel) {
												atomicAdd(((REAL_G *)params_OptiX.dL_dparams_SH_15) + (GaussInd * 4)    , dL_dparam * d_dB_dalpha);
												tmp2 = MAD_G(GC_SH_15.x, d_dB_dalpha, tmp2);
											}

											// 7yzzz - 3yz
											// dmx: -4kx(7yzzz - 3yz)
											// dmy: -4ky(7yzzz - 3yz) + k(7zzz - 3z)
											// dmz: -4kz(7yzzz - 3yz) + k(21yzz - 3y)
											dL_dm.y = MAD_G(((REAL_G)-0.66904654355728917) * (vSH.z * ( 7 * zz - 3)), tmp2, dL_dm.y);
											dL_dm.z = MAD_G(((REAL_G)-0.66904654355728917) * (vSH.y * (21 * zz - 3)), tmp2, dL_dm.z);

											// *** *** *** *** ***

											tmp = ((REAL_G)0.10578554691520430) * ((zz * (35 * zz - 30)) + 3);
											dL_dparam = tmp * alpha_next;
											tmp2 = 0;

											// dL_dR_4_0
											RSH_band = MAD_G(tmp, GC_SH_15.y, RSH_band);
											if (R_channel) {
												atomicAdd(((REAL_G *)params_OptiX.dL_dparams_SH_15) + (GaussInd * 4) + 1, dL_dparam * d_dR_dalpha);
												tmp2 = MAD_G(GC_SH_15.y, d_dR_dalpha, tmp2);
											}

											// dL_dG_4_0
											GSH_band = MAD_G(tmp, GC_SH_15.z, GSH_band);
											if (G_channel) {
												atomicAdd(((REAL_G *)params_OptiX.dL_dparams_SH_15) + (GaussInd * 4) + 2, dL_dparam * d_dG_dalpha);
												tmp2 = MAD_G(GC_SH_15.z, d_dG_dalpha, tmp2);
											}

											// dL_dB_4_0
											BSH_band = MAD_G(tmp, GC_SH_15.w, BSH_band);
											if (B_channel) {
												atomicAdd(((REAL_G *)params_OptiX.dL_dparams_SH_15) + (GaussInd * 4) + 3, dL_dparam * d_dB_dalpha);
												tmp2 = MAD_G(GC_SH_15.w, d_dB_dalpha, tmp2);
											}

											// 35zzzz - 30zz + 3
											// dmx: -4kx(35zzzz - 30zz + 3)
											// dmy: -4ky(35zzzz - 30zz + 3)
											// dmz: -4kz(35zzzz - 30zz + 3) + k(140zzz - 60z)
											dL_dm.z = MAD_G(((REAL_G)0.10578554691520430) * (vSH.z * (140 * zz - 60)), tmp2, dL_dm.z);

											// *** *** *** *** ***

											tmp = ((REAL_G)-0.66904654355728917) * xz * (7 * zz - 3);					
											dL_dparam = tmp * alpha_next;
											tmp2 = 0;

											// dL_dR_4_1
											RSH_band = MAD_G(tmp, GC_SH_16.x, RSH_band);
											if (R_channel) {
												atomicAdd(((REAL_G *)params_OptiX.dL_dparams_SH_16) + (GaussInd * 4)    , dL_dparam * d_dR_dalpha);
												tmp2 = MAD_G(GC_SH_16.x, d_dR_dalpha, tmp2);
											}

											// dL_dG_4_1
											GSH_band = MAD_G(tmp, GC_SH_16.y, GSH_band);
											if (G_channel) {
												atomicAdd(((REAL_G *)params_OptiX.dL_dparams_SH_16) + (GaussInd * 4) + 1, dL_dparam * d_dG_dalpha);
												tmp2 = MAD_G(GC_SH_16.y, d_dG_dalpha, tmp2);
											}

											// dL_dB_4_1
											BSH_band = MAD_G(tmp, GC_SH_16.z, BSH_band);
											if (B_channel) {
												atomicAdd(((REAL_G *)params_OptiX.dL_dparams_SH_16) + (GaussInd * 4) + 2, dL_dparam * d_dB_dalpha);
												tmp2 = MAD_G(GC_SH_16.z, d_dB_dalpha, tmp2);
											}

											// 7xzzz - 3xz
											// dmx: -4kx(7xzzz - 3xz) + k(7zzz - 3z)
											// dmy: -4ky(7xzzz - 3xz)
											// dmz: -4kz(7xzzz - 3xz) + k(21xzz - 3x)
											dL_dm.x = MAD_G(((REAL_G)-0.66904654355728917) * (vSH.z * ( 7 * zz - 3)), tmp2, dL_dm.x);
											dL_dm.z = MAD_G(((REAL_G)-0.66904654355728917) * (vSH.x * (21 * zz - 3)), tmp2, dL_dm.z);

											// *** *** *** *** ***

											tmp = ((REAL_G)0.47308734787878001) * (xx - yy) * (7 * zz - 1);		
											dL_dparam = tmp * alpha_next;
											tmp2 = 0;

											// dL_dR_4_2
											RSH_band = MAD_G(tmp, GC_SH_16.w, RSH_band);
											if (R_channel) {
												atomicAdd(((REAL_G *)params_OptiX.dL_dparams_SH_16) + (GaussInd * 4) + 3, dL_dparam * d_dR_dalpha);
												tmp2 = MAD_G(GC_SH_16.w, d_dR_dalpha, tmp2);
											}

											// dL_dG_4_2
											GSH_band = MAD_G(tmp, GC_SH_17.x, GSH_band);
											if (G_channel) {
												atomicAdd(((REAL_G *)params_OptiX.dL_dparams_SH_17) + (GaussInd * 4)    , dL_dparam * d_dG_dalpha);
												tmp2 = MAD_G(GC_SH_17.x, d_dG_dalpha, tmp2);
											}

											// dL_dB_4_2
											BSH_band = MAD_G(tmp, GC_SH_17.y, BSH_band);
											if (B_channel) {
												atomicAdd(((REAL_G *)params_OptiX.dL_dparams_SH_17) + (GaussInd * 4) + 1, dL_dparam * d_dB_dalpha);
												tmp2 = MAD_G(GC_SH_17.y, d_dB_dalpha, tmp2);
											}

											// 7xxzz - 7yyzz - xx + yy
											// dmx: -4kx(7xxzz - 7yyzz - xx + yy) + k(14xzz - 2x)
											// dmy: -4ky(7xxzz - 7yyzz - xx + yy) + k(-14yzz + 2y)
											// dmz: -4kz(7xxzz - 7yyzz - xx + yy) + k(14xxz - 14yyz)
											dL_dm.x = MAD_G(((REAL_G)0.47308734787878001) * (vSH.x * ( 14 * zz - 2      )), tmp2, dL_dm.x);
											dL_dm.y = MAD_G(((REAL_G)0.47308734787878001) * (vSH.y * (-14 * zz + 2      )), tmp2, dL_dm.y);
											dL_dm.z = MAD_G(((REAL_G)0.47308734787878001) * (vSH.z * ( 14 * xx - 14 * yy)), tmp2, dL_dm.z);

											// *** *** *** *** ***

											tmp = ((REAL_G)-1.77013076977993053) * xz * (xx - 3 * yy);					
											dL_dparam = tmp * alpha_next;
											tmp2 = 0;

											// dL_dR_4_3
											RSH_band = MAD_G(tmp, GC_SH_17.z, RSH_band);
											if (R_channel) {
												atomicAdd(((REAL_G *)params_OptiX.dL_dparams_SH_17) + (GaussInd * 4) + 2, dL_dparam * d_dR_dalpha);
												tmp2 = MAD_G(GC_SH_17.z, d_dR_dalpha, tmp2);
											}

											// dL_dG_4_3
											GSH_band = MAD_G(tmp, GC_SH_17.w, GSH_band);
											if (G_channel) {
												atomicAdd(((REAL_G *)params_OptiX.dL_dparams_SH_17) + (GaussInd * 4) + 3, dL_dparam * d_dG_dalpha);
												tmp2 = MAD_G(GC_SH_17.w, d_dG_dalpha, tmp2);
											}

											// dL_dB_4_3
											BSH_band = MAD_G(tmp, GC_SH_18.x, BSH_band);
											if (B_channel) {
												atomicAdd(((REAL_G *)params_OptiX.dL_dparams_SH_18) + (GaussInd * 4)    , dL_dparam * d_dB_dalpha);
												tmp2 = MAD_G(GC_SH_18.x, d_dB_dalpha, tmp2);
											}

											// xxxz - 3xyyz
											// dmx: -4kx(xxxz - 3xyyz) + k(3xxz - 3yyz)
											// dmy: -4ky(xxxz - 3xyyz) + k(-6xyz)
											// dmz: -4kz(xxxz - 3xyyz) + k(xxx - 3xyy)
											dL_dm.x = MAD_G(((REAL_G)-1.77013076977993053) * (vSH.z * ( 3 * xx - 3 * yy)), tmp2, dL_dm.x);
											dL_dm.y = MAD_G(((REAL_G)-1.77013076977993053) * (vSH.x * (-6 * yz         )), tmp2, dL_dm.y);
											dL_dm.z = MAD_G(((REAL_G)-1.77013076977993053) * (vSH.x * (     xx - 3 * yy)), tmp2, dL_dm.z);

											// *** *** *** *** ***

											tmp = ((REAL_G)0.62583573544917613) * ((xx * (xx - 3 * yy)) - (yy * (3 * xx - yy)));
											dL_dparam = tmp * alpha_next;
											tmp2 = 0;

											// dL_dR_4_4
											RSH_band = MAD_G(tmp, GC_SH_18.y, RSH_band);
											if (R_channel) {
												atomicAdd(((REAL_G *)params_OptiX.dL_dparams_SH_18) + (GaussInd * 4) + 1, dL_dparam * d_dR_dalpha);
												tmp2 = MAD_G(GC_SH_18.y, d_dR_dalpha, tmp2);
											}

											// dL_dG_4_4
											GSH_band = MAD_G(tmp, GC_SH_18.z, GSH_band);
											if (G_channel) {
												atomicAdd(((REAL_G *)params_OptiX.dL_dparams_SH_18) + (GaussInd * 4) + 2, dL_dparam * d_dG_dalpha);
												tmp2 = MAD_G(GC_SH_18.z, d_dG_dalpha, tmp2);
											}

											// dL_dB_4_4
											BSH_band = MAD_G(tmp, GC_SH_18.w, BSH_band);
											if (B_channel) {
												atomicAdd(((REAL_G *)params_OptiX.dL_dparams_SH_18) + (GaussInd * 4) + 3, dL_dparam * d_dB_dalpha);
												tmp2 = MAD_G(GC_SH_18.w, d_dB_dalpha, tmp2);
											}

											// xxxx - 6xxyy + yyyy
											// dmx: -4kx(xxxx - 6xxyy + yyyy) + k(4xxx - 12xyy)
											// dmy: -4ky(xxxx - 6xxyy + yyyy) + k(-12xxy + 4yyy)
											// dmz: -4kz(xxxx - 6xxyy + yyyy)
											dL_dm.x = MAD_G(((REAL_G)0.62583573544917613) * (vSH.x * (  4 * xx - 12 * yy)), tmp2, dL_dm.x);
											dL_dm.y = MAD_G(((REAL_G)0.62583573544917613) * (vSH.y * (-12 * xx +  4 * yy)), tmp2, dL_dm.y);

											// *** *** *** *** ***

											RSH += RSH_band;
											GSH += GSH_band;
											BSH += BSH_band;

											tmp2 = 0;
											if (R_channel) tmp2 = MAD_G(RSH_band, d_dR_dalpha, tmp2);
											if (G_channel) tmp2 = MAD_G(GSH_band, d_dG_dalpha, tmp2);
											if (B_channel) tmp2 = MAD_G(BSH_band, d_dB_dalpha, tmp2);
											tmp2 *= -4; // !!! !!! !!!

											dL_dm.x = MAD_G(vSH.x, tmp2, dL_dm.x);
											dL_dm.y = MAD_G(vSH.y, tmp2, dL_dm.y);
											dL_dm.z = MAD_G(vSH.z, tmp2, dL_dm.z);
										}
									} else {
										float GC_SH_12 = params_OptiX.GC_SH_12[GaussInd];

										// *** *** *** *** ***
							
										// !!! !!! !!!
										// dL_dB_3_3
										BSH_band = MAD_G(tmp, GC_SH_12, BSH_band);
										if (B_channel) {
											atomicAdd(((REAL_G *)params_OptiX.dL_dparams_SH_12) + GaussInd, dL_dparam * d_dB_dalpha);
											tmp2 = MAD_G(GC_SH_12, d_dB_dalpha, tmp2);
										}

										// xxx - 3xyy
										// dmx: -3kx(xxx - 3xyy) + k(3xx - 3yy)
										// dmy: -3ky(xxx - 3xyy) + k(-6xy)
										// dmz: -3kz(xxx - 3xyy)
										dL_dm.x = MAD_G(((REAL_G)-0.5900435899266435) * ( 3 * xx - 3 * yy), tmp2, dL_dm.x);
										dL_dm.y = MAD_G(((REAL_G)-0.5900435899266435) * (-6 * xy         ), tmp2, dL_dm.y);
						
										// *** *** *** *** ***

										RSH += RSH_band;
										GSH += GSH_band;
										BSH += BSH_band;

										tmp2 = 0;
										if (R_channel) tmp2 = MAD_G(RSH_band, d_dR_dalpha, tmp2);
										if (G_channel) tmp2 = MAD_G(GSH_band, d_dG_dalpha, tmp2);
										if (B_channel) tmp2 = MAD_G(BSH_band, d_dB_dalpha, tmp2);
										tmp2 *= -3; // !!! !!! !!!

										dL_dm.x = MAD_G(vSH.x, tmp2, dL_dm.x);
										dL_dm.y = MAD_G(vSH.y, tmp2, dL_dm.y);
										dL_dm.z = MAD_G(vSH.z, tmp2, dL_dm.z);
									}
								}
							}
						}
					} else {
						float GC_SH_3 = params_OptiX.GC_SH_3[GaussInd];

						// *** *** *** *** ***

						// !!! !!! !!!
						// dL_dB_1_1
						BSH_band = MAD_G(tmp, GC_SH_3, BSH_band);
						if (B_channel) {
							atomicAdd(((REAL_G *)params_OptiX.dL_dparams_SH_3) + GaussInd, dL_dparam * d_dB_dalpha);
							tmp2 = MAD_G(GC_SH_3, d_dB_dalpha, tmp2);
						}

						dL_dm.x = MAD_G(((REAL_G)-0.4886025119029199), tmp2, dL_dm.x);

						// *** *** *** *** ***

						RSH += RSH_band;
						GSH += GSH_band;
						BSH += BSH_band;

						tmp2 = 0;
						if (R_channel) tmp2 = MAD_G(RSH_band, d_dR_dalpha, tmp2);
						if (G_channel) tmp2 = MAD_G(GSH_band, d_dG_dalpha, tmp2);
						if (B_channel) tmp2 = MAD_G(BSH_band, d_dB_dalpha, tmp2);
						tmp2 = -tmp2; // !!! !!! !!!

						dL_dm.x = MAD_G(vSH.x, tmp2, dL_dm.x);
						dL_dm.y = MAD_G(vSH.y, tmp2, dL_dm.y);
						dL_dm.z = MAD_G(vSH.z, tmp2, dL_dm.z);
					}

					// !!! !!! !!!
					dL_dm.x *= vSH_norm_inv;
					dL_dm.y *= vSH_norm_inv;
					dL_dm.z *= vSH_norm_inv;
				}
			}

			// *************************************************************************************

			// !!! !!! !!!
			RSH = RSH + ((REAL_G)0.5);
			GSH = GSH + ((REAL_G)0.5);
			BSH = BSH + ((REAL_G)0.5);

			RSH = (RSH < 0) ? 0 : RSH;
			GSH = (GSH < 0) ? 0 : GSH;
			BSH = (BSH < 0) ? 0 : BSH;
			// !!! !!! !!!

			// *************************************************************************************

			// !!! !!! !!!
			tmp3 = dI_dalpha - d_dI_dalpha_prev; // !!! !!! !!!
			d_dI_dalpha_prev = MAD_G(RSH, d_dR_dalpha, MAD_G(GSH, d_dG_dalpha, BSH * d_dB_dalpha)); // !!! !!! !!!
			dI_dalpha = tmp3 + d_dI_dalpha_prev; // !!! !!! !!!

			tmp3 = dI_dalpha / (1 - alpha_next);
			tmp3 = (tmp3 < -dI_dalpha_max) ? -dI_dalpha_max : tmp3;
			tmp3 = (tmp3 >  dI_dalpha_max) ?  dI_dalpha_max : tmp3;
			// !!! !!! !!!

			// *************************************************************************************

			// !!! !!! !!!
			if (isnan(tmp3)) {
				tmp3 = d_dI_dalpha_prev;
				k = max_Gaussians_per_ray; // !!! !!! !!!
			}
			// !!! !!! !!!

			// *************************************************************************************

			tmp3 = (tmp3 * alpha_next);

			// *************************************************************************************

			// dL_dalpha
			dL_dparam = tmp3 * (1.0f - tmp2);
			atomicAdd(((REAL_G *)params_OptiX.dL_dparams_1) + (GaussInd * 4) + 3, dL_dparam);

			// *************************************************************************************

			tmp3 = tmp3 / (s * s);

			// *************************************************************************************

			vecx_tmp = tmp3 * vecx_tmp; // !!! !!! !!!
			vecy_tmp = tmp3 * vecy_tmp; // !!! !!! !!!
			vecz_tmp = tmp3 * vecz_tmp; // !!! !!! !!!

			// *************************************************************************************

			// dL_dsX
			/*REAL_G dot_product_1 = MAD_G(Ox, R11, MAD_G(Oy, R21, Oz * R31));
			REAL_G dot_product_2 = MAD_G(v.x, R11, MAD_G(v.y, R21, v.z * R31));
			dL_dparam = vecx_tmp * sXInv * MAD_G(-tmp1, dot_product_2, dot_product_1);
			atomicAdd(((REAL_G *)params_OptiX.dL_dparams_2) + (GaussInd * 4) + 3, dL_dparam);*/

			// dL_dsY
			REAL_G dot_product_1 = MAD_G(Ox, R12, MAD_G(Oy, R22, Oz * R32));
			REAL_G dot_product_2 = MAD_G(v.x, R12, MAD_G(v.y, R22, v.z * R32));
			dL_dparam = vecy_tmp * sYInv * MAD_G(-tmp1, dot_product_2, dot_product_1);
			atomicAdd(((REAL_G *)params_OptiX.dL_dparams_3) + (GaussInd * 4), dL_dparam);

			// dL_dsZ
			dot_product_1 = MAD_G(Ox, R13, MAD_G(Oy, R23, Oz * R33));
			dot_product_2 = MAD_G(v.x, R13, MAD_G(v.y, R23, v.z * R33));
			dL_dparam = vecz_tmp * sZInv * MAD_G(-tmp1, dot_product_2, dot_product_1);
			atomicAdd(((REAL_G *)params_OptiX.dL_dparams_3) + (GaussInd * 4) + 1, dL_dparam);

			// *************************************************************************************

			// !!! !!! !!!
			vecx_tmp = vecx_tmp * sXInv;
			vecy_tmp = vecy_tmp * sYInv;
			vecz_tmp = vecz_tmp * sZInv;
			// !!! !!! !!!

			// *************************************************************************************

			// dL_dmX
			dL_dparam = (dL_dm.x * alpha_next) + MAD_G(vecx_tmp, R11, MAD_G(vecy_tmp, R12, vecz_tmp * R13));
			atomicAdd(((REAL_G *)params_OptiX.dL_dparams_2) + (GaussInd * 4), dL_dparam);

			// dL_dmY
			dL_dparam = (dL_dm.y * alpha_next) + MAD_G(vecx_tmp, R21, MAD_G(vecy_tmp, R22, vecz_tmp * R23));
			atomicAdd(((REAL_G *)params_OptiX.dL_dparams_2) + (GaussInd * 4) + 1, dL_dparam);

			// dL_dmZ
			dL_dparam = (dL_dm.z * alpha_next) + MAD_G(vecx_tmp, R31, MAD_G(vecy_tmp, R32, vecz_tmp * R33));
			atomicAdd(((REAL_G *)params_OptiX.dL_dparams_2) + (GaussInd * 4) + 2, dL_dparam);

			// *************************************************************************************

			REAL_G tmp7 = 1 - (aa / s);
			REAL_G tmp8 = 1 - (bb / s);
			REAL_G tmp9 = 1 - (cc / s);
			REAL_G tmp10 = 1 - (dd / s);

			cd = -cd / s; // !!! !!! !!!
			ab = -ab / s; // !!! !!! !!!
			REAL_G a_inv = GC_3.w * cd; // (-bcd) / s
			REAL_G b_inv = GC_3.z * cd; // (-acd) / s
			REAL_G c_inv = ab * GC_4.y; // (-abd) / s
			REAL_G d_inv = ab * GC_4.x; // (-abc) / s

			// *************************************************************************************

			// dL_da
			REAL_G dR11_da = GC_3.z * (tmp7 + tmp8);
			REAL_G dR12_da = MAD_G(-GC_4.y, tmp7, d_inv);
			REAL_G dR13_da = MAD_G(GC_4.x, tmp7, c_inv);

			REAL_G dR21_da = MAD_G(GC_4.y, tmp7, d_inv);
			REAL_G dR22_da = GC_3.z * (tmp7 + tmp9);
			REAL_G dR23_da = MAD_G(-GC_3.w, tmp7, b_inv);

			REAL_G dR31_da = MAD_G(-GC_4.x, tmp7, c_inv);
			REAL_G dR32_da = MAD_G(GC_3.w, tmp7, b_inv);
			REAL_G dR33_da = GC_3.z * (tmp7 + tmp10);

			REAL_G vecx_tmp2 = MAD_G(vecx_tmp, dR11_da, MAD_G(vecy_tmp, dR12_da, vecz_tmp * dR13_da));
			REAL_G vecy_tmp2 = MAD_G(vecx_tmp, dR21_da, MAD_G(vecy_tmp, dR22_da, vecz_tmp * dR23_da));
			REAL_G vecz_tmp2 = MAD_G(vecx_tmp, dR31_da, MAD_G(vecy_tmp, dR32_da, vecz_tmp * dR33_da));

			dot_product_1 = MAD_G(vecx_tmp2, Ox, MAD_G(vecy_tmp2, Oy, vecz_tmp2 * Oz));
			dot_product_2 = MAD_G(vecx_tmp2, v.x, MAD_G(vecy_tmp2, v.y, vecz_tmp2 * v.z));

			dL_dparam = MAD_G(tmp1, dot_product_2, -dot_product_1);
			atomicAdd(((REAL_G *)params_OptiX.dL_dparams_3) + (GaussInd * 4) + 2, dL_dparam);

			// *** *** *** *** ***

			// dL_db
			dR11_da = GC_3.w * (tmp8 + tmp7);
			dR12_da = MAD_G(GC_4.x, tmp8, -c_inv);
			dR13_da = MAD_G(GC_4.y, tmp8, d_inv);

			dR21_da = MAD_G(GC_4.x, tmp8, c_inv);
			dR22_da = -GC_3.w * (tmp8 + tmp10);
			dR23_da = MAD_G(-GC_3.z, tmp8, a_inv);

			dR31_da = MAD_G(GC_4.y, tmp8, -d_inv);
			dR32_da = MAD_G(GC_3.z, tmp8, a_inv);
			dR33_da = -GC_3.w * (tmp8 + tmp9);

			vecx_tmp2 = MAD_G(vecx_tmp, dR11_da, MAD_G(vecy_tmp, dR12_da, vecz_tmp * dR13_da));
			vecy_tmp2 = MAD_G(vecx_tmp, dR21_da, MAD_G(vecy_tmp, dR22_da, vecz_tmp * dR23_da));
			vecz_tmp2 = MAD_G(vecx_tmp, dR31_da, MAD_G(vecy_tmp, dR32_da, vecz_tmp * dR33_da));

			dot_product_1 = MAD_G(vecx_tmp2, Ox, MAD_G(vecy_tmp2, Oy, vecz_tmp2 * Oz));
			dot_product_2 = MAD_G(vecx_tmp2, v.x, MAD_G(vecy_tmp2, v.y, vecz_tmp2 * v.z));

			dL_dparam = MAD_G(tmp1, dot_product_2, -dot_product_1);
			atomicAdd(((REAL_G *)params_OptiX.dL_dparams_3) + (GaussInd * 4) + 3, dL_dparam);

			// *** *** *** *** ***

			// dL_dc
			dR11_da = -GC_4.x * (tmp9 + tmp10);
			dR12_da = MAD_G(GC_3.w, tmp9, -b_inv);
			dR13_da = MAD_G(GC_3.z, tmp9, a_inv);

			dR21_da = MAD_G(GC_3.w, tmp9, b_inv);
			dR22_da = GC_4.x * (tmp9 + tmp7);
			dR23_da = MAD_G(GC_4.y, tmp9, -d_inv);

			dR31_da = MAD_G(-GC_3.z, tmp9, a_inv);
			dR32_da = MAD_G(GC_4.y, tmp9, d_inv);
			dR33_da = -GC_4.x * (tmp9 + tmp8);

			vecx_tmp2 = MAD_G(vecx_tmp, dR11_da, MAD_G(vecy_tmp, dR12_da, vecz_tmp * dR13_da));
			vecy_tmp2 = MAD_G(vecx_tmp, dR21_da, MAD_G(vecy_tmp, dR22_da, vecz_tmp * dR23_da));
			vecz_tmp2 = MAD_G(vecx_tmp, dR31_da, MAD_G(vecy_tmp, dR32_da, vecz_tmp * dR33_da));

			dot_product_1 = MAD_G(vecx_tmp2, Ox, MAD_G(vecy_tmp2, Oy, vecz_tmp2 * Oz));
			dot_product_2 = MAD_G(vecx_tmp2, v.x, MAD_G(vecy_tmp2, v.y, vecz_tmp2 * v.z));

			dL_dparam = MAD_G(tmp1, dot_product_2, -dot_product_1);
			atomicAdd(((REAL_G *)params_OptiX.dL_dparams_4) + (GaussInd * 2), dL_dparam);

			// *** *** *** *** ***

			// dL_dd
			dR11_da = -GC_4.y * (tmp10 + tmp9);
			dR12_da = MAD_G(-GC_3.z, tmp10, a_inv);
			dR13_da = MAD_G(GC_3.w, tmp10, b_inv);

			dR21_da = MAD_G(GC_3.z, tmp10, a_inv);
			dR22_da = -GC_4.y * (tmp10 + tmp8);
			dR23_da = MAD_G(GC_4.x, tmp10, -c_inv);

			dR31_da = MAD_G(GC_3.w, tmp10, -b_inv);
			dR32_da = MAD_G(GC_4.x, tmp10, c_inv);
			dR33_da = GC_4.y * (tmp10 + tmp7);

			vecx_tmp2 = MAD_G(vecx_tmp, dR11_da, MAD_G(vecy_tmp, dR12_da, vecz_tmp * dR13_da));
			vecy_tmp2 = MAD_G(vecx_tmp, dR21_da, MAD_G(vecy_tmp, dR22_da, vecz_tmp * dR23_da));
			vecz_tmp2 = MAD_G(vecx_tmp, dR31_da, MAD_G(vecy_tmp, dR32_da, vecz_tmp * dR33_da));

			dot_product_1 = MAD_G(vecx_tmp2, Ox, MAD_G(vecy_tmp2, Oy, vecz_tmp2 * Oz));
			dot_product_2 = MAD_G(vecx_tmp2, v.x, MAD_G(vecy_tmp2, v.y, vecz_tmp2 * v.z));

			dL_dparam = MAD_G(tmp1, dot_product_2, -dot_product_1);
			atomicAdd(((REAL_G *)params_OptiX.dL_dparams_4) + (GaussInd * 2) + 1, dL_dparam);

			// *************************************************************************************

			alpha_prev = alpha_next;
		}
	}
}

// *************************************************************************************************

template<int SH_degree>
struct dev_SH {
};

template<>
struct dev_SH<1> {
	float4 GC_SH_1, m_SH_1, v_SH_1;
	float4 GC_SH_2, m_SH_2, v_SH_2;
	float GC_SH_3, m_SH_3, v_SH_3; // !!! !!! !!!

	REAL4_G dL_dparams_SH_1;
	REAL4_G dL_dparams_SH_2;
	REAL_G dL_dparams_SH_3; // !!! !!! !!!
};

template<>
struct dev_SH<2> {
	float4 GC_SH_1, m_SH_1, v_SH_1;
	float4 GC_SH_2, m_SH_2, v_SH_2;
	float4 GC_SH_3, m_SH_3, v_SH_3;
	float4 GC_SH_4, m_SH_4, v_SH_4;
	float4 GC_SH_5, m_SH_5, v_SH_5;
	float4 GC_SH_6, m_SH_6, v_SH_6;

	REAL4_G dL_dparams_SH_1;
	REAL4_G dL_dparams_SH_2;
	REAL4_G dL_dparams_SH_3;
	REAL4_G dL_dparams_SH_4;
	REAL4_G dL_dparams_SH_5;
	REAL4_G dL_dparams_SH_6;
};

template<>
struct dev_SH<3> {
	float4 GC_SH_1, m_SH_1, v_SH_1;
	float4 GC_SH_2, m_SH_2, v_SH_2;
	float4 GC_SH_3, m_SH_3, v_SH_3;
	float4 GC_SH_4, m_SH_4, v_SH_4;
	float4 GC_SH_5, m_SH_5, v_SH_5;
	float4 GC_SH_6, m_SH_6, v_SH_6;
	float4 GC_SH_7, m_SH_7, v_SH_7;
	float4 GC_SH_8, m_SH_8, v_SH_8;
	float4 GC_SH_9, m_SH_9, v_SH_9;
	float4 GC_SH_10, m_SH_10, v_SH_10;
	float4 GC_SH_11, m_SH_11, v_SH_11;
	float GC_SH_12, m_SH_12, v_SH_12; // !!! !!! !!!

	REAL4_G dL_dparams_SH_1;
	REAL4_G dL_dparams_SH_2;
	REAL4_G dL_dparams_SH_3;
	REAL4_G dL_dparams_SH_4;
	REAL4_G dL_dparams_SH_5;
	REAL4_G dL_dparams_SH_6;
	REAL4_G dL_dparams_SH_7;
	REAL4_G dL_dparams_SH_8;
	REAL4_G dL_dparams_SH_9;
	REAL4_G dL_dparams_SH_10;
	REAL4_G dL_dparams_SH_11;
	REAL_G dL_dparams_SH_12; // !!! !!! !!!
};

template<>
struct dev_SH<4> {
	float4 GC_SH_1, m_SH_1, v_SH_1;
	float4 GC_SH_2, m_SH_2, v_SH_2;
	float4 GC_SH_3, m_SH_3, v_SH_3;
	float4 GC_SH_4, m_SH_4, v_SH_4;
	float4 GC_SH_5, m_SH_5, v_SH_5;
	float4 GC_SH_6, m_SH_6, v_SH_6;
	float4 GC_SH_7, m_SH_7, v_SH_7;
	float4 GC_SH_8, m_SH_8, v_SH_8;
	float4 GC_SH_9, m_SH_9, v_SH_9;
	float4 GC_SH_10, m_SH_10, v_SH_10;
	float4 GC_SH_11, m_SH_11, v_SH_11;
	float4 GC_SH_12, m_SH_12, v_SH_12;
	float4 GC_SH_13, m_SH_13, v_SH_13;
	float4 GC_SH_14, m_SH_14, v_SH_14;
	float4 GC_SH_15, m_SH_15, v_SH_15;
	float4 GC_SH_16, m_SH_16, v_SH_16;
	float4 GC_SH_17, m_SH_17, v_SH_17;
	float4 GC_SH_18, m_SH_18, v_SH_18;

	REAL4_G dL_dparams_SH_1;
	REAL4_G dL_dparams_SH_2;
	REAL4_G dL_dparams_SH_3;
	REAL4_G dL_dparams_SH_4;
	REAL4_G dL_dparams_SH_5;
	REAL4_G dL_dparams_SH_6;
	REAL4_G dL_dparams_SH_7;
	REAL4_G dL_dparams_SH_8;
	REAL4_G dL_dparams_SH_9;
	REAL4_G dL_dparams_SH_10;
	REAL4_G dL_dparams_SH_11;
	REAL4_G dL_dparams_SH_12;
	REAL4_G dL_dparams_SH_13;
	REAL4_G dL_dparams_SH_14;
	REAL4_G dL_dparams_SH_15;
	REAL4_G dL_dparams_SH_16;
	REAL4_G dL_dparams_SH_17;
	REAL4_G dL_dparams_SH_18;
};

// *** *** *** *** ***

#define backward_SH_float4(n, lr1, lr2, lr3, lr4) \
	SH.dL_dparams_SH_##n = ((REAL4_G *)params_OptiX.dL_dparams_SH_##n)[tid]; \
	SH.m_SH_##n = ((float4 *)params_OptiX.m_SH_##n)[tid]; \
	SH.v_SH_##n = ((float4 *)params_OptiX.v_SH_##n)[tid]; \
	SH.GC_SH_##n = ((float4 *)params_OptiX.GC_SH_##n)[tid]; \
	\
	SH.m_SH_##n.x = (beta1 * SH.m_SH_##n.x) + ((1.0f - beta1) * SH.dL_dparams_SH_##n.x); \
	SH.m_SH_##n.y = (beta1 * SH.m_SH_##n.y) + ((1.0f - beta1) * SH.dL_dparams_SH_##n.y); \
	SH.m_SH_##n.z = (beta1 * SH.m_SH_##n.z) + ((1.0f - beta1) * SH.dL_dparams_SH_##n.z); \
	SH.m_SH_##n.w = (beta1 * SH.m_SH_##n.w) + ((1.0f - beta1) * SH.dL_dparams_SH_##n.w); \
	\
	SH.v_SH_##n.x = (beta2 * SH.v_SH_##n.x) + ((1.0f - beta2) * SH.dL_dparams_SH_##n.x * SH.dL_dparams_SH_##n.x); \
	SH.v_SH_##n.y = (beta2 * SH.v_SH_##n.y) + ((1.0f - beta2) * SH.dL_dparams_SH_##n.y * SH.dL_dparams_SH_##n.y); \
	SH.v_SH_##n.z = (beta2 * SH.v_SH_##n.z) + ((1.0f - beta2) * SH.dL_dparams_SH_##n.z * SH.dL_dparams_SH_##n.z); \
	SH.v_SH_##n.w = (beta2 * SH.v_SH_##n.w) + ((1.0f - beta2) * SH.dL_dparams_SH_##n.w * SH.dL_dparams_SH_##n.w); \
	\
	SH.GC_SH_##n.x -= ((lr1 * (SH.m_SH_##n.x * tmp1)) / (sqrtf(SH.v_SH_##n.x * tmp2) + epsilon)); \
	SH.GC_SH_##n.y -= ((lr2 * (SH.m_SH_##n.y * tmp1)) / (sqrtf(SH.v_SH_##n.y * tmp2) + epsilon)); \
	SH.GC_SH_##n.z -= ((lr3 * (SH.m_SH_##n.z * tmp1)) / (sqrtf(SH.v_SH_##n.z * tmp2) + epsilon)); \
	SH.GC_SH_##n.w -= ((lr4 * (SH.m_SH_##n.w * tmp1)) / (sqrtf(SH.v_SH_##n.w * tmp2) + epsilon));

// *** *** *** *** ***

#define backward_SH_float(n, lr) \
	SH.dL_dparams_SH_##n = ((REAL_G *)params_OptiX.dL_dparams_SH_##n)[tid]; \
	SH.m_SH_##n = ((float *)params_OptiX.m_SH_##n)[tid]; \
	SH.v_SH_##n = ((float *)params_OptiX.v_SH_##n)[tid]; \
	SH.GC_SH_##n = ((float *)params_OptiX.GC_SH_##n)[tid]; \
	\
	SH.m_SH_##n = (beta1 * SH.m_SH_##n) + ((1.0f - beta1) * SH.dL_dparams_SH_##n); \
	\
	SH.v_SH_##n = (beta2 * SH.v_SH_##n) + ((1.0f - beta2) * SH.dL_dparams_SH_##n * SH.dL_dparams_SH_##n); \
	\
	SH.GC_SH_##n -= ((lr * (SH.m_SH_##n * tmp1)) / (sqrtf(SH.v_SH_##n * tmp2) + epsilon));

// *** *** *** *** ***

#define backward_SH_store(n, index) \
	params_OptiX.GC_SH_##n[index] = SH.GC_SH_##n; \
	params_OptiX.m_SH_##n[index] = SH.m_SH_##n; \
	params_OptiX.v_SH_##n[index] = SH.v_SH_##n;

// *** *** *** *** ***

template<int SH_degree>
__global__ void dev_UpdateGradientOptiX(SOptiXRenderParams<SH_degree> params_OptiX) {
	const float beta1 = 0.9f;
	const float beta2 = 0.999f;
	const float epsilon = 0.00000001f;
	float t = params_OptiX.epoch;

	float tmp1 = 1.0f / (1.0f - powf(beta1, t));
	float tmp2 = 1.0f / (1.0f - powf(beta2, t));

	// *****************************************************************************************

	__shared__ unsigned counter1;
	__shared__ unsigned counter2;

	int tid = (blockIdx.x * blockDim.x) + threadIdx.x;

	// *****************************************************************************************

	REAL4_G dL_dparam_1;
	float4 m1;
	float4 v1;
	float4 GC_1;

	REAL4_G dL_dparam_2;
	float4 m2;
	float4 v2;
	float4 GC_2;

	float3 dm;

	REAL4_G dL_dparam_3;
	float4 m3;
	float4 v3;
	float4 GC_3;

	float3 scale;

	REAL2_G dL_dparam_4;
	float2 m4;
	float2 v4;
	float2 GC_4;

	// Spherical harmonics
	dev_SH<SH_degree> SH;

	bool isOpaqueEnough;
	bool isMovedEnough;
	bool isBigEnough;
	bool isNotTooBig;
	bool isBigEnoughToSplit;

	if (tid < params_OptiX.numberOfGaussians) {
		float lr;

		dL_dparam_1 = ((REAL4_G *)params_OptiX.dL_dparams_1)[tid];
		m1 = ((float4 *)params_OptiX.m11)[tid];
		v1 = ((float4 *)params_OptiX.v11)[tid];
		GC_1 = ((float4 *)params_OptiX.GC_part_1_1)[tid];

		m1.x = (beta1 * m1.x) + ((1.0f - beta1) * dL_dparam_1.x);
		m1.y = (beta1 * m1.y) + ((1.0f - beta1) * dL_dparam_1.y);
		m1.z = (beta1 * m1.z) + ((1.0f - beta1) * dL_dparam_1.z);
		m1.w = (beta1 * m1.w) + ((1.0f - beta1) * dL_dparam_1.w);

		v1.x = (beta2 * v1.x) + ((1.0f - beta2) * dL_dparam_1.x * dL_dparam_1.x);
		v1.y = (beta2 * v1.y) + ((1.0f - beta2) * dL_dparam_1.y * dL_dparam_1.y);
		v1.z = (beta2 * v1.z) + ((1.0f - beta2) * dL_dparam_1.z * dL_dparam_1.z);
		v1.w = (beta2 * v1.w) + ((1.0f - beta2) * dL_dparam_1.w * dL_dparam_1.w);

		lr = (
			(lr_SH0 * expf(params_OptiX.epoch * lr_SH0_exponential_decay_coefficient) < lr_SH0_final) ?
			lr_SH0_final :
			lr_SH0 * expf(params_OptiX.epoch * lr_SH0_exponential_decay_coefficient)
		);
		
		// R
		GC_1.x -= ((lr * (m1.x * tmp1)) / (sqrtf(v1.x * tmp2) + epsilon));
		// G
		GC_1.y -= ((lr * (m1.y * tmp1)) / (sqrtf(v1.y * tmp2) + epsilon));
		// B
		GC_1.z -= ((lr * (m1.z * tmp1)) / (sqrtf(v1.z * tmp2) + epsilon));
		
		lr = (
			(lr_alpha * expf(params_OptiX.epoch * lr_alpha_exponential_decay_coefficient) < lr_alpha_final) ?
			lr_alpha_final :
			lr_alpha * expf(params_OptiX.epoch * lr_alpha_exponential_decay_coefficient)
		);

		// alpha
		GC_1.w -= ((lr * (m1.w * tmp1)) / (sqrtf(v1.w * tmp2) + epsilon));
		isOpaqueEnough = (GC_1.w >= alpha_threshold_for_Gauss_removal);
		if ((GC_1.w < alpha_threshold_for_Gauss_removal) && (params_OptiX.epoch > densification_end_epoch))
			GC_1.w = alpha_threshold_for_Gauss_removal;

		// *****************************************************************************************

		dL_dparam_2 = ((REAL4_G *)params_OptiX.dL_dparams_2)[tid];
		m2 = ((float4 *)params_OptiX.m21)[tid];
		v2 = ((float4 *)params_OptiX.v21)[tid];
		GC_2 = ((float4 *)params_OptiX.GC_part_2_1)[tid];

		m2.x = (beta1 * m2.x) + ((1.0f - beta1) * dL_dparam_2.x);
		m2.y = (beta1 * m2.y) + ((1.0f - beta1) * dL_dparam_2.y);
		m2.z = (beta1 * m2.z) + ((1.0f - beta1) * dL_dparam_2.z);
		m2.w = (beta1 * m2.w) + ((1.0f - beta1) * dL_dparam_2.w);

		v2.x = (beta2 * v2.x) + ((1.0f - beta2) * dL_dparam_2.x * dL_dparam_2.x);
		v2.y = (beta2 * v2.y) + ((1.0f - beta2) * dL_dparam_2.y * dL_dparam_2.y);
		v2.z = (beta2 * v2.z) + ((1.0f - beta2) * dL_dparam_2.z * dL_dparam_2.z);
		v2.w = (beta2 * v2.w) + ((1.0f - beta2) * dL_dparam_2.w * dL_dparam_2.w);

		lr = (
			(lr_m * expf(params_OptiX.epoch * lr_m_exponential_decay_coefficient) < lr_m_final) ?
			lr_m_final :
			lr_m * expf(params_OptiX.epoch * lr_m_exponential_decay_coefficient)
		);
		lr = scene_extent * lr;

		// mX, mY, mZ
		dm = make_float3(
			((lr * (m2.x * tmp1)) / (sqrtf(v2.x * tmp2) + epsilon)),
			((lr * (m2.y * tmp1)) / (sqrtf(v2.y * tmp2) + epsilon)),
			((lr * (m2.z * tmp1)) / (sqrtf(v2.z * tmp2) + epsilon))
		);

		lr = (
			(lr_s * expf(params_OptiX.epoch * lr_s_exponential_decay_coefficient) < lr_s_final) ?
			lr_s_final:
			lr_s * expf(params_OptiX.epoch * lr_s_exponential_decay_coefficient)
		);
		lr = scene_extent * lr;

		// sX
		//GC_2.w -= ((lr * (m2.w * tmp1)) / (sqrtf(v2.w * tmp2) + epsilon));

		isMovedEnough = (sqrtf((dm.x * dm.x) + (dm.y * dm.y) + (dm.z * dm.z)) >= scene_extent * mu_grad_norm_threshold_for_densification);

		// *****************************************************************************************

		dL_dparam_3 = ((REAL4_G *)params_OptiX.dL_dparams_3)[tid];
		m3 = ((float4 *)params_OptiX.m31)[tid];
		v3 = ((float4 *)params_OptiX.v31)[tid];
		GC_3 = ((float4 *)params_OptiX.GC_part_3_1)[tid];

		m3.x = (beta1 * m3.x) + ((1.0f - beta1) * dL_dparam_3.x);
		m3.y = (beta1 * m3.y) + ((1.0f - beta1) * dL_dparam_3.y);
		m3.z = (beta1 * m3.z) + ((1.0f - beta1) * dL_dparam_3.z);
		m3.w = (beta1 * m3.w) + ((1.0f - beta1) * dL_dparam_3.w);

		v3.x = (beta2 * v3.x) + ((1.0f - beta2) * dL_dparam_3.x * dL_dparam_3.x);
		v3.y = (beta2 * v3.y) + ((1.0f - beta2) * dL_dparam_3.y * dL_dparam_3.y);
		v3.z = (beta2 * v3.z) + ((1.0f - beta2) * dL_dparam_3.z * dL_dparam_3.z);
		v3.w = (beta2 * v3.w) + ((1.0f - beta2) * dL_dparam_3.w * dL_dparam_3.w);

		// sY
		GC_3.x -= ((lr * (m3.x * tmp1)) / (sqrtf(v3.x * tmp2) + epsilon));
		// sZ
		GC_3.y -= ((lr * (m3.y * tmp1)) / (sqrtf(v3.y * tmp2) + epsilon));

		lr = (
			(lr_q * expf(params_OptiX.epoch * lr_q_exponential_decay_coefficient) < lr_q_final) ?
			lr_q_final :
			lr_q * expf(params_OptiX.epoch * lr_q_exponential_decay_coefficient)
		);

		// qr
		GC_3.z -= ((lr * (m3.z * tmp1)) / (sqrtf(v3.z * tmp2) + epsilon));
		// qi
		GC_3.w -= ((lr * (m3.w * tmp1)) / (sqrtf(v3.w * tmp2) + epsilon));

		scale = make_float3(
			expf(GC_2.w),
			expf(GC_3.x),
			expf(GC_3.y)
		);

		float length = sqrtf((scale.x * scale.x) + (scale.y * scale.y) + (scale.z * scale.z));

		isBigEnough = (length >= scene_extent * min_s_norm_threshold_for_Gauss_removal);
		isNotTooBig = (length <= scene_extent * max_s_norm_threshold_for_Gauss_removal);
		isBigEnoughToSplit = (length > scene_extent * s_norm_threshold_for_split_strategy);

		// *****************************************************************************************

		dL_dparam_4 = ((REAL2_G *)params_OptiX.dL_dparams_4)[tid];
		m4 = ((float2 *)params_OptiX.m41)[tid];
		v4 = ((float2 *)params_OptiX.v41)[tid];
		GC_4 = ((float2 *)params_OptiX.GC_part_4_1)[tid];

		m4.x = (beta1 * m4.x) + ((1.0f - beta1) * dL_dparam_4.x);
		m4.y = (beta1 * m4.y) + ((1.0f - beta1) * dL_dparam_4.y);

		v4.x = (beta2 * v4.x) + ((1.0f - beta2) * dL_dparam_4.x * dL_dparam_4.x);
		v4.y = (beta2 * v4.y) + ((1.0f - beta2) * dL_dparam_4.y * dL_dparam_4.y);

		// qj
		GC_4.x -= ((lr * (m4.x * tmp1)) / (sqrtf(v4.x * tmp2) + epsilon));
		// qk
		GC_4.y -= ((lr * (m4.y * tmp1)) / (sqrtf(v4.y * tmp2) + epsilon));

		// *****************************************************************************************

		// Spherical harmonics
		if constexpr (SH_degree >= 1) {
			float lr1;
			
			lr1 = (
				(lr_SH1 * expf(params_OptiX.epoch * lr_SH1_exponential_decay_coefficient) < lr_SH1_final) ?
				lr_SH1_final :
				lr_SH1 * expf(params_OptiX.epoch * lr_SH1_exponential_decay_coefficient)
			);

			backward_SH_float4(1, lr1, lr1, lr1, lr1);
			backward_SH_float4(2, lr1, lr1, lr1, lr1);

			if constexpr (SH_degree >= 2) {
				float lr2;

				lr2 = (
					(lr_SH2 * expf(params_OptiX.epoch * lr_SH2_exponential_decay_coefficient) < lr_SH2_final) ?
					lr_SH2_final :
					lr_SH2 * expf(params_OptiX.epoch * lr_SH2_exponential_decay_coefficient)
				);

				backward_SH_float4(3, lr1, lr2, lr2, lr2);
				backward_SH_float4(4, lr2, lr2, lr2, lr2);
				backward_SH_float4(5, lr2, lr2, lr2, lr2);
				backward_SH_float4(6, lr2, lr2, lr2, lr2);

				if constexpr (SH_degree >= 3) {
					lr1 = (
						(lr_SH3 * expf(params_OptiX.epoch * lr_SH3_exponential_decay_coefficient) < lr_SH3_final) ?
						lr_SH3_final :
						lr_SH3 * expf(params_OptiX.epoch * lr_SH3_exponential_decay_coefficient)
					);

					backward_SH_float4(7, lr1, lr1, lr1, lr1);
					backward_SH_float4(8, lr1, lr1, lr1, lr1);
					backward_SH_float4(9, lr1, lr1, lr1, lr1);
					backward_SH_float4(10, lr1, lr1, lr1, lr1);
					backward_SH_float4(11, lr1, lr1, lr1, lr1);

					if constexpr (SH_degree >= 4) {
						lr2 = (
							(lr_SH4 * expf(params_OptiX.epoch * lr_SH4_exponential_decay_coefficient) < lr_SH4_final) ?
							lr_SH4_final :
							lr_SH4 * expf(params_OptiX.epoch * lr_SH4_exponential_decay_coefficient)
						);

						backward_SH_float4(12, lr1, lr2, lr2, lr2);
						backward_SH_float4(13, lr2, lr2, lr2, lr2);
						backward_SH_float4(14, lr2, lr2, lr2, lr2);
						backward_SH_float4(15, lr2, lr2, lr2, lr2);
						backward_SH_float4(16, lr2, lr2, lr2, lr2);
						backward_SH_float4(17, lr2, lr2, lr2, lr2);
						backward_SH_float4(18, lr2, lr2, lr2, lr2);
					} else {
						// !!! !!! !!!
						backward_SH_float(12, lr1);
					}
				}
			} else {
				// !!! !!! !!!
				backward_SH_float(3, lr1);
			}
		}
	}

	// *********************************************************************************************

	bool densification_epoch = (
		(params_OptiX.epoch >= densification_start_epoch) &&
		(params_OptiX.epoch <= densification_end_epoch) &&
		((params_OptiX.epoch % densification_frequency) == 0)
	);
	bool clone_or_split;

	unsigned GaussInd;
	unsigned sampleNum;

	if (densification_epoch) {
		if (threadIdx.x == 0) {
			counter1 = 0;
			counter2 = 0;
		}
		__syncthreads();

		clone_or_split = (isMovedEnough && (params_OptiX.numberOfGaussians <= max_Gaussians_per_model));
		if (tid < params_OptiX.numberOfGaussians) {
			if (isOpaqueEnough && isBigEnough && isNotTooBig) {
				if (clone_or_split) {
					GaussInd = atomicAdd(&counter1, 1);
					if (isBigEnoughToSplit) sampleNum = atomicAdd(&counter2, 6);
				}
			} else
				needsToBeRemoved[tid] = 0;
		}
		__syncthreads();

		if (threadIdx.x == 0) {
			counter1 = atomicAdd((unsigned *)params_OptiX.counter1, counter1);
			counter2 = atomicAdd((unsigned *)params_OptiX.counter2, counter2);
		}
		__syncthreads();

		GaussInd += (params_OptiX.numberOfGaussians + counter1);
		sampleNum += counter2;
	}

	// *********************************************************************************************

	if (tid < params_OptiX.numberOfGaussians) {	
		if (!densification_epoch) {
			//scale.x = ((scale.x < scene_extent * min_s_coefficients_clipping_threshold) ? scene_extent * min_s_coefficients_clipping_threshold : scale.x);
			scale.y = ((scale.y < scene_extent * min_s_coefficients_clipping_threshold) ? scene_extent * min_s_coefficients_clipping_threshold : scale.y);
			scale.z = ((scale.z < scene_extent * min_s_coefficients_clipping_threshold) ? scene_extent * min_s_coefficients_clipping_threshold : scale.z);

			//scale.x = ((scale.x > scene_extent * max_s_norm_threshold_for_Gauss_removal) ? scene_extent * max_s_norm_threshold_for_Gauss_removal : scale.x);
			scale.y = ((scale.y > scene_extent * max_s_norm_threshold_for_Gauss_removal) ? scene_extent * max_s_norm_threshold_for_Gauss_removal : scale.y);
			scale.z = ((scale.z > scene_extent * max_s_norm_threshold_for_Gauss_removal) ? scene_extent * max_s_norm_threshold_for_Gauss_removal : scale.z);

			// *** *** *** *** ***

			//GC_2.w = logf(scale.x);
			GC_3.x = logf(scale.y);
			GC_3.y = logf(scale.z);

			// *** *** *** *** ***

			((float4 *)params_OptiX.GC_part_1_1)[tid] = GC_1;
			((float4 *)params_OptiX.m11)[tid] = m1;
			((float4 *)params_OptiX.v11)[tid] = v1;

			// *** *** *** *** ***

			// Spherical harmonics
			if constexpr (SH_degree >= 1) {
				backward_SH_store(1, tid);
				backward_SH_store(2, tid);
				backward_SH_store(3, tid);

				if constexpr (SH_degree >= 2) {
					backward_SH_store(4, tid);
					backward_SH_store(5, tid);
					backward_SH_store(6, tid);

					if constexpr (SH_degree >= 3) {
						backward_SH_store(7, tid);
						backward_SH_store(8, tid);
						backward_SH_store(9, tid);
						backward_SH_store(10, tid);
						backward_SH_store(11, tid);
						backward_SH_store(12, tid);
						
						if constexpr (SH_degree >= 4) {
							backward_SH_store(13, tid);
							backward_SH_store(14, tid);
							backward_SH_store(15, tid);
							backward_SH_store(16, tid);
							backward_SH_store(17, tid);
							backward_SH_store(18, tid);
						}
					}
				}
			}

			// *** *** *** *** ***

			// mX
			GC_2.x -= dm.x;
			// mY
			GC_2.y -= dm.y;
			// mZ
			GC_2.z -= dm.z;

			// *** *** *** *** ***

			((float4 *)params_OptiX.GC_part_2_1)[tid] = GC_2;
			((float4 *)params_OptiX.m21)[tid] = m2;
			((float4 *)params_OptiX.v21)[tid] = v2;

			((float4 *)params_OptiX.GC_part_3_1)[tid] = GC_3;
			((float4 *)params_OptiX.m31)[tid] = m3;
			((float4 *)params_OptiX.v31)[tid] = v3;

			((float2 *)params_OptiX.GC_part_4_1)[tid] = GC_4;
			((float2 *)params_OptiX.m41)[tid] = m4;
			((float2 *)params_OptiX.v41)[tid] = v4;
		} else {
			if (isOpaqueEnough && isBigEnough && isNotTooBig) {
				if (clone_or_split) {
					if (isBigEnoughToSplit) {
						float Z1, Z2;
						RandomNormalFloat(sampleNum, Z1, Z2); // !!! !!! !!!
						float Z3, Z4;
						RandomNormalFloat(sampleNum + 2, Z3, Z4); // !!! !!! !!!
						float Z5, Z6;
						RandomNormalFloat(sampleNum + 4, Z5, Z6); // !!! !!! !!!

						// *** *** *** *** ***

						float4 q = make_float4(GC_3.z, GC_3.w, GC_4.x, GC_4.y);

						float R11, R12, R13;
						float R21, R22, R23;
						float R31, R32, R33;

						ComputeRotationMatrix(
							q,

							R11, R12, R13,
							R21, R22, R23,
							R31, R32, R33
						);

						// *** *** *** *** ***						

						float3 mu = make_float3(GC_2.x, GC_2.y, GC_2.z);

						float3 P1 = RandomMultinormalFloat(
							mu,
							scale,

							R11, R12, R13,
							R21, R22, R23,
							R31, R32, R33, 
							
							Z1, Z2, Z3
						);

						float3 P2 = RandomMultinormalFloat(
							mu,
							scale,

							R11, R12, R13,
							R21, R22, R23,
							R31, R32, R33,

							Z4, Z5, Z6
						);

						// *** *** *** *** ***

						//scale.x /= split_ratio;
						scale.y /= split_ratio;
						scale.z /= split_ratio;

						// *** *** *** *** ***

						//scale.x = ((scale.x < scene_extent * min_s_coefficients_clipping_threshold) ? scene_extent * min_s_coefficients_clipping_threshold : scale.x);
						scale.y = ((scale.y < scene_extent * min_s_coefficients_clipping_threshold) ? scene_extent * min_s_coefficients_clipping_threshold : scale.y);
						scale.z = ((scale.z < scene_extent * min_s_coefficients_clipping_threshold) ? scene_extent * min_s_coefficients_clipping_threshold : scale.z);

						//scale.x = ((scale.x > scene_extent * max_s_norm_threshold_for_Gauss_removal) ? scene_extent * max_s_norm_threshold_for_Gauss_removal : scale.x);
						scale.y = ((scale.y > scene_extent * max_s_norm_threshold_for_Gauss_removal) ? scene_extent * max_s_norm_threshold_for_Gauss_removal : scale.y);
						scale.z = ((scale.z > scene_extent * max_s_norm_threshold_for_Gauss_removal) ? scene_extent * max_s_norm_threshold_for_Gauss_removal : scale.z);

						// *** *** *** *** ***

						//GC_2.w = logf(scale.x);
						GC_3.x = logf(scale.y);
						GC_3.y = logf(scale.z);

						// *** *** *** *** ***

						((float4 *)params_OptiX.GC_part_1_1)[tid] = GC_1;
						((float4 *)params_OptiX.m11)[tid] = m1;
						((float4 *)params_OptiX.v11)[tid] = v1;

						// *** *** *** *** ***

						// Spherical harmonics
						if constexpr (SH_degree >= 1) {
							backward_SH_store(1, tid);
							backward_SH_store(2, tid);
							backward_SH_store(3, tid);

							if constexpr (SH_degree >= 2) {
								backward_SH_store(4, tid);
								backward_SH_store(5, tid);
								backward_SH_store(6, tid);

								if constexpr (SH_degree >= 3) {
									backward_SH_store(7, tid);
									backward_SH_store(8, tid);
									backward_SH_store(9, tid);
									backward_SH_store(10, tid);
									backward_SH_store(11, tid);
									backward_SH_store(12, tid);

									if constexpr (SH_degree >= 4) {
										backward_SH_store(13, tid);
										backward_SH_store(14, tid);
										backward_SH_store(15, tid);
										backward_SH_store(16, tid);
										backward_SH_store(17, tid);
										backward_SH_store(18, tid);
									}
								}
							}
						}

						// *** *** *** *** ***

						GC_2.x = P1.x;
						GC_2.y = P1.y;
						GC_2.z = P1.z;

						// *** *** *** *** ***

						((float4 *)params_OptiX.GC_part_2_1)[tid] = GC_2;
						((float4 *)params_OptiX.m21)[tid] = m2;
						((float4 *)params_OptiX.v21)[tid] = v2;

						((float4 *)params_OptiX.GC_part_3_1)[tid] = GC_3;
						((float4 *)params_OptiX.m31)[tid] = m3;
						((float4 *)params_OptiX.v31)[tid] = v3;

						((float2 *)params_OptiX.GC_part_4_1)[tid] = GC_4;
						((float2 *)params_OptiX.m41)[tid] = m4;
						((float2 *)params_OptiX.v41)[tid] = v4;

						// *** *** *** *** ***

						((float4 *)params_OptiX.GC_part_1_1)[GaussInd] = GC_1;
						((float4 *)params_OptiX.m11)[GaussInd] = m1;
						((float4 *)params_OptiX.v11)[GaussInd] = v1;

						// *** *** *** *** ***

						// Spherical harmonics
						if constexpr (SH_degree >= 1) {
							backward_SH_store(1, GaussInd);
							backward_SH_store(2, GaussInd);
							backward_SH_store(3, GaussInd);

							if constexpr (SH_degree >= 2) {
								backward_SH_store(4, GaussInd);
								backward_SH_store(5, GaussInd);
								backward_SH_store(6, GaussInd);

								if constexpr (SH_degree >= 3) {
									backward_SH_store(7, GaussInd);
									backward_SH_store(8, GaussInd);
									backward_SH_store(9, GaussInd);
									backward_SH_store(10, GaussInd);
									backward_SH_store(11, GaussInd);
									backward_SH_store(12, GaussInd);

									if constexpr (SH_degree >= 4) {
										backward_SH_store(13, GaussInd);
										backward_SH_store(14, GaussInd);
										backward_SH_store(15, GaussInd);
										backward_SH_store(16, GaussInd);
										backward_SH_store(17, GaussInd);
										backward_SH_store(18, GaussInd);
									}
								}
							}
						}

						// *** *** *** *** ***

						GC_2.x = P2.x;
						GC_2.y = P2.y;
						GC_2.z = P2.z;

						// *** *** *** *** ***

						((float4 *)params_OptiX.GC_part_2_1)[GaussInd] = GC_2;
						((float4 *)params_OptiX.m21)[GaussInd] = m2;
						((float4 *)params_OptiX.v21)[GaussInd] = v2;

						((float4 *)params_OptiX.GC_part_3_1)[GaussInd] = GC_3;
						((float4 *)params_OptiX.m31)[GaussInd] = m3;
						((float4 *)params_OptiX.v31)[GaussInd] = v3;

						((float2 *)params_OptiX.GC_part_4_1)[GaussInd] = GC_4;
						((float2 *)params_OptiX.m41)[GaussInd] = m4;
						((float2 *)params_OptiX.v41)[GaussInd] = v4;
					} else {
						//scale.x = ((scale.x < scene_extent * min_s_coefficients_clipping_threshold) ? scene_extent * min_s_coefficients_clipping_threshold : scale.x);
						scale.y = ((scale.y < scene_extent * min_s_coefficients_clipping_threshold) ? scene_extent * min_s_coefficients_clipping_threshold : scale.y);
						scale.z = ((scale.z < scene_extent * min_s_coefficients_clipping_threshold) ? scene_extent * min_s_coefficients_clipping_threshold : scale.z);

						//scale.x = ((scale.x > scene_extent * max_s_norm_threshold_for_Gauss_removal) ? scene_extent * max_s_norm_threshold_for_Gauss_removal : scale.x);
						scale.y = ((scale.y > scene_extent * max_s_norm_threshold_for_Gauss_removal) ? scene_extent * max_s_norm_threshold_for_Gauss_removal : scale.y);
						scale.z = ((scale.z > scene_extent * max_s_norm_threshold_for_Gauss_removal) ? scene_extent * max_s_norm_threshold_for_Gauss_removal : scale.z);

						// *** *** *** *** ***

						//GC_2.w = logf(scale.x);
						GC_3.x = logf(scale.y);
						GC_3.y = logf(scale.z);

						// *** *** *** *** ***

						((float4 *)params_OptiX.GC_part_1_1)[tid] = GC_1;
						((float4 *)params_OptiX.m11)[tid] = m1;
						((float4 *)params_OptiX.v11)[tid] = v1;

						// *** *** *** *** ***

						// Spherical harmonics
						if constexpr (SH_degree >= 1) {
							backward_SH_store(1, tid);
							backward_SH_store(2, tid);
							backward_SH_store(3, tid);

							if constexpr (SH_degree >= 2) {
								backward_SH_store(4, tid);
								backward_SH_store(5, tid);
								backward_SH_store(6, tid);

								if constexpr (SH_degree >= 3) {
									backward_SH_store(7, tid);
									backward_SH_store(8, tid);
									backward_SH_store(9, tid);
									backward_SH_store(10, tid);
									backward_SH_store(11, tid);
									backward_SH_store(12, tid);

									if constexpr (SH_degree >= 4) {
										backward_SH_store(13, tid);
										backward_SH_store(14, tid);
										backward_SH_store(15, tid);
										backward_SH_store(16, tid);
										backward_SH_store(17, tid);
										backward_SH_store(18, tid);
									}
								}
							}
						}

						// *** *** *** *** ***

						((float4 *)params_OptiX.GC_part_2_1)[tid] = GC_2;
						((float4 *)params_OptiX.m21)[tid] = m2;
						((float4 *)params_OptiX.v21)[tid] = v2;

						((float4 *)params_OptiX.GC_part_3_1)[tid] = GC_3;
						((float4 *)params_OptiX.m31)[tid] = m3;
						((float4 *)params_OptiX.v31)[tid] = v3;

						((float2 *)params_OptiX.GC_part_4_1)[tid] = GC_4;
						((float2 *)params_OptiX.m41)[tid] = m4;
						((float2 *)params_OptiX.v41)[tid] = v4;

						// *** *** *** *** ***

						((float4 *)params_OptiX.GC_part_1_1)[GaussInd] = GC_1;
						((float4 *)params_OptiX.m11)[GaussInd] = m1;
						((float4 *)params_OptiX.v11)[GaussInd] = v1;

						// *** *** *** *** ***

						// Spherical harmonics
						if constexpr (SH_degree >= 1) {
							backward_SH_store(1, GaussInd);
							backward_SH_store(2, GaussInd);
							backward_SH_store(3, GaussInd);

							if constexpr (SH_degree >= 2) {
								backward_SH_store(4, GaussInd);
								backward_SH_store(5, GaussInd);
								backward_SH_store(6, GaussInd);

								if constexpr (SH_degree >= 3) {
									backward_SH_store(7, GaussInd);
									backward_SH_store(8, GaussInd);
									backward_SH_store(9, GaussInd);
									backward_SH_store(10, GaussInd);
									backward_SH_store(11, GaussInd);
									backward_SH_store(12, GaussInd);

									if constexpr (SH_degree >= 4) {
										backward_SH_store(13, GaussInd);
										backward_SH_store(14, GaussInd);
										backward_SH_store(15, GaussInd);
										backward_SH_store(16, GaussInd);
										backward_SH_store(17, GaussInd);
										backward_SH_store(18, GaussInd);
									}
								}
							}
						}

						// *** *** *** *** ***

						// mX
						GC_2.x -= dm.x;
						// mY
						GC_2.y -= dm.y;
						// mZ
						GC_2.z -= dm.z;

						// *** *** *** *** ***

						((float4 *)params_OptiX.GC_part_2_1)[GaussInd] = GC_2;
						((float4 *)params_OptiX.m21)[GaussInd] = m2;
						((float4 *)params_OptiX.v21)[GaussInd] = v2;

						((float4 *)params_OptiX.GC_part_3_1)[GaussInd] = GC_3;
						((float4 *)params_OptiX.m31)[GaussInd] = m3;
						((float4 *)params_OptiX.v31)[GaussInd] = v3;

						((float2 *)params_OptiX.GC_part_4_1)[GaussInd] = GC_4;
						((float2 *)params_OptiX.m41)[GaussInd] = m4;
						((float2 *)params_OptiX.v41)[GaussInd] = v4;
					}
				} else {
					//scale.x = ((scale.x < scene_extent * min_s_coefficients_clipping_threshold) ? scene_extent * min_s_coefficients_clipping_threshold : scale.x);
					scale.y = ((scale.y < scene_extent * min_s_coefficients_clipping_threshold) ? scene_extent * min_s_coefficients_clipping_threshold : scale.y);
					scale.z = ((scale.z < scene_extent * min_s_coefficients_clipping_threshold) ? scene_extent * min_s_coefficients_clipping_threshold : scale.z);

					//scale.x = ((scale.x > scene_extent * max_s_norm_threshold_for_Gauss_removal) ? scene_extent * max_s_norm_threshold_for_Gauss_removal : scale.x);
					scale.y = ((scale.y > scene_extent * max_s_norm_threshold_for_Gauss_removal) ? scene_extent * max_s_norm_threshold_for_Gauss_removal : scale.y);
					scale.z = ((scale.z > scene_extent * max_s_norm_threshold_for_Gauss_removal) ? scene_extent * max_s_norm_threshold_for_Gauss_removal : scale.z);

					// *** *** *** *** ***

					//GC_2.w = logf(scale.x);
					GC_3.x = logf(scale.y);
					GC_3.y = logf(scale.z);

					// *** *** *** *** ***

					((float4 *)params_OptiX.GC_part_1_1)[tid] = GC_1;
					((float4 *)params_OptiX.m11)[tid] = m1;
					((float4 *)params_OptiX.v11)[tid] = v1;

					// *** *** *** *** ***

					// Spherical harmonics
					if constexpr (SH_degree >= 1) {
						backward_SH_store(1, tid);
						backward_SH_store(2, tid);
						backward_SH_store(3, tid);

						if constexpr (SH_degree >= 2) {
							backward_SH_store(4, tid);
							backward_SH_store(5, tid);
							backward_SH_store(6, tid);

							if constexpr (SH_degree >= 3) {
								backward_SH_store(7, tid);
								backward_SH_store(8, tid);
								backward_SH_store(9, tid);
								backward_SH_store(10, tid);
								backward_SH_store(11, tid);
								backward_SH_store(12, tid);

								if constexpr (SH_degree >= 4) {
									backward_SH_store(13, tid);
									backward_SH_store(14, tid);
									backward_SH_store(15, tid);
									backward_SH_store(16, tid);
									backward_SH_store(17, tid);
									backward_SH_store(18, tid);
								}
							}
						}
					}

					// *** *** *** *** ***

					// mX
					GC_2.x -= dm.x;
					// mY
					GC_2.y -= dm.y;
					// mZ
					GC_2.z -= dm.z;

					// *** *** *** *** ***

					((float4 *)params_OptiX.GC_part_2_1)[tid] = GC_2;
					((float4 *)params_OptiX.m21)[tid] = m2;
					((float4 *)params_OptiX.v21)[tid] = v2;

					((float4 *)params_OptiX.GC_part_3_1)[tid] = GC_3;
					((float4 *)params_OptiX.m31)[tid] = m3;
					((float4 *)params_OptiX.v31)[tid] = v3;

					((float2 *)params_OptiX.GC_part_4_1)[tid] = GC_4;
					((float2 *)params_OptiX.m41)[tid] = m4;
					((float2 *)params_OptiX.v41)[tid] = v4;
				}
			}
		}
	}
}

// *************************************************************************************************

__global__ void ComputeArraysForGradientComputation(
	REAL_G *dev_mu1R, REAL_G *dev_mu2R,
	REAL_G *dev_sigma12R,
	REAL_G *dev_sigma1R_square, REAL_G *dev_sigma2R_square,

	REAL_G *dev_mu1G, REAL_G *dev_mu2G,
	REAL_G *dev_sigma12G,
	REAL_G *dev_sigma1G_square, REAL_G *dev_sigma2G_square,

	REAL_G *dev_mu1B, REAL_G *dev_mu2B,
	REAL_G *dev_sigma12B,
	REAL_G *dev_sigma1B_square, REAL_G *dev_sigma2B_square,

	REAL_G *dev_tmp1R, REAL_G *dev_tmp2R, REAL_G *dev_tmp3R,
	REAL_G *dev_tmp1G, REAL_G *dev_tmp2G, REAL_G *dev_tmp3G,
	REAL_G *dev_tmp1B, REAL_G *dev_tmp2B, REAL_G *dev_tmp3B,

	int width, int height, int kernel_radius
) {
	int tid = (blockIdx.x * blockDim.x) + threadIdx.x;
	int x = tid % width;
	int y = tid / width;
	int size = width * height;
	if (tid < size) {
		if ((x >= kernel_radius) && (x < width - kernel_radius) && (y >= kernel_radius) && (y < height - kernel_radius)) {
			REAL_G mu1R = dev_mu1R[tid] / size;
			REAL_G mu2R = dev_mu2R[tid] / size;
			REAL_G sigma12R =  dev_sigma12R[tid] / size;
			REAL_G sigma1R_square = dev_sigma1R_square[tid] / size;
			REAL_G sigma2R_square = dev_sigma2R_square[tid] / size;

			REAL_G mu1G = dev_mu1G[tid] / size;
			REAL_G mu2G = dev_mu2G[tid] / size;
			REAL_G sigma12G =  dev_sigma12G[tid] / size;
			REAL_G sigma1G_square = dev_sigma1G_square[tid] / size;
			REAL_G sigma2G_square = dev_sigma2G_square[tid] / size;

			REAL_G mu1B = dev_mu1B[tid] / size;
			REAL_G mu2B = dev_mu2B[tid] / size;
			REAL_G sigma12B =  dev_sigma12B[tid] / size;
			REAL_G sigma1B_square = dev_sigma1B_square[tid] / size;
			REAL_G sigma2B_square = dev_sigma2B_square[tid] / size;

			REAL_G c1 = ((REAL_G)(0.01 * 0.01));
			REAL_G c2 = ((REAL_G)(0.03 * 0.03));

			REAL_G AR = (2 * mu1R * mu2R) + c1;
			REAL_G BR = (2 * (sigma12R - (mu1R * mu2R))) + c2;
			REAL_G CR = ((mu1R * mu1R) + (mu2R * mu2R) + c1);
			REAL_G DR = ((sigma1R_square - (mu1R * mu1R)) + (sigma2R_square - (mu2R * mu2R)) + c2);

			REAL_G AG = (2 * mu1G * mu2G) + c1;
			REAL_G BG = (2 * (sigma12G - (mu1G * mu2G))) + c2;
			REAL_G CG = ((mu1G * mu1G) + (mu2G * mu2G) + c1);
			REAL_G DG = ((sigma1G_square - (mu1G * mu1G)) + (sigma2G_square - (mu2G * mu2G)) + c2);

			REAL_G AB = (2 * mu1B * mu2B) + c1;
			REAL_G BB = (2 * (sigma12B - (mu1B * mu2B))) + c2;
			REAL_G CB = ((mu1B * mu1B) + (mu2B * mu2B) + c1);
			REAL_G DB = ((sigma1B_square - (mu1B * mu1B)) + (sigma2B_square - (mu2B * mu2B)) + c2);

			REAL_G tmp1R = (2 * ((CR * DR * mu2R * (BR - AR)) - (AR * BR * mu1R * (DR - CR)))) / (CR * CR * DR * DR);
			REAL_G tmp2R = (2 * AR * CR * DR) / (CR * CR * DR * DR);
			REAL_G tmp3R = (2 * AR * BR * CR) / (CR * CR * DR * DR);

			REAL_G tmp1G = (2 * ((CG * DG * mu2G * (BG - AG)) - (AG * BG * mu1G * (DG - CG)))) / (CG * CG * DG * DG);
			REAL_G tmp2G = (2 * AG * CG * DG) / (CG * CG * DG * DG);
			REAL_G tmp3G = (2 * AG * BG * CG) / (CG * CG * DG * DG);

			REAL_G tmp1B = (2 * ((CB * DB * mu2B * (BB - AB)) - (AB * BB * mu1B * (DB - CB)))) / (CB * CB * DB * DB);
			REAL_G tmp2B = (2 * AB * CB * DB) / (CB * CB * DB * DB);
			REAL_G tmp3B = (2 * AB * BB * CB) / (CB * CB * DB * DB);

			dev_tmp1R[tid] = tmp1R;
			dev_tmp2R[tid] = tmp2R;
			dev_tmp3R[tid] = tmp3R;

			dev_tmp1G[tid] = tmp1G;
			dev_tmp2G[tid] = tmp2G;
			dev_tmp3G[tid] = tmp3G;

			dev_tmp1B[tid] = tmp1B;
			dev_tmp2B[tid] = tmp2B;
			dev_tmp3B[tid] = tmp3B;
		} else {
			dev_tmp1R[tid] = 0;
			dev_tmp2R[tid] = 0;
			dev_tmp3R[tid] = 0;

			dev_tmp1G[tid] = 0;
			dev_tmp2G[tid] = 0;
			dev_tmp3G[tid] = 0;

			dev_tmp1B[tid] = 0;
			dev_tmp2B[tid] = 0;
			dev_tmp3B[tid] = 0;
		}
	}
}

// *************************************************************************************************

__global__ void ComputeGradientSSIM(
	REAL_G *dev_conv1R,
	REAL_G *dev_conv2R, REAL_G *dev_img2R,
	REAL_G *dev_conv3R, REAL_G *dev_img1R,

	REAL_G *dev_conv1G,
	REAL_G *dev_conv2G, REAL_G *dev_img2G,
	REAL_G *dev_conv3G, REAL_G *dev_img1G,

	REAL_G *dev_conv1B,
	REAL_G *dev_conv2B, REAL_G *dev_img2B,
	REAL_G *dev_conv3B, REAL_G *dev_img1B,

	REAL_G *dev_gradR, REAL_G *dev_gradG, REAL_G *dev_gradB,

	int width, int height, int kernel_size
) {
	int tid = (blockIdx.x * blockDim.x) + threadIdx.x;
	int x = tid % width;
	int y = tid / width;
	int size = width * height;
	if ((tid < size) && (x >= kernel_size - 1) && (y >= kernel_size - 1)) {
		int ind = ((y - (kernel_size - 1)) * width) + (x - (kernel_size - 1));

		REAL_G conv1R = dev_conv1R[tid] / size;
		REAL_G conv2R = dev_conv2R[tid] / size;
		REAL_G img2R = dev_img2R[ind];
		REAL_G conv3R = dev_conv3R[tid] / size;
		REAL_G img1R = dev_img1R[ind];

		REAL_G conv1G = dev_conv1G[tid] / size;
		REAL_G conv2G = dev_conv2G[tid] / size;
		REAL_G img2G = dev_img2G[ind];
		REAL_G conv3G = dev_conv3G[tid] / size;
		REAL_G img1G = dev_img1G[ind];

		REAL_G conv1B = dev_conv1B[tid] / size;
		REAL_G conv2B = dev_conv2B[tid] / size;
		REAL_G img2B = dev_img2B[ind];
		REAL_G conv3B = dev_conv3B[tid] / size;
		REAL_G img1B = dev_img1B[ind];

		REAL_G gradR = ((conv3R * img1R) - conv1R - (conv2R * img2R)) / (2 * 3 * (width - (kernel_size - 1)) * (height - (kernel_size - 1)));
		REAL_G gradG = ((conv3G * img1G) - conv1G - (conv2G * img2G)) / (2 * 3 * (width - (kernel_size - 1)) * (height - (kernel_size - 1)));
		REAL_G gradB = ((conv3B * img1B) - conv1B - (conv2B * img2B)) / (2 * 3 * (width - (kernel_size - 1)) * (height - (kernel_size - 1)));

		dev_gradR[tid] = gradR;
		dev_gradG[tid] = gradG;
		dev_gradB[tid] = gradB;
	}
}

// *************************************************************************************************

#ifdef SSIM_REDUCE_MEMORY_OVERHEAD
__global__ void UnpackImage(
	unsigned *bitmap_ref,
	REAL_G *bitmap_ref_R, REAL_G *bitmap_ref_G, REAL_G *bitmap_ref_B,
	int width, int height, int kernel_size
) {
	int tid = (blockIdx.x * blockDim.x) + threadIdx.x;
	int x = tid % width;
	int y = tid / width;
	int arraySizeReal = (width + (kernel_size - 1)) * (height + (kernel_size - 1));
	if (tid < arraySizeReal) {
		if ((x >= 0) && (x < width) && (y >= 0) && (y < height)) {
			unsigned color_ref = bitmap_ref[(y * width) + x];
			unsigned char R = color_ref >> 16;
			unsigned char G = (color_ref >> 8) & 255;
			unsigned char B = color_ref & 255;
			bitmap_ref_R[(y * (width + (kernel_size - 1))) + x] = (R + ((REAL_G)0.5)) / ((REAL_G)256);
			bitmap_ref_G[(y * (width + (kernel_size - 1))) + x] = (G + ((REAL_G)0.5)) / ((REAL_G)256);
			bitmap_ref_B[(y * (width + (kernel_size - 1))) + x] = (B + ((REAL_G)0.5)) / ((REAL_G)256);
		} else {
			bitmap_ref_R[tid] = 0;
			bitmap_ref_G[tid] = 0;
			bitmap_ref_B[tid] = 0;
		}
	}
}
#endif

// *************************************************************************************************

template<int SH_degree>
bool UpdateGradientOptiX(SOptiXRenderParams<SH_degree> &params_OptiX) {
	cudaError_t error_CUDA;
	OptixResult error_OptiX;
	
	try {

		//**********************************************************************************************
		// SSIM                                                                                        *
		//**********************************************************************************************

		const int kernel_size = 11;
		const int kernel_radius = kernel_size >> 1;
		const REAL_G sigma = ((REAL_G)1.5);

		int arraySizeReal = (params_OptiX.width + (kernel_size - 1)) * (params_OptiX.height + (kernel_size - 1)); // !!! !!! !!!
		int arraySizeComplex = (((params_OptiX.width + (kernel_size - 1)) >> 1) + 1) * (params_OptiX.height + (kernel_size - 1)); // !!! !!! !!!

		cufftResult error_CUFFT;

		// *********************************************************************************************

	#ifdef SSIM_REDUCE_MEMORY_OVERHEAD
		UnpackImage<<<(arraySizeReal + 255) >> 8, 256>>>(
			params_OptiX.bitmap_ref + (params_OptiX.poseNum * params_OptiX.width * params_OptiX.height),
			params_OptiX.bitmap_ref_R, params_OptiX.bitmap_ref_G, params_OptiX.bitmap_ref_B,
			params_OptiX.width, params_OptiX.height, kernel_size
			);
		error_CUDA = cudaGetLastError();
		if (error_CUDA != cudaSuccess) throw 0;

		// ************************************************************************************************

		// Compute mu's for reference images

		// R channel
		error_CUFFT = DFFT_G(params_OptiX.planr2c, params_OptiX.bitmap_ref_R + (0 * arraySizeReal), params_OptiX.F_2);
		if (error_CUFFT != CUFFT_SUCCESS) throw 0;

		MultiplyPointwiseComplex<<<(arraySizeComplex + 255) >> 8, 256>>>(params_OptiX.F_2, params_OptiX.F_1, params_OptiX.F_2, arraySizeComplex);
		error_CUDA = cudaGetLastError();
		if (error_CUDA != cudaSuccess) throw 0;

		error_CUFFT = IDFFT_G(params_OptiX.planc2r, params_OptiX.F_2, params_OptiX.mu_bitmap_ref_R + (0 * arraySizeReal));
		if (error_CUFFT != CUFFT_SUCCESS) throw 0;

		// *** *** *** *** ***

		// G channel
		error_CUFFT = DFFT_G(params_OptiX.planr2c, params_OptiX.bitmap_ref_G + (0 * arraySizeReal), params_OptiX.F_2);
		if (error_CUFFT != CUFFT_SUCCESS) throw 0;

		MultiplyPointwiseComplex<<<(arraySizeComplex + 255) >> 8, 256>>>(params_OptiX.F_2, params_OptiX.F_1, params_OptiX.F_2, arraySizeComplex);
		error_CUDA = cudaGetLastError();
		if (error_CUDA != cudaSuccess) throw 0;

		error_CUFFT = IDFFT_G(params_OptiX.planc2r, params_OptiX.F_2, params_OptiX.mu_bitmap_ref_G + (0 * arraySizeReal));
		if (error_CUFFT != CUFFT_SUCCESS) throw 0;

		// *** *** *** *** ***

		// B channel
		error_CUFFT = DFFT_G(params_OptiX.planr2c, params_OptiX.bitmap_ref_B + (0 * arraySizeReal), params_OptiX.F_2);
		if (error_CUFFT != CUFFT_SUCCESS) throw 0;

		MultiplyPointwiseComplex<<<(arraySizeComplex + 255) >> 8, 256>>>(params_OptiX.F_2, params_OptiX.F_1, params_OptiX.F_2, arraySizeComplex);
		error_CUDA = cudaGetLastError();
		if (error_CUDA != cudaSuccess) throw 0;

		error_CUFFT = IDFFT_G(params_OptiX.planc2r, params_OptiX.F_2, params_OptiX.mu_bitmap_ref_B + (0 * arraySizeReal));
		if (error_CUFFT != CUFFT_SUCCESS) throw 0;

		// ************************************************************************************************

		// Compute mu's for reference images square

		// R channel
		// mu_bitmap_out_bitmap_ref_R = bitmap_ref_R * bitmap_ref_R
		MultiplyPointwiseReal<<<(arraySizeReal + 255) >> 8, 256>>>(
			params_OptiX.bitmap_ref_R + (0 * arraySizeReal),
			params_OptiX.bitmap_ref_R + (0 * arraySizeReal),
			params_OptiX.mu_bitmap_out_bitmap_ref_R,
			arraySizeReal
			);
		error_CUDA = cudaGetLastError();
		if (error_CUDA != cudaSuccess) throw 0;

		error_CUFFT = DFFT_G(params_OptiX.planr2c, params_OptiX.mu_bitmap_out_bitmap_ref_R, params_OptiX.F_2);
		if (error_CUFFT != CUFFT_SUCCESS) throw 0;

		MultiplyPointwiseComplex<<<(arraySizeComplex + 255) >> 8, 256>>>(params_OptiX.F_2, params_OptiX.F_1, params_OptiX.F_2, arraySizeComplex);
		error_CUDA = cudaGetLastError();
		if (error_CUDA != cudaSuccess) throw 0;

		error_CUFFT = IDFFT_G(params_OptiX.planc2r, params_OptiX.F_2, params_OptiX.mu_bitmap_ref_R_square + (0 * arraySizeReal));
		if (error_CUFFT != CUFFT_SUCCESS) throw 0;

		// *** *** *** *** ***

		// G channel
		// mu_bitmap_out_bitmap_ref_R = bitmap_ref_G * bitmap_ref_G
		MultiplyPointwiseReal<<<(arraySizeReal + 255) >> 8, 256>>>(
			params_OptiX.bitmap_ref_G + (0 * arraySizeReal),
			params_OptiX.bitmap_ref_G + (0 * arraySizeReal),
			params_OptiX.mu_bitmap_out_bitmap_ref_R,
			arraySizeReal
			);
		error_CUDA = cudaGetLastError();
		if (error_CUDA != cudaSuccess) throw 0;

		error_CUFFT = DFFT_G(params_OptiX.planr2c, params_OptiX.mu_bitmap_out_bitmap_ref_R, params_OptiX.F_2);
		if (error_CUFFT != CUFFT_SUCCESS) throw 0;

		MultiplyPointwiseComplex<<<(arraySizeComplex + 255) >> 8, 256>>>(params_OptiX.F_2, params_OptiX.F_1, params_OptiX.F_2, arraySizeComplex);
		error_CUDA = cudaGetLastError();
		if (error_CUDA != cudaSuccess) throw 0;

		error_CUFFT = IDFFT_G(params_OptiX.planc2r, params_OptiX.F_2, params_OptiX.mu_bitmap_ref_G_square + (0 * arraySizeReal));
		if (error_CUFFT != CUFFT_SUCCESS) throw 0;

		// *** *** *** *** ***

		// B channel
		// mu_bitmap_out_bitmap_ref_R = bitmap_ref_B * bitmap_ref_B
		MultiplyPointwiseReal<<<(arraySizeReal + 255) >> 8, 256>>>(
			params_OptiX.bitmap_ref_B + (0 * arraySizeReal),
			params_OptiX.bitmap_ref_B + (0 * arraySizeReal),
			params_OptiX.mu_bitmap_out_bitmap_ref_R,
			arraySizeReal
			);
		error_CUDA = cudaGetLastError();
		if (error_CUDA != cudaSuccess) throw 0;

		error_CUFFT = DFFT_G(params_OptiX.planr2c, params_OptiX.mu_bitmap_out_bitmap_ref_R, params_OptiX.F_2);
		if (error_CUFFT != CUFFT_SUCCESS) throw 0;

		MultiplyPointwiseComplex<<<(arraySizeComplex + 255) >> 8, 256>>>(params_OptiX.F_2, params_OptiX.F_1, params_OptiX.F_2, arraySizeComplex);
		error_CUDA = cudaGetLastError();
		if (error_CUDA != cudaSuccess) throw 0;

		error_CUFFT = IDFFT_G(params_OptiX.planc2r, params_OptiX.F_2, params_OptiX.mu_bitmap_ref_B_square + (0 * arraySizeReal));
		if (error_CUFFT != CUFFT_SUCCESS) throw 0;

	#endif

		// *********************************************************************************************

		//********************************
		// Compute mu's for output image *
		//********************************

		// R channel
		error_CUFFT = DFFT_G(params_OptiX.planr2c, params_OptiX.bitmap_out_R, params_OptiX.F_2);
		if (error_CUFFT != CUFFT_SUCCESS) throw 0;

		MultiplyPointwiseComplex<<<(arraySizeComplex + 255) >> 8, 256>>>(params_OptiX.F_2, params_OptiX.F_1, params_OptiX.F_2, arraySizeComplex);
		error_CUDA = cudaGetLastError();
		if (error_CUDA != cudaSuccess) throw 0;

		error_CUFFT = IDFFT_G(params_OptiX.planc2r, params_OptiX.F_2, params_OptiX.mu_bitmap_out_R);
		if (error_CUFFT != CUFFT_SUCCESS) throw 0;

		// *** *** *** *** ***

		// G channel
		error_CUFFT = DFFT_G(params_OptiX.planr2c, params_OptiX.bitmap_out_G, params_OptiX.F_2);
		if (error_CUFFT != CUFFT_SUCCESS) throw 0;

		MultiplyPointwiseComplex<<<(arraySizeComplex + 255) >> 8, 256>>>(params_OptiX.F_2, params_OptiX.F_1, params_OptiX.F_2, arraySizeComplex);
		error_CUDA = cudaGetLastError();
		if (error_CUDA != cudaSuccess) throw 0;

		error_CUFFT = IDFFT_G(params_OptiX.planc2r, params_OptiX.F_2, params_OptiX.mu_bitmap_out_G);
		if (error_CUFFT != CUFFT_SUCCESS) throw 0;

		// *** *** *** *** ***

		// B channel
		error_CUFFT = DFFT_G(params_OptiX.planr2c, params_OptiX.bitmap_out_B, params_OptiX.F_2);
		if (error_CUFFT != CUFFT_SUCCESS) throw 0;

		MultiplyPointwiseComplex<<<(arraySizeComplex + 255) >> 8, 256>>>(params_OptiX.F_2, params_OptiX.F_1, params_OptiX.F_2, arraySizeComplex);
		error_CUDA = cudaGetLastError();
		if (error_CUDA != cudaSuccess) throw 0;

		error_CUFFT = IDFFT_G(params_OptiX.planc2r, params_OptiX.F_2, params_OptiX.mu_bitmap_out_B);
		if (error_CUFFT != CUFFT_SUCCESS) throw 0;

		// *********************************************************************************************

		//***************************************
		// Compute mu's for output image square *
		//***************************************

		// R channel
		// mu_bitmap_out_bitmap_ref_R = bitmap_out_R * bitmap_out_R
		MultiplyPointwiseReal<<<(arraySizeReal + 255) >> 8, 256>>>(
			params_OptiX.bitmap_out_R,
			params_OptiX.bitmap_out_R,
			params_OptiX.mu_bitmap_out_bitmap_ref_R,
			arraySizeReal
			);
		error_CUDA = cudaGetLastError();
		if (error_CUDA != cudaSuccess) throw 0;

		error_CUFFT = DFFT_G(params_OptiX.planr2c, params_OptiX.mu_bitmap_out_bitmap_ref_R, params_OptiX.F_2);
		if (error_CUFFT != CUFFT_SUCCESS) throw 0;

		MultiplyPointwiseComplex<<<(arraySizeComplex + 255) >> 8, 256>>>(params_OptiX.F_2, params_OptiX.F_1, params_OptiX.F_2, arraySizeComplex);
		error_CUDA = cudaGetLastError();
		if (error_CUDA != cudaSuccess) throw 0;

		error_CUFFT = IDFFT_G(params_OptiX.planc2r, params_OptiX.F_2, params_OptiX.mu_bitmap_out_R_square);
		if (error_CUFFT != CUFFT_SUCCESS) throw 0;

		// *** *** *** *** ***

		// G channel
		// mu_bitmap_out_bitmap_ref_R = bitmap_out_G * bitmap_out_G
		MultiplyPointwiseReal<<<(arraySizeReal + 255) >> 8, 256>>>(
			params_OptiX.bitmap_out_G,
			params_OptiX.bitmap_out_G,
			params_OptiX.mu_bitmap_out_bitmap_ref_R,
			arraySizeReal
			);
		error_CUDA = cudaGetLastError();
		if (error_CUDA != cudaSuccess) throw 0;

		error_CUFFT = DFFT_G(params_OptiX.planr2c, params_OptiX.mu_bitmap_out_bitmap_ref_R, params_OptiX.F_2);
		if (error_CUFFT != CUFFT_SUCCESS) throw 0;

		MultiplyPointwiseComplex<<<(arraySizeComplex + 255) >> 8, 256>>>(params_OptiX.F_2, params_OptiX.F_1, params_OptiX.F_2, arraySizeComplex);
		error_CUDA = cudaGetLastError();
		if (error_CUDA != cudaSuccess) throw 0;

		error_CUFFT = IDFFT_G(params_OptiX.planc2r, params_OptiX.F_2, params_OptiX.mu_bitmap_out_G_square);
		if (error_CUFFT != CUFFT_SUCCESS) throw 0;

		// *** *** *** *** ***

		// B channel
		// mu_bitmap_out_bitmap_ref_R = bitmap_out_B * bitmap_out_B
		MultiplyPointwiseReal<<<(arraySizeReal + 255) >> 8, 256>>>(
			params_OptiX.bitmap_out_B,
			params_OptiX.bitmap_out_B,
			params_OptiX.mu_bitmap_out_bitmap_ref_R,
			arraySizeReal
			);
		error_CUDA = cudaGetLastError();
		if (error_CUDA != cudaSuccess) throw 0;

		error_CUFFT = DFFT_G(params_OptiX.planr2c, params_OptiX.mu_bitmap_out_bitmap_ref_R, params_OptiX.F_2);
		if (error_CUFFT != CUFFT_SUCCESS) throw 0;

		MultiplyPointwiseComplex<<<(arraySizeComplex + 255) >> 8, 256>>>(params_OptiX.F_2, params_OptiX.F_1, params_OptiX.F_2, arraySizeComplex);
		error_CUDA = cudaGetLastError();
		if (error_CUDA != cudaSuccess) throw 0;

		error_CUFFT = IDFFT_G(params_OptiX.planc2r, params_OptiX.F_2, params_OptiX.mu_bitmap_out_B_square);
		if (error_CUFFT != CUFFT_SUCCESS) throw 0;

		// *********************************************************************************************

		//***************************************************************
		// Compute mu's for product of output image and reference image *
		//***************************************************************

		// R channel
		// mu_bitmap_out_bitmap_ref_R = bitmap_out_R * bitmap_ref_R
	#ifndef SSIM_REDUCE_MEMORY_OVERHEAD
		MultiplyPointwiseReal<<<(arraySizeReal + 255) >> 8, 256>>>(
			params_OptiX.bitmap_out_R,
			params_OptiX.bitmap_ref_R + (params_OptiX.poseNum * arraySizeReal), // !!! !!! !!!
			params_OptiX.mu_bitmap_out_bitmap_ref_R,
			arraySizeReal
			);
	#else
		MultiplyPointwiseReal<<<(arraySizeReal + 255) >> 8, 256>>>(
			params_OptiX.bitmap_out_R,
			params_OptiX.bitmap_ref_R + (0 * arraySizeReal), // !!! !!! !!!
			params_OptiX.mu_bitmap_out_bitmap_ref_R,
			arraySizeReal
			);
	#endif
		error_CUDA = cudaGetLastError();
		if (error_CUDA != cudaSuccess) throw 0;

		error_CUFFT = DFFT_G(params_OptiX.planr2c, params_OptiX.mu_bitmap_out_bitmap_ref_R, params_OptiX.F_2);
		if (error_CUFFT != CUFFT_SUCCESS) throw 0;

		MultiplyPointwiseComplex<<<(arraySizeComplex + 255) >> 8, 256>>>(params_OptiX.F_2, params_OptiX.F_1, params_OptiX.F_2, arraySizeComplex);
		error_CUDA = cudaGetLastError();
		if (error_CUDA != cudaSuccess) throw 0;

		error_CUFFT = IDFFT_G(params_OptiX.planc2r, params_OptiX.F_2, params_OptiX.mu_bitmap_out_bitmap_ref_R);
		if (error_CUFFT != CUFFT_SUCCESS) throw 0;

		// *** *** *** *** ***

		// G channel
		// mu_bitmap_out_bitmap_ref_G = bitmap_out_G * bitmap_ref_G
	#ifndef SSIM_REDUCE_MEMORY_OVERHEAD
		MultiplyPointwiseReal<<<(arraySizeReal + 255) >> 8, 256>>>(
			params_OptiX.bitmap_out_G,
			params_OptiX.bitmap_ref_G + (params_OptiX.poseNum * arraySizeReal), // !!! !!! !!!
			params_OptiX.mu_bitmap_out_bitmap_ref_G,
			arraySizeReal
			);
	#else
		MultiplyPointwiseReal<<<(arraySizeReal + 255) >> 8, 256>>>(
			params_OptiX.bitmap_out_G,
			params_OptiX.bitmap_ref_G + (0 * arraySizeReal), // !!! !!! !!!
			params_OptiX.mu_bitmap_out_bitmap_ref_G,
			arraySizeReal
			);
	#endif
		error_CUDA = cudaGetLastError();
		if (error_CUDA != cudaSuccess) throw 0;

		error_CUFFT = DFFT_G(params_OptiX.planr2c, params_OptiX.mu_bitmap_out_bitmap_ref_G, params_OptiX.F_2);
		if (error_CUFFT != CUFFT_SUCCESS) throw 0;

		MultiplyPointwiseComplex<<<(arraySizeComplex + 255) >> 8, 256>>>(params_OptiX.F_2, params_OptiX.F_1, params_OptiX.F_2, arraySizeComplex);
		error_CUDA = cudaGetLastError();
		if (error_CUDA != cudaSuccess) throw 0;

		error_CUFFT = IDFFT_G(params_OptiX.planc2r, params_OptiX.F_2, params_OptiX.mu_bitmap_out_bitmap_ref_G);
		if (error_CUFFT != CUFFT_SUCCESS) throw 0;

		// *** *** *** *** ***

		// B channel
		// mu_bitmap_out_bitmap_ref_B = bitmap_out_B * bitmap_ref_B
	#ifndef SSIM_REDUCE_MEMORY_OVERHEAD
		MultiplyPointwiseReal<<<(arraySizeReal + 255) >> 8, 256>>>(
			params_OptiX.bitmap_out_B,
			params_OptiX.bitmap_ref_B + (params_OptiX.poseNum * arraySizeReal), // !!! !!! !!!
			params_OptiX.mu_bitmap_out_bitmap_ref_B,
			arraySizeReal
			);
	#else
		MultiplyPointwiseReal<<<(arraySizeReal + 255) >> 8, 256>>>(
			params_OptiX.bitmap_out_B,
			params_OptiX.bitmap_ref_B + (0 * arraySizeReal), // !!! !!! !!!
			params_OptiX.mu_bitmap_out_bitmap_ref_B,
			arraySizeReal
			);
	#endif
		error_CUDA = cudaGetLastError();
		if (error_CUDA != cudaSuccess) throw 0;

		error_CUFFT = DFFT_G(params_OptiX.planr2c, params_OptiX.mu_bitmap_out_bitmap_ref_B, params_OptiX.F_2);
		if (error_CUFFT != CUFFT_SUCCESS) throw 0;

		MultiplyPointwiseComplex<<<(arraySizeComplex + 255) >> 8, 256>>>(params_OptiX.F_2, params_OptiX.F_1, params_OptiX.F_2, arraySizeComplex);
		error_CUDA = cudaGetLastError();
		if (error_CUDA != cudaSuccess) throw 0;

		error_CUFFT = IDFFT_G(params_OptiX.planc2r, params_OptiX.F_2, params_OptiX.mu_bitmap_out_bitmap_ref_B);
		if (error_CUFFT != CUFFT_SUCCESS) throw 0;

		// *********************************************************************************************

	#ifndef SSIM_REDUCE_MEMORY_OVERHEAD
		ComputeArraysForGradientComputation<<<(arraySizeReal + 255) >> 8, 256>>>(
			params_OptiX.mu_bitmap_out_R, params_OptiX.mu_bitmap_ref_R + (params_OptiX.poseNum * arraySizeReal),
			params_OptiX.mu_bitmap_out_bitmap_ref_R,
			params_OptiX.mu_bitmap_out_R_square, params_OptiX.mu_bitmap_ref_R_square + (params_OptiX.poseNum * arraySizeReal),

			params_OptiX.mu_bitmap_out_G, params_OptiX.mu_bitmap_ref_G + (params_OptiX.poseNum * arraySizeReal),
			params_OptiX.mu_bitmap_out_bitmap_ref_G,
			params_OptiX.mu_bitmap_out_G_square, params_OptiX.mu_bitmap_ref_G_square + (params_OptiX.poseNum * arraySizeReal),

			params_OptiX.mu_bitmap_out_B, params_OptiX.mu_bitmap_ref_B + (params_OptiX.poseNum * arraySizeReal),
			params_OptiX.mu_bitmap_out_bitmap_ref_B,
			params_OptiX.mu_bitmap_out_B_square, params_OptiX.mu_bitmap_ref_B_square + (params_OptiX.poseNum * arraySizeReal),

			// !!! !!! !!!
			params_OptiX.mu_bitmap_out_R, params_OptiX.mu_bitmap_out_bitmap_ref_R, params_OptiX.mu_bitmap_out_R_square,
			params_OptiX.mu_bitmap_out_G, params_OptiX.mu_bitmap_out_bitmap_ref_G, params_OptiX.mu_bitmap_out_G_square,
			params_OptiX.mu_bitmap_out_B, params_OptiX.mu_bitmap_out_bitmap_ref_B, params_OptiX.mu_bitmap_out_B_square,
			// !!! !!! !!!

			params_OptiX.width + (kernel_size - 1), params_OptiX.height + (kernel_size - 1), kernel_radius
			);
	#else
		ComputeArraysForGradientComputation<<<(arraySizeReal + 255) >> 8, 256>>>(
			params_OptiX.mu_bitmap_out_R, params_OptiX.mu_bitmap_ref_R + (0 * arraySizeReal),
			params_OptiX.mu_bitmap_out_bitmap_ref_R,
			params_OptiX.mu_bitmap_out_R_square, params_OptiX.mu_bitmap_ref_R_square + (0 * arraySizeReal),

			params_OptiX.mu_bitmap_out_G, params_OptiX.mu_bitmap_ref_G + (0 * arraySizeReal),
			params_OptiX.mu_bitmap_out_bitmap_ref_G,
			params_OptiX.mu_bitmap_out_G_square, params_OptiX.mu_bitmap_ref_G_square + (0 * arraySizeReal),

			params_OptiX.mu_bitmap_out_B, params_OptiX.mu_bitmap_ref_B + (0 * arraySizeReal),
			params_OptiX.mu_bitmap_out_bitmap_ref_B,
			params_OptiX.mu_bitmap_out_B_square, params_OptiX.mu_bitmap_ref_B_square + (0 * arraySizeReal),

			// !!! !!! !!!
			params_OptiX.mu_bitmap_out_R, params_OptiX.mu_bitmap_out_bitmap_ref_R, params_OptiX.mu_bitmap_out_R_square,
			params_OptiX.mu_bitmap_out_G, params_OptiX.mu_bitmap_out_bitmap_ref_G, params_OptiX.mu_bitmap_out_G_square,
			params_OptiX.mu_bitmap_out_B, params_OptiX.mu_bitmap_out_bitmap_ref_B, params_OptiX.mu_bitmap_out_B_square,
			// !!! !!! !!!

			params_OptiX.width + (kernel_size - 1), params_OptiX.height + (kernel_size - 1), kernel_radius
			);
	#endif
		error_CUDA = cudaGetLastError();
		if (error_CUDA != cudaSuccess) throw 0;

		// *********************************************************************************************

		//**********************************************************
		// Compute auxiliary convolutions for gradient computation *
		//**********************************************************

		// convolution_1_R
		error_CUFFT = DFFT_G(params_OptiX.planr2c, params_OptiX.mu_bitmap_out_R, params_OptiX.F_2);
		if (error_CUFFT != CUFFT_SUCCESS) throw 0;

		MultiplyPointwiseComplex<<<(arraySizeComplex + 255) >> 8, 256>>>(params_OptiX.F_2, params_OptiX.F_1, params_OptiX.F_2, arraySizeComplex);
		error_CUDA = cudaGetLastError();
		if (error_CUDA != cudaSuccess) throw 0;

		error_CUFFT = IDFFT_G(params_OptiX.planc2r, params_OptiX.F_2, params_OptiX.mu_bitmap_out_R);
		if (error_CUFFT != CUFFT_SUCCESS) throw 0;

		// *** *** *** *** ***

		// convolution_1_G
		error_CUFFT = DFFT_G(params_OptiX.planr2c, params_OptiX.mu_bitmap_out_G, params_OptiX.F_2);
		if (error_CUFFT != CUFFT_SUCCESS) throw 0;

		MultiplyPointwiseComplex<<<(arraySizeComplex + 255) >> 8, 256>>>(params_OptiX.F_2, params_OptiX.F_1, params_OptiX.F_2, arraySizeComplex);
		error_CUDA = cudaGetLastError();
		if (error_CUDA != cudaSuccess) throw 0;

		error_CUFFT = IDFFT_G(params_OptiX.planc2r, params_OptiX.F_2, params_OptiX.mu_bitmap_out_G);
		if (error_CUFFT != CUFFT_SUCCESS) throw 0;

		// *** *** *** *** ***

		// convolution_1_B
		error_CUFFT = DFFT_G(params_OptiX.planr2c, params_OptiX.mu_bitmap_out_B, params_OptiX.F_2);
		if (error_CUFFT != CUFFT_SUCCESS) throw 0;

		MultiplyPointwiseComplex<<<(arraySizeComplex + 255) >> 8, 256>>>(params_OptiX.F_2, params_OptiX.F_1, params_OptiX.F_2, arraySizeComplex);
		error_CUDA = cudaGetLastError();
		if (error_CUDA != cudaSuccess) throw 0;

		error_CUFFT = IDFFT_G(params_OptiX.planc2r, params_OptiX.F_2, params_OptiX.mu_bitmap_out_B);
		if (error_CUFFT != CUFFT_SUCCESS) throw 0;

		// *** *** *** *** ***

		// convolution_2_R
		error_CUFFT = DFFT_G(params_OptiX.planr2c, params_OptiX.mu_bitmap_out_bitmap_ref_R, params_OptiX.F_2);
		if (error_CUFFT != CUFFT_SUCCESS) throw 0;

		MultiplyPointwiseComplex<<<(arraySizeComplex + 255) >> 8, 256>>>(params_OptiX.F_2, params_OptiX.F_1, params_OptiX.F_2, arraySizeComplex);
		error_CUDA = cudaGetLastError();
		if (error_CUDA != cudaSuccess) throw 0;

		error_CUFFT = IDFFT_G(params_OptiX.planc2r, params_OptiX.F_2, params_OptiX.mu_bitmap_out_bitmap_ref_R);
		if (error_CUFFT != CUFFT_SUCCESS) throw 0;

		// *** *** *** *** ***

		// convolution_2_G
		error_CUFFT = DFFT_G(params_OptiX.planr2c, params_OptiX.mu_bitmap_out_bitmap_ref_G, params_OptiX.F_2);
		if (error_CUFFT != CUFFT_SUCCESS) throw 0;

		MultiplyPointwiseComplex<<<(arraySizeComplex + 255) >> 8, 256>>>(params_OptiX.F_2, params_OptiX.F_1, params_OptiX.F_2, arraySizeComplex);
		error_CUDA = cudaGetLastError();
		if (error_CUDA != cudaSuccess) throw 0;

		error_CUFFT = IDFFT_G(params_OptiX.planc2r, params_OptiX.F_2, params_OptiX.mu_bitmap_out_bitmap_ref_G);
		if (error_CUFFT != CUFFT_SUCCESS) throw 0;

		// *** *** *** *** ***

		// convolution_2_B
		error_CUFFT = DFFT_G(params_OptiX.planr2c, params_OptiX.mu_bitmap_out_bitmap_ref_B, params_OptiX.F_2);
		if (error_CUFFT != CUFFT_SUCCESS) throw 0;

		MultiplyPointwiseComplex<<<(arraySizeComplex + 255) >> 8, 256>>>(params_OptiX.F_2, params_OptiX.F_1, params_OptiX.F_2, arraySizeComplex);
		error_CUDA = cudaGetLastError();
		if (error_CUDA != cudaSuccess) throw 0;

		error_CUFFT = IDFFT_G(params_OptiX.planc2r, params_OptiX.F_2, params_OptiX.mu_bitmap_out_bitmap_ref_B);
		if (error_CUFFT != CUFFT_SUCCESS) throw 0;

		// *** *** *** *** ***

		// convolution_3_R
		error_CUFFT = DFFT_G(params_OptiX.planr2c, params_OptiX.mu_bitmap_out_R_square, params_OptiX.F_2);
		if (error_CUFFT != CUFFT_SUCCESS) throw 0;

		MultiplyPointwiseComplex<<<(arraySizeComplex + 255) >> 8, 256>>>(params_OptiX.F_2, params_OptiX.F_1, params_OptiX.F_2, arraySizeComplex);
		error_CUDA = cudaGetLastError();
		if (error_CUDA != cudaSuccess) throw 0;

		error_CUFFT = IDFFT_G(params_OptiX.planc2r, params_OptiX.F_2, params_OptiX.mu_bitmap_out_R_square);
		if (error_CUFFT != CUFFT_SUCCESS) throw 0;

		// *** *** *** *** ***

		// convolution_3_G
		error_CUFFT = DFFT_G(params_OptiX.planr2c, params_OptiX.mu_bitmap_out_G_square, params_OptiX.F_2);
		if (error_CUFFT != CUFFT_SUCCESS) throw 0;

		MultiplyPointwiseComplex<<<(arraySizeComplex + 255) >> 8, 256>>>(params_OptiX.F_2, params_OptiX.F_1, params_OptiX.F_2, arraySizeComplex);
		error_CUDA = cudaGetLastError();
		if (error_CUDA != cudaSuccess) throw 0;

		error_CUFFT = IDFFT_G(params_OptiX.planc2r, params_OptiX.F_2, params_OptiX.mu_bitmap_out_G_square);
		if (error_CUFFT != CUFFT_SUCCESS) throw 0;

		// *** *** *** *** ***

		// convolution_3_B
		error_CUFFT = DFFT_G(params_OptiX.planr2c, params_OptiX.mu_bitmap_out_B_square, params_OptiX.F_2);
		if (error_CUFFT != CUFFT_SUCCESS) throw 0;

		MultiplyPointwiseComplex<<<(arraySizeComplex + 255) >> 8, 256>>>(params_OptiX.F_2, params_OptiX.F_1, params_OptiX.F_2, arraySizeComplex);
		error_CUDA = cudaGetLastError();
		if (error_CUDA != cudaSuccess) throw 0;

		error_CUFFT = IDFFT_G(params_OptiX.planc2r, params_OptiX.F_2, params_OptiX.mu_bitmap_out_B_square);
		if (error_CUFFT != CUFFT_SUCCESS) throw 0;

		// *********************************************************************************************

	#ifndef SSIM_REDUCE_MEMORY_OVERHEAD
		ComputeGradientSSIM<<<(arraySizeReal + 255) >> 8, 256>>>(
			params_OptiX.mu_bitmap_out_R,
			params_OptiX.mu_bitmap_out_bitmap_ref_R, params_OptiX.bitmap_ref_R + (params_OptiX.poseNum * arraySizeReal),
			params_OptiX.mu_bitmap_out_R_square, params_OptiX.bitmap_out_R,

			params_OptiX.mu_bitmap_out_G,
			params_OptiX.mu_bitmap_out_bitmap_ref_G, params_OptiX.bitmap_ref_G + (params_OptiX.poseNum * arraySizeReal),
			params_OptiX.mu_bitmap_out_G_square, params_OptiX.bitmap_out_G,

			params_OptiX.mu_bitmap_out_B,
			params_OptiX.mu_bitmap_out_bitmap_ref_B, params_OptiX.bitmap_ref_B + (params_OptiX.poseNum * arraySizeReal),
			params_OptiX.mu_bitmap_out_B_square, params_OptiX.bitmap_out_B,

			// !!! !!! !!!
			params_OptiX.mu_bitmap_out_R, params_OptiX.mu_bitmap_out_G, params_OptiX.mu_bitmap_out_B,
			// !!! !!! !!!

			params_OptiX.width + (kernel_size - 1), params_OptiX.height + (kernel_size - 1), kernel_size
			);
	#else
		ComputeGradientSSIM<<<(arraySizeReal + 255) >> 8, 256>>>(
			params_OptiX.mu_bitmap_out_R,
			params_OptiX.mu_bitmap_out_bitmap_ref_R, params_OptiX.bitmap_ref_R + (0 * arraySizeReal),
			params_OptiX.mu_bitmap_out_R_square, params_OptiX.bitmap_out_R,

			params_OptiX.mu_bitmap_out_G,
			params_OptiX.mu_bitmap_out_bitmap_ref_G, params_OptiX.bitmap_ref_G + (0 * arraySizeReal),
			params_OptiX.mu_bitmap_out_G_square, params_OptiX.bitmap_out_G,

			params_OptiX.mu_bitmap_out_B,
			params_OptiX.mu_bitmap_out_bitmap_ref_B, params_OptiX.bitmap_ref_B + (0 * arraySizeReal),
			params_OptiX.mu_bitmap_out_B_square, params_OptiX.bitmap_out_B,

			// !!! !!! !!!
			params_OptiX.mu_bitmap_out_R, params_OptiX.mu_bitmap_out_G, params_OptiX.mu_bitmap_out_B,
			// !!! !!! !!!

			params_OptiX.width + (kernel_size - 1), params_OptiX.height + (kernel_size - 1), kernel_size
		);
	#endif
		error_CUDA = cudaGetLastError();
		if (error_CUDA != cudaSuccess) throw 0;

		// *********************************************************************************************

		// TEST
		/*if (params_OptiX.epoch == 1000) {
		REAL_G *buf = NULL;
		buf = (REAL_G *)malloc(sizeof(REAL_G) * arraySizeReal);
		if (buf == NULL) throw 0;
		unsigned *bitmap_ref = (unsigned *)malloc(sizeof(unsigned) * params_OptiX.width * params_OptiX.height);

		// R channel
		error_CUDA = cudaMemcpy(buf, params_OptiX.mu_bitmap_out_R, sizeof(REAL_G) * arraySizeReal, cudaMemcpyDeviceToHost);
		if (error_CUDA != cudaSuccess) throw 0;

		for (int i = 0; i < params_OptiX.height; ++i) {
		for (int j = 0; j < params_OptiX.width; ++j) {
		REAL_G Rf = buf[((kernel_radius + i) * (params_OptiX.width + (kernel_size - 1))) + (kernel_radius + j)] / arraySizeReal;
		if (Rf < ((REAL_G)0.0)) Rf = ((REAL_G)0.0);
		if (Rf > ((REAL_G)1.0)) Rf = ((REAL_G)1.0);
		unsigned char Ri = Rf * ((REAL_G)255.0);
		bitmap_ref[(i * params_OptiX.width) + j] = (((unsigned) Ri) << 16);
		}
		}

		// G channel
		error_CUDA = cudaMemcpy(buf, params_OptiX.mu_bitmap_out_G, sizeof(REAL_G) *  arraySizeReal, cudaMemcpyDeviceToHost);
		if (error_CUDA != cudaSuccess) throw 0;

		for (int i = 0; i < params_OptiX.height; ++i) {
		for (int j = 0; j < params_OptiX.width; ++j) {
		REAL_G Gf = buf[((kernel_radius + i) * (params_OptiX.width + (kernel_size - 1))) + (kernel_radius + j)] / arraySizeReal;
		if (Gf < ((REAL_G)0.0)) Gf = ((REAL_G)0.0);
		if (Gf > ((REAL_G)1.0)) Gf = ((REAL_G)1.0);
		unsigned char Gi = Gf * ((REAL_G)255.0);
		bitmap_ref[(i * params_OptiX.width) + j] |= (((unsigned) Gi) << 8);
		}
		}

		// B channel
		error_CUDA = cudaMemcpy(buf, params_OptiX.mu_bitmap_out_B, sizeof(REAL_G) *  arraySizeReal, cudaMemcpyDeviceToHost);
		if (error_CUDA != cudaSuccess) throw 0;

		for (int i = 0; i < params_OptiX.height; ++i) {
		for (int j = 0; j < params_OptiX.width; ++j) {
		REAL_G Bf = buf[((kernel_radius + i) * (params_OptiX.width + (kernel_size - 1))) + (kernel_radius + j)] / arraySizeReal;
		if (Bf < ((REAL_G)0.0)) Bf = ((REAL_G)0.0);
		if (Bf > ((REAL_G)1.0)) Bf = ((REAL_G)1.0);
		unsigned char Bi = Bf * ((REAL_G)255.0);
		bitmap_ref[(i * params_OptiX.width) + j] |= ((unsigned)Bi);
		}
		}

		// Copy to bitmap on hdd
		unsigned char *foo = (unsigned char *)malloc(3 * params_OptiX.width * params_OptiX.height);
		for (int i = 0; i < params_OptiX.height; ++i) {
		for (int j = 0; j < params_OptiX.width; ++j) {
		unsigned char R = bitmap_ref[(i * params_OptiX.width) + j] >> 16;
		unsigned char G = (bitmap_ref[(i * params_OptiX.width) + j] >> 8) & 255;
		unsigned char B = bitmap_ref[(i * params_OptiX.width) + j] & 255;
		foo[((((params_OptiX.width - 1 - i) * params_OptiX.height) + j) * 3) + 2] = R;
		foo[((((params_OptiX.width - 1 - i) * params_OptiX.height) + j) * 3) + 1] = G;
		foo[(((params_OptiX.width - 1 - i) * params_OptiX.height) + j) * 3] = B;		
		}
		}

		FILE *f = fopen("test.bmp", "rb+");
		fseek(f, 54, SEEK_SET);
		fwrite(foo, sizeof(int) * params_OptiX.width * params_OptiX.height, 1, f);
		fclose(f);

		free(buf);
		free(bitmap_ref);
		}*/

		//***********************************************************************************************

		ComputeGradient<SH_degree><<<((params_OptiX.width * params_OptiX.height) + 63) >> 6, 64>>>(params_OptiX);
		error_CUDA = cudaGetLastError();
		if (error_CUDA != cudaSuccess) throw 0;

		// *** *** *** *** ***

		bool densification = (
			(params_OptiX.epoch >= densification_start_epoch_host) &&
			(params_OptiX.epoch <= densification_end_epoch_host) &&
			((params_OptiX.epoch % densification_frequency_host) == 0)
		);
		bool next_densification = (
			densification &&
			(params_OptiX.epoch + densification_frequency_host <= densification_end_epoch_host)
		);

		// *** *** *** *** ***

		if (densification) {
			error_CUDA = cudaMemset(params_OptiX.counter1, 0, sizeof(unsigned) * 1);
			if (error_CUDA != cudaSuccess) throw 0;

			// !!! !!! !!! EXPERIMENTAL !!! !!! !!!
			cuMemsetD32(((CUdeviceptr)needsToBeRemoved_host), 1, params_OptiX.tmpArraysGroup1Size);
			// !!! !!! !!! EXPERIMENTAL !!! !!! !!!
		}

		// *** *** *** *** ***

		dev_UpdateGradientOptiX<SH_degree><<<(params_OptiX.numberOfGaussians + 63) >> 6, 64>>>(params_OptiX);
		error_CUDA = cudaGetLastError();
		if (error_CUDA != cudaSuccess) throw 0;

		// *** *** *** *** ***

		if (densification) {
			// !!! !!! !!! EXPERIMENTAL !!! !!! !!!
			int numberOfNewGaussians;

			error_CUDA = cudaMemcpy(&numberOfNewGaussians, params_OptiX.counter1, sizeof(unsigned) * 1, cudaMemcpyDeviceToHost);
			if (error_CUDA != cudaSuccess) throw 0;

			params_OptiX.numberOfGaussians += numberOfNewGaussians; // !!! !!! !!!

			// !!! !!! !!!
			if (params_OptiX.numberOfGaussians > params_OptiX.scatterBufferSize) {
				params_OptiX.scatterBufferSize = params_OptiX.numberOfGaussians; // !!! !!! !!!

				error_CUDA = cudaFree(Gaussians_indices_after_removal_host);
				if (error_CUDA != cudaSuccess) throw 0;

				error_CUDA = cudaFree(scatterBuffer);
				if (error_CUDA != cudaSuccess) throw 0;

				error_CUDA = cudaMalloc(&Gaussians_indices_after_removal_host, sizeof(int) * params_OptiX.scatterBufferSize);
				if (error_CUDA != cudaSuccess) throw 0;

				error_CUDA = cudaMalloc(&scatterBuffer, sizeof(float) * 6 * ((params_OptiX.scatterBufferSize + 31) & -32));
				if (error_CUDA != cudaSuccess) throw 0;
			}
			// !!! !!! !!!

			thrust::exclusive_scan(
				thrust::device_pointer_cast(needsToBeRemoved_host),
				thrust::device_pointer_cast(needsToBeRemoved_host) + params_OptiX.numberOfGaussians,
				thrust::device_pointer_cast(Gaussians_indices_after_removal_host)
			);

			int numberOfGaussiansNew = thrust::reduce(
				thrust::device_pointer_cast(needsToBeRemoved_host),
				thrust::device_pointer_cast(needsToBeRemoved_host) + params_OptiX.numberOfGaussians
			);

			// ************************************************************************************************

			// !!! !!! !!!
			if (numberOfGaussiansNew > params_OptiX.tmpArraysGroup2Size) {
				params_OptiX.tmpArraysGroup2Size = numberOfGaussiansNew; // !!! !!! !!!
				
				// *** *** *** *** ***

				error_CUDA = cudaFree(params_OptiX.dL_dparams_1);
				if (error_CUDA != cudaSuccess) throw 0;

				error_CUDA = cudaFree(params_OptiX.dL_dparams_2);
				if (error_CUDA != cudaSuccess) throw 0;

				error_CUDA = cudaFree(params_OptiX.dL_dparams_3);
				if (error_CUDA != cudaSuccess) throw 0;

				error_CUDA = cudaFree(params_OptiX.dL_dparams_4);
				if (error_CUDA != cudaSuccess) throw 0;

				// Spherical harmonics
				if constexpr (SH_degree >= 1) {
					error_CUDA = cudaFree(params_OptiX.dL_dparams_SH_1);
					if (error_CUDA != cudaSuccess) throw 0;

					error_CUDA = cudaFree(params_OptiX.dL_dparams_SH_2);
					if (error_CUDA != cudaSuccess) throw 0;

					error_CUDA = cudaFree(params_OptiX.dL_dparams_SH_3);
					if (error_CUDA != cudaSuccess) throw 0;

					if constexpr (SH_degree >= 2) {
						error_CUDA = cudaFree(params_OptiX.dL_dparams_SH_4);
						if (error_CUDA != cudaSuccess) throw 0;

						error_CUDA = cudaFree(params_OptiX.dL_dparams_SH_5);
						if (error_CUDA != cudaSuccess) throw 0;

						error_CUDA = cudaFree(params_OptiX.dL_dparams_SH_6);
						if (error_CUDA != cudaSuccess) throw 0;

						if constexpr (SH_degree >= 3) {
							error_CUDA = cudaFree(params_OptiX.dL_dparams_SH_7);
							if (error_CUDA != cudaSuccess) throw 0;

							error_CUDA = cudaFree(params_OptiX.dL_dparams_SH_8);
							if (error_CUDA != cudaSuccess) throw 0;

							error_CUDA = cudaFree(params_OptiX.dL_dparams_SH_9);
							if (error_CUDA != cudaSuccess) throw 0;

							error_CUDA = cudaFree(params_OptiX.dL_dparams_SH_10);
							if (error_CUDA != cudaSuccess) throw 0;

							error_CUDA = cudaFree(params_OptiX.dL_dparams_SH_11);
							if (error_CUDA != cudaSuccess) throw 0;

							error_CUDA = cudaFree(params_OptiX.dL_dparams_SH_12);
							if (error_CUDA != cudaSuccess) throw 0;
						
							if constexpr (SH_degree >= 4) {
								error_CUDA = cudaFree(params_OptiX.dL_dparams_SH_13);
								if (error_CUDA != cudaSuccess) throw 0;

								error_CUDA = cudaFree(params_OptiX.dL_dparams_SH_14);
								if (error_CUDA != cudaSuccess) throw 0;

								error_CUDA = cudaFree(params_OptiX.dL_dparams_SH_15);
								if (error_CUDA != cudaSuccess) throw 0;

								error_CUDA = cudaFree(params_OptiX.dL_dparams_SH_16);
								if (error_CUDA != cudaSuccess) throw 0;

								error_CUDA = cudaFree(params_OptiX.dL_dparams_SH_17);
								if (error_CUDA != cudaSuccess) throw 0;

								error_CUDA = cudaFree(params_OptiX.dL_dparams_SH_18);
								if (error_CUDA != cudaSuccess) throw 0;
							}
						}
					}
				}

				// *** *** *** *** ***

				error_CUDA = cudaMalloc(&params_OptiX.dL_dparams_1, sizeof(REAL4_G) * params_OptiX.tmpArraysGroup2Size); // !!! !!! !!!
				if (error_CUDA != cudaSuccess) throw 0;

				error_CUDA = cudaMalloc(&params_OptiX.dL_dparams_2, sizeof(REAL4_G) * params_OptiX.tmpArraysGroup2Size); // !!! !!! !!!
				if (error_CUDA != cudaSuccess) throw 0;

				error_CUDA = cudaMalloc(&params_OptiX.dL_dparams_3, sizeof(REAL4_G) * params_OptiX.tmpArraysGroup2Size); // !!! !!! !!!
				if (error_CUDA != cudaSuccess) throw 0;

				// !!! !!! !!!
				error_CUDA = cudaMalloc(&params_OptiX.dL_dparams_4, sizeof(REAL2_G) * params_OptiX.tmpArraysGroup2Size); // !!! !!! !!!
				if (error_CUDA != cudaSuccess) throw 0;

				// Spherical harmonics
				if constexpr (SH_degree >= 1) {
					error_CUDA = cudaMalloc(&params_OptiX.dL_dparams_SH_1, sizeof(REAL4_G) * params_OptiX.tmpArraysGroup2Size); // !!! !!! !!!
					if (error_CUDA != cudaSuccess) throw 0;

					error_CUDA = cudaMalloc(&params_OptiX.dL_dparams_SH_2, sizeof(REAL4_G) * params_OptiX.tmpArraysGroup2Size); // !!! !!! !!!
					if (error_CUDA != cudaSuccess) throw 0;

					if constexpr (SH_degree >= 2) {
						error_CUDA = cudaMalloc(&params_OptiX.dL_dparams_SH_3, sizeof(REAL4_G) * params_OptiX.tmpArraysGroup2Size); // !!! !!! !!!
						if (error_CUDA != cudaSuccess) throw 0;

						error_CUDA = cudaMalloc(&params_OptiX.dL_dparams_SH_4, sizeof(REAL4_G) * params_OptiX.tmpArraysGroup2Size); // !!! !!! !!!
						if (error_CUDA != cudaSuccess) throw 0;

						error_CUDA = cudaMalloc(&params_OptiX.dL_dparams_SH_5, sizeof(REAL4_G) * params_OptiX.tmpArraysGroup2Size); // !!! !!! !!!
						if (error_CUDA != cudaSuccess) throw 0;

						error_CUDA = cudaMalloc(&params_OptiX.dL_dparams_SH_6, sizeof(REAL4_G) * params_OptiX.tmpArraysGroup2Size); // !!! !!! !!!
						if (error_CUDA != cudaSuccess) throw 0;

						if constexpr (SH_degree >= 3) {
							error_CUDA = cudaMalloc(&params_OptiX.dL_dparams_SH_7, sizeof(REAL4_G) * params_OptiX.tmpArraysGroup2Size); // !!! !!! !!!
							if (error_CUDA != cudaSuccess) throw 0;

							error_CUDA = cudaMalloc(&params_OptiX.dL_dparams_SH_8, sizeof(REAL4_G) * params_OptiX.tmpArraysGroup2Size); // !!! !!! !!!
							if (error_CUDA != cudaSuccess) throw 0;

							error_CUDA = cudaMalloc(&params_OptiX.dL_dparams_SH_9, sizeof(REAL4_G) * params_OptiX.tmpArraysGroup2Size); // !!! !!! !!!
							if (error_CUDA != cudaSuccess) throw 0;

							error_CUDA = cudaMalloc(&params_OptiX.dL_dparams_SH_10, sizeof(REAL4_G) * params_OptiX.tmpArraysGroup2Size); // !!! !!! !!!
							if (error_CUDA != cudaSuccess) throw 0;

							error_CUDA = cudaMalloc(&params_OptiX.dL_dparams_SH_11, sizeof(REAL4_G) * params_OptiX.tmpArraysGroup2Size); // !!! !!! !!!
							if (error_CUDA != cudaSuccess) throw 0;

							if constexpr (SH_degree >= 4) {
								error_CUDA = cudaMalloc(&params_OptiX.dL_dparams_SH_12, sizeof(REAL4_G) * params_OptiX.tmpArraysGroup2Size); // !!! !!! !!!
								if (error_CUDA != cudaSuccess) throw 0;

								error_CUDA = cudaMalloc(&params_OptiX.dL_dparams_SH_13, sizeof(REAL4_G) * params_OptiX.tmpArraysGroup2Size); // !!! !!! !!!
								if (error_CUDA != cudaSuccess) throw 0;

								error_CUDA = cudaMalloc(&params_OptiX.dL_dparams_SH_14, sizeof(REAL4_G) * params_OptiX.tmpArraysGroup2Size); // !!! !!! !!!
								if (error_CUDA != cudaSuccess) throw 0;

								error_CUDA = cudaMalloc(&params_OptiX.dL_dparams_SH_15, sizeof(REAL4_G) * params_OptiX.tmpArraysGroup2Size); // !!! !!! !!!
								if (error_CUDA != cudaSuccess) throw 0;

								error_CUDA = cudaMalloc(&params_OptiX.dL_dparams_SH_16, sizeof(REAL4_G) * params_OptiX.tmpArraysGroup2Size); // !!! !!! !!!
								if (error_CUDA != cudaSuccess) throw 0;

								error_CUDA = cudaMalloc(&params_OptiX.dL_dparams_SH_17, sizeof(REAL4_G) * params_OptiX.tmpArraysGroup2Size); // !!! !!! !!!
								if (error_CUDA != cudaSuccess) throw 0;

								error_CUDA = cudaMalloc(&params_OptiX.dL_dparams_SH_18, sizeof(REAL4_G) * params_OptiX.tmpArraysGroup2Size); // !!! !!! !!!
								if (error_CUDA != cudaSuccess) throw 0;
							} else {
								// !!! !!! !!!
								error_CUDA = cudaMalloc(&params_OptiX.dL_dparams_SH_12, sizeof(REAL_G) * params_OptiX.tmpArraysGroup2Size); // !!! !!! !!!
								if (error_CUDA != cudaSuccess) throw 0;
							}
						}
					} else {
						// !!! !!! !!!
						error_CUDA = cudaMalloc(&params_OptiX.dL_dparams_SH_3, sizeof(REAL_G) * params_OptiX.tmpArraysGroup2Size); // !!! !!! !!!
						if (error_CUDA != cudaSuccess) throw 0;
					}
				}

				// *** *** *** *** ***

				error_CUDA = cudaFree(params_OptiX.instancesBuffer);
				if (error_CUDA != cudaSuccess) throw 0;

				error_CUDA = cudaMalloc(&params_OptiX.instancesBuffer, sizeof(OptixInstance) * ((params_OptiX.tmpArraysGroup2Size + 31) & -32)); // !!! !!! !!!
				if (error_CUDA != cudaSuccess) throw 0;
			}
			// !!! !!! !!!

			// *** *** *** *** ***

			// !!! !!! !!!
			bool needsToReallocMemory = false;
			if (
				next_densification &&
				(numberOfGaussiansNew * tmp_arrays_growth_factor_host > params_OptiX.tmpArraysGroup1Size) && 
				(numberOfGaussiansNew <= max_Gaussians_per_model_host)
			) {
				needsToReallocMemory = true;
				params_OptiX.tmpArraysGroup1Size = numberOfGaussiansNew * tmp_arrays_growth_factor_host;
			} else {
				if (!next_densification) {
					needsToReallocMemory = true;
					params_OptiX.tmpArraysGroup1Size = numberOfGaussiansNew;
				}
			}
			// !!! !!! !!!

			// *** *** *** *** ***

			// GC
			thrust::scatter_if(
				thrust::device_pointer_cast(params_OptiX.GC_part_1_1),
				thrust::device_pointer_cast(params_OptiX.GC_part_1_1) + params_OptiX.numberOfGaussians,
				thrust::device_pointer_cast(Gaussians_indices_after_removal_host),
				thrust::device_pointer_cast(needsToBeRemoved_host),
				thrust::device_pointer_cast((float4 *)scatterBuffer)
			);
			if (needsToReallocMemory) {
				error_CUDA = cudaFree(params_OptiX.GC_part_1_1);
				if (error_CUDA != cudaSuccess) throw 0;

				error_CUDA = cudaMalloc(&params_OptiX.GC_part_1_1, sizeof(float4) * params_OptiX.tmpArraysGroup1Size);
				if (error_CUDA != cudaSuccess) throw 0;
			}
			cudaMemcpy(params_OptiX.GC_part_1_1, scatterBuffer, sizeof(float4) * numberOfGaussiansNew, cudaMemcpyDeviceToDevice);

			thrust::scatter_if(
				thrust::device_pointer_cast(params_OptiX.GC_part_2_1),
				thrust::device_pointer_cast(params_OptiX.GC_part_2_1) + params_OptiX.numberOfGaussians,
				thrust::device_pointer_cast(Gaussians_indices_after_removal_host),
				thrust::device_pointer_cast(needsToBeRemoved_host),
				thrust::device_pointer_cast((float4 *)scatterBuffer)
			);
			if (needsToReallocMemory) {
				error_CUDA = cudaFree(params_OptiX.GC_part_2_1);
				if (error_CUDA != cudaSuccess) throw 0;

				error_CUDA = cudaMalloc(&params_OptiX.GC_part_2_1, sizeof(float4) * params_OptiX.tmpArraysGroup1Size);
				if (error_CUDA != cudaSuccess) throw 0;
			}
			cudaMemcpy(params_OptiX.GC_part_2_1, scatterBuffer, sizeof(float4) * numberOfGaussiansNew, cudaMemcpyDeviceToDevice);

			thrust::scatter_if(
				thrust::device_pointer_cast(params_OptiX.GC_part_3_1),
				thrust::device_pointer_cast(params_OptiX.GC_part_3_1) + params_OptiX.numberOfGaussians,
				thrust::device_pointer_cast(Gaussians_indices_after_removal_host),
				thrust::device_pointer_cast(needsToBeRemoved_host),
				thrust::device_pointer_cast((float4 *)scatterBuffer)
			);
			if (needsToReallocMemory) {
				error_CUDA = cudaFree(params_OptiX.GC_part_3_1);
				if (error_CUDA != cudaSuccess) throw 0;

				error_CUDA = cudaMalloc(&params_OptiX.GC_part_3_1, sizeof(float4) * params_OptiX.tmpArraysGroup1Size);
				if (error_CUDA != cudaSuccess) throw 0;
			}
			cudaMemcpy(params_OptiX.GC_part_3_1, scatterBuffer, sizeof(float4) * numberOfGaussiansNew, cudaMemcpyDeviceToDevice);

			thrust::scatter_if(
				thrust::device_pointer_cast(params_OptiX.GC_part_4_1),
				thrust::device_pointer_cast(params_OptiX.GC_part_4_1) + params_OptiX.numberOfGaussians,
				thrust::device_pointer_cast(Gaussians_indices_after_removal_host),
				thrust::device_pointer_cast(needsToBeRemoved_host),
				thrust::device_pointer_cast((float2 *)scatterBuffer)
			);
			if (needsToReallocMemory) {
				error_CUDA = cudaFree(params_OptiX.GC_part_4_1);
				if (error_CUDA != cudaSuccess) throw 0;

				// !!! !!! !!!
				error_CUDA = cudaMalloc(&params_OptiX.GC_part_4_1, sizeof(float2) * params_OptiX.tmpArraysGroup1Size);
				if (error_CUDA != cudaSuccess) throw 0;
			}
			// !!! !!! !!!
			cudaMemcpy(params_OptiX.GC_part_4_1, scatterBuffer, sizeof(float2) * numberOfGaussiansNew, cudaMemcpyDeviceToDevice);

			// Spherical harmonics
			if constexpr (SH_degree >= 1) {
				// GC_SH_1
				thrust::scatter_if(
					thrust::device_pointer_cast(params_OptiX.GC_SH_1),
					thrust::device_pointer_cast(params_OptiX.GC_SH_1) + params_OptiX.numberOfGaussians,
					thrust::device_pointer_cast(Gaussians_indices_after_removal_host),
					thrust::device_pointer_cast(needsToBeRemoved_host),
					thrust::device_pointer_cast((float4 *)scatterBuffer)
				);
				if (needsToReallocMemory) {
					error_CUDA = cudaFree(params_OptiX.GC_SH_1);
					if (error_CUDA != cudaSuccess) throw 0;

					error_CUDA = cudaMalloc(&params_OptiX.GC_SH_1, sizeof(float4) * params_OptiX.tmpArraysGroup1Size);
					if (error_CUDA != cudaSuccess) throw 0;
				}
				cudaMemcpy(params_OptiX.GC_SH_1, scatterBuffer, sizeof(float4) * numberOfGaussiansNew, cudaMemcpyDeviceToDevice);

				// GC_SH_2
				thrust::scatter_if(
					thrust::device_pointer_cast(params_OptiX.GC_SH_2),
					thrust::device_pointer_cast(params_OptiX.GC_SH_2) + params_OptiX.numberOfGaussians,
					thrust::device_pointer_cast(Gaussians_indices_after_removal_host),
					thrust::device_pointer_cast(needsToBeRemoved_host),
					thrust::device_pointer_cast((float4 *)scatterBuffer)
				);
				if (needsToReallocMemory) {
					error_CUDA = cudaFree(params_OptiX.GC_SH_2);
					if (error_CUDA != cudaSuccess) throw 0;

					error_CUDA = cudaMalloc(&params_OptiX.GC_SH_2, sizeof(float4) * params_OptiX.tmpArraysGroup1Size);
					if (error_CUDA != cudaSuccess) throw 0;
				}
				cudaMemcpy(params_OptiX.GC_SH_2, scatterBuffer, sizeof(float4) * numberOfGaussiansNew, cudaMemcpyDeviceToDevice);

				if constexpr (SH_degree >= 2) {
					// GC_SH_3
					thrust::scatter_if(
						thrust::device_pointer_cast(params_OptiX.GC_SH_3),
						thrust::device_pointer_cast(params_OptiX.GC_SH_3) + params_OptiX.numberOfGaussians,
						thrust::device_pointer_cast(Gaussians_indices_after_removal_host),
						thrust::device_pointer_cast(needsToBeRemoved_host),
						thrust::device_pointer_cast((float4 *)scatterBuffer)
					);
					if (needsToReallocMemory) {
						error_CUDA = cudaFree(params_OptiX.GC_SH_3);
						if (error_CUDA != cudaSuccess) throw 0;

						error_CUDA = cudaMalloc(&params_OptiX.GC_SH_3, sizeof(float4) * params_OptiX.tmpArraysGroup1Size);
						if (error_CUDA != cudaSuccess) throw 0;
					}
					cudaMemcpy(params_OptiX.GC_SH_3, scatterBuffer, sizeof(float4) * numberOfGaussiansNew, cudaMemcpyDeviceToDevice);

					// GC_SH_4
					thrust::scatter_if(
						thrust::device_pointer_cast(params_OptiX.GC_SH_4),
						thrust::device_pointer_cast(params_OptiX.GC_SH_4) + params_OptiX.numberOfGaussians,
						thrust::device_pointer_cast(Gaussians_indices_after_removal_host),
						thrust::device_pointer_cast(needsToBeRemoved_host),
						thrust::device_pointer_cast((float4 *)scatterBuffer)
					);
					if (needsToReallocMemory) {
						error_CUDA = cudaFree(params_OptiX.GC_SH_4);
						if (error_CUDA != cudaSuccess) throw 0;

						error_CUDA = cudaMalloc(&params_OptiX.GC_SH_4, sizeof(float4) * params_OptiX.tmpArraysGroup1Size);
						if (error_CUDA != cudaSuccess) throw 0;
					}
					cudaMemcpy(params_OptiX.GC_SH_4, scatterBuffer, sizeof(float4) * numberOfGaussiansNew, cudaMemcpyDeviceToDevice);

					// GC_SH_5
					thrust::scatter_if(
						thrust::device_pointer_cast(params_OptiX.GC_SH_5),
						thrust::device_pointer_cast(params_OptiX.GC_SH_5) + params_OptiX.numberOfGaussians,
						thrust::device_pointer_cast(Gaussians_indices_after_removal_host),
						thrust::device_pointer_cast(needsToBeRemoved_host),
						thrust::device_pointer_cast((float4 *)scatterBuffer)
					);
					if (needsToReallocMemory) {
						error_CUDA = cudaFree(params_OptiX.GC_SH_5);
						if (error_CUDA != cudaSuccess) throw 0;

						error_CUDA = cudaMalloc(&params_OptiX.GC_SH_5, sizeof(float4) * params_OptiX.tmpArraysGroup1Size);
						if (error_CUDA != cudaSuccess) throw 0;
					}
					cudaMemcpy(params_OptiX.GC_SH_5, scatterBuffer, sizeof(float4) * numberOfGaussiansNew, cudaMemcpyDeviceToDevice);

					// GC_SH_6
					thrust::scatter_if(
						thrust::device_pointer_cast(params_OptiX.GC_SH_6),
						thrust::device_pointer_cast(params_OptiX.GC_SH_6) + params_OptiX.numberOfGaussians,
						thrust::device_pointer_cast(Gaussians_indices_after_removal_host),
						thrust::device_pointer_cast(needsToBeRemoved_host),
						thrust::device_pointer_cast((float4 *)scatterBuffer)
					);
					if (needsToReallocMemory) {
						error_CUDA = cudaFree(params_OptiX.GC_SH_6);
						if (error_CUDA != cudaSuccess) throw 0;

						error_CUDA = cudaMalloc(&params_OptiX.GC_SH_6, sizeof(float4) * params_OptiX.tmpArraysGroup1Size);
						if (error_CUDA != cudaSuccess) throw 0;
					}
					cudaMemcpy(params_OptiX.GC_SH_6, scatterBuffer, sizeof(float4) * numberOfGaussiansNew, cudaMemcpyDeviceToDevice);

					if constexpr (SH_degree >= 3) {
						// GC_SH_7
						thrust::scatter_if(
							thrust::device_pointer_cast(params_OptiX.GC_SH_7),
							thrust::device_pointer_cast(params_OptiX.GC_SH_7) + params_OptiX.numberOfGaussians,
							thrust::device_pointer_cast(Gaussians_indices_after_removal_host),
							thrust::device_pointer_cast(needsToBeRemoved_host),
							thrust::device_pointer_cast((float4 *)scatterBuffer)
						);
						if (needsToReallocMemory) {
							error_CUDA = cudaFree(params_OptiX.GC_SH_7);
							if (error_CUDA != cudaSuccess) throw 0;

							error_CUDA = cudaMalloc(&params_OptiX.GC_SH_7, sizeof(float4) * params_OptiX.tmpArraysGroup1Size);
							if (error_CUDA != cudaSuccess) throw 0;
						}
						cudaMemcpy(params_OptiX.GC_SH_7, scatterBuffer, sizeof(float4) * numberOfGaussiansNew, cudaMemcpyDeviceToDevice);

						// GC_SH_8
						thrust::scatter_if(
							thrust::device_pointer_cast(params_OptiX.GC_SH_8),
							thrust::device_pointer_cast(params_OptiX.GC_SH_8) + params_OptiX.numberOfGaussians,
							thrust::device_pointer_cast(Gaussians_indices_after_removal_host),
							thrust::device_pointer_cast(needsToBeRemoved_host),
							thrust::device_pointer_cast((float4 *)scatterBuffer)
						);
						if (needsToReallocMemory) {
							error_CUDA = cudaFree(params_OptiX.GC_SH_8);
							if (error_CUDA != cudaSuccess) throw 0;

							error_CUDA = cudaMalloc(&params_OptiX.GC_SH_8, sizeof(float4) * params_OptiX.tmpArraysGroup1Size);
							if (error_CUDA != cudaSuccess) throw 0;
						}
						cudaMemcpy(params_OptiX.GC_SH_8, scatterBuffer, sizeof(float4) * numberOfGaussiansNew, cudaMemcpyDeviceToDevice);

						// GC_SH_9
						thrust::scatter_if(
							thrust::device_pointer_cast(params_OptiX.GC_SH_9),
							thrust::device_pointer_cast(params_OptiX.GC_SH_9) + params_OptiX.numberOfGaussians,
							thrust::device_pointer_cast(Gaussians_indices_after_removal_host),
							thrust::device_pointer_cast(needsToBeRemoved_host),
							thrust::device_pointer_cast((float4 *)scatterBuffer)
						);
						if (needsToReallocMemory) {
							error_CUDA = cudaFree(params_OptiX.GC_SH_9);
							if (error_CUDA != cudaSuccess) throw 0;

							error_CUDA = cudaMalloc(&params_OptiX.GC_SH_9, sizeof(float4) * params_OptiX.tmpArraysGroup1Size);
							if (error_CUDA != cudaSuccess) throw 0;
						}
						cudaMemcpy(params_OptiX.GC_SH_9, scatterBuffer, sizeof(float4) * numberOfGaussiansNew, cudaMemcpyDeviceToDevice);

						// GC_SH_10
						thrust::scatter_if(
							thrust::device_pointer_cast(params_OptiX.GC_SH_10),
							thrust::device_pointer_cast(params_OptiX.GC_SH_10) + params_OptiX.numberOfGaussians,
							thrust::device_pointer_cast(Gaussians_indices_after_removal_host),
							thrust::device_pointer_cast(needsToBeRemoved_host),
							thrust::device_pointer_cast((float4 *)scatterBuffer)
						);
						if (needsToReallocMemory) {
							error_CUDA = cudaFree(params_OptiX.GC_SH_10);
							if (error_CUDA != cudaSuccess) throw 0;

							error_CUDA = cudaMalloc(&params_OptiX.GC_SH_10, sizeof(float4) * params_OptiX.tmpArraysGroup1Size);
							if (error_CUDA != cudaSuccess) throw 0;
						}
						cudaMemcpy(params_OptiX.GC_SH_10, scatterBuffer, sizeof(float4) * numberOfGaussiansNew, cudaMemcpyDeviceToDevice);

						// GC_SH_11
						thrust::scatter_if(
							thrust::device_pointer_cast(params_OptiX.GC_SH_11),
							thrust::device_pointer_cast(params_OptiX.GC_SH_11) + params_OptiX.numberOfGaussians,
							thrust::device_pointer_cast(Gaussians_indices_after_removal_host),
							thrust::device_pointer_cast(needsToBeRemoved_host),
							thrust::device_pointer_cast((float4 *)scatterBuffer)
						);
						if (needsToReallocMemory) {
							error_CUDA = cudaFree(params_OptiX.GC_SH_11);
							if (error_CUDA != cudaSuccess) throw 0;

							error_CUDA = cudaMalloc(&params_OptiX.GC_SH_11, sizeof(float4) * params_OptiX.tmpArraysGroup1Size);
							if (error_CUDA != cudaSuccess) throw 0;
						}
						cudaMemcpy(params_OptiX.GC_SH_11, scatterBuffer, sizeof(float4) * numberOfGaussiansNew, cudaMemcpyDeviceToDevice);

						if constexpr (SH_degree >= 4) {
							// GC_SH_12
							thrust::scatter_if(
								thrust::device_pointer_cast(params_OptiX.GC_SH_12),
								thrust::device_pointer_cast(params_OptiX.GC_SH_12) + params_OptiX.numberOfGaussians,
								thrust::device_pointer_cast(Gaussians_indices_after_removal_host),
								thrust::device_pointer_cast(needsToBeRemoved_host),
								thrust::device_pointer_cast((float4 *)scatterBuffer)
							);
							if (needsToReallocMemory) {
								error_CUDA = cudaFree(params_OptiX.GC_SH_12);
								if (error_CUDA != cudaSuccess) throw 0;

								error_CUDA = cudaMalloc(&params_OptiX.GC_SH_12, sizeof(float4) * params_OptiX.tmpArraysGroup1Size);
								if (error_CUDA != cudaSuccess) throw 0;
							}
							cudaMemcpy(params_OptiX.GC_SH_12, scatterBuffer, sizeof(float4) * numberOfGaussiansNew, cudaMemcpyDeviceToDevice);

							// GC_SH_13
							thrust::scatter_if(
								thrust::device_pointer_cast(params_OptiX.GC_SH_13),
								thrust::device_pointer_cast(params_OptiX.GC_SH_13) + params_OptiX.numberOfGaussians,
								thrust::device_pointer_cast(Gaussians_indices_after_removal_host),
								thrust::device_pointer_cast(needsToBeRemoved_host),
								thrust::device_pointer_cast((float4 *)scatterBuffer)
							);
							if (needsToReallocMemory) {
								error_CUDA = cudaFree(params_OptiX.GC_SH_13);
								if (error_CUDA != cudaSuccess) throw 0;

								error_CUDA = cudaMalloc(&params_OptiX.GC_SH_13, sizeof(float4) * params_OptiX.tmpArraysGroup1Size);
								if (error_CUDA != cudaSuccess) throw 0;
							}
							cudaMemcpy(params_OptiX.GC_SH_13, scatterBuffer, sizeof(float4) * numberOfGaussiansNew, cudaMemcpyDeviceToDevice);

							// GC_SH_14
							thrust::scatter_if(
								thrust::device_pointer_cast(params_OptiX.GC_SH_14),
								thrust::device_pointer_cast(params_OptiX.GC_SH_14) + params_OptiX.numberOfGaussians,
								thrust::device_pointer_cast(Gaussians_indices_after_removal_host),
								thrust::device_pointer_cast(needsToBeRemoved_host),
								thrust::device_pointer_cast((float4 *)scatterBuffer)
							);
							if (needsToReallocMemory) {
								error_CUDA = cudaFree(params_OptiX.GC_SH_14);
								if (error_CUDA != cudaSuccess) throw 0;

								error_CUDA = cudaMalloc(&params_OptiX.GC_SH_14, sizeof(float4) * params_OptiX.tmpArraysGroup1Size);
								if (error_CUDA != cudaSuccess) throw 0;
							}
							cudaMemcpy(params_OptiX.GC_SH_14, scatterBuffer, sizeof(float4) * numberOfGaussiansNew, cudaMemcpyDeviceToDevice);

							// GC_SH_15
							thrust::scatter_if(
								thrust::device_pointer_cast(params_OptiX.GC_SH_15),
								thrust::device_pointer_cast(params_OptiX.GC_SH_15) + params_OptiX.numberOfGaussians,
								thrust::device_pointer_cast(Gaussians_indices_after_removal_host),
								thrust::device_pointer_cast(needsToBeRemoved_host),
								thrust::device_pointer_cast((float4 *)scatterBuffer)
							);
							if (needsToReallocMemory) {
								error_CUDA = cudaFree(params_OptiX.GC_SH_15);
								if (error_CUDA != cudaSuccess) throw 0;

								error_CUDA = cudaMalloc(&params_OptiX.GC_SH_15, sizeof(float4) * params_OptiX.tmpArraysGroup1Size);
								if (error_CUDA != cudaSuccess) throw 0;
							}
							cudaMemcpy(params_OptiX.GC_SH_15, scatterBuffer, sizeof(float4) * numberOfGaussiansNew, cudaMemcpyDeviceToDevice);

							// GC_SH_16
							thrust::scatter_if(
								thrust::device_pointer_cast(params_OptiX.GC_SH_16),
								thrust::device_pointer_cast(params_OptiX.GC_SH_16) + params_OptiX.numberOfGaussians,
								thrust::device_pointer_cast(Gaussians_indices_after_removal_host),
								thrust::device_pointer_cast(needsToBeRemoved_host),
								thrust::device_pointer_cast((float4 *)scatterBuffer)
							);
							if (needsToReallocMemory) {
								error_CUDA = cudaFree(params_OptiX.GC_SH_16);
								if (error_CUDA != cudaSuccess) throw 0;

								error_CUDA = cudaMalloc(&params_OptiX.GC_SH_16, sizeof(float4) * params_OptiX.tmpArraysGroup1Size);
								if (error_CUDA != cudaSuccess) throw 0;
							}
							cudaMemcpy(params_OptiX.GC_SH_16, scatterBuffer, sizeof(float4) * numberOfGaussiansNew, cudaMemcpyDeviceToDevice);

							// GC_SH_17
							thrust::scatter_if(
								thrust::device_pointer_cast(params_OptiX.GC_SH_17),
								thrust::device_pointer_cast(params_OptiX.GC_SH_17) + params_OptiX.numberOfGaussians,
								thrust::device_pointer_cast(Gaussians_indices_after_removal_host),
								thrust::device_pointer_cast(needsToBeRemoved_host),
								thrust::device_pointer_cast((float4 *)scatterBuffer)
							);
							if (needsToReallocMemory) {
								error_CUDA = cudaFree(params_OptiX.GC_SH_17);
								if (error_CUDA != cudaSuccess) throw 0;

								error_CUDA = cudaMalloc(&params_OptiX.GC_SH_17, sizeof(float4) * params_OptiX.tmpArraysGroup1Size);
								if (error_CUDA != cudaSuccess) throw 0;
							}
							cudaMemcpy(params_OptiX.GC_SH_17, scatterBuffer, sizeof(float4) * numberOfGaussiansNew, cudaMemcpyDeviceToDevice);

							// GC_SH_18
							thrust::scatter_if(
								thrust::device_pointer_cast(params_OptiX.GC_SH_18),
								thrust::device_pointer_cast(params_OptiX.GC_SH_18) + params_OptiX.numberOfGaussians,
								thrust::device_pointer_cast(Gaussians_indices_after_removal_host),
								thrust::device_pointer_cast(needsToBeRemoved_host),
								thrust::device_pointer_cast((float4 *)scatterBuffer)
							);
							if (needsToReallocMemory) {
								error_CUDA = cudaFree(params_OptiX.GC_SH_18);
								if (error_CUDA != cudaSuccess) throw 0;

								error_CUDA = cudaMalloc(&params_OptiX.GC_SH_18, sizeof(float4) * params_OptiX.tmpArraysGroup1Size);
								if (error_CUDA != cudaSuccess) throw 0;
							}
							cudaMemcpy(params_OptiX.GC_SH_18, scatterBuffer, sizeof(float4) * numberOfGaussiansNew, cudaMemcpyDeviceToDevice);
						} else {
							// !!! !!! !!!
							// GC_SH_12
							thrust::scatter_if(
								thrust::device_pointer_cast(params_OptiX.GC_SH_12),
								thrust::device_pointer_cast(params_OptiX.GC_SH_12) + params_OptiX.numberOfGaussians,
								thrust::device_pointer_cast(Gaussians_indices_after_removal_host),
								thrust::device_pointer_cast(needsToBeRemoved_host),
								thrust::device_pointer_cast((float *)scatterBuffer)
							);
							if (needsToReallocMemory) {
								error_CUDA = cudaFree(params_OptiX.GC_SH_12);
								if (error_CUDA != cudaSuccess) throw 0;

								error_CUDA = cudaMalloc(&params_OptiX.GC_SH_12, sizeof(float) * params_OptiX.tmpArraysGroup1Size);
								if (error_CUDA != cudaSuccess) throw 0;
							}
							cudaMemcpy(params_OptiX.GC_SH_12, scatterBuffer, sizeof(float) * numberOfGaussiansNew, cudaMemcpyDeviceToDevice);
						}
					}
				} else {
					// !!! !!! !!!
					// GC_SH_3
					thrust::scatter_if(
						thrust::device_pointer_cast(params_OptiX.GC_SH_3),
						thrust::device_pointer_cast(params_OptiX.GC_SH_3) + params_OptiX.numberOfGaussians,
						thrust::device_pointer_cast(Gaussians_indices_after_removal_host),
						thrust::device_pointer_cast(needsToBeRemoved_host),
						thrust::device_pointer_cast((float *)scatterBuffer)
					);
					if (needsToReallocMemory) {
						error_CUDA = cudaFree(params_OptiX.GC_SH_3);
						if (error_CUDA != cudaSuccess) throw 0;

						error_CUDA = cudaMalloc(&params_OptiX.GC_SH_3, sizeof(float) * params_OptiX.tmpArraysGroup1Size);
						if (error_CUDA != cudaSuccess) throw 0;
					}
					cudaMemcpy(params_OptiX.GC_SH_3, scatterBuffer, sizeof(float) * numberOfGaussiansNew, cudaMemcpyDeviceToDevice);
				}
			}

			// *** *** *** *** ***

			// m
			thrust::scatter_if(
				thrust::device_pointer_cast(params_OptiX.m11),
				thrust::device_pointer_cast(params_OptiX.m11) + params_OptiX.numberOfGaussians,
				thrust::device_pointer_cast(Gaussians_indices_after_removal_host),
				thrust::device_pointer_cast(needsToBeRemoved_host),
				thrust::device_pointer_cast((float4 *)scatterBuffer)
			);
			if (needsToReallocMemory) {
				error_CUDA = cudaFree(params_OptiX.m11);
				if (error_CUDA != cudaSuccess) throw 0;

				error_CUDA = cudaMalloc(&params_OptiX.m11, sizeof(float4) * params_OptiX.tmpArraysGroup1Size);
				if (error_CUDA != cudaSuccess) throw 0;
			}
			cudaMemcpy(params_OptiX.m11, scatterBuffer, sizeof(float4) * numberOfGaussiansNew, cudaMemcpyDeviceToDevice);

			thrust::scatter_if(
				thrust::device_pointer_cast(params_OptiX.m21),
				thrust::device_pointer_cast(params_OptiX.m21) + params_OptiX.numberOfGaussians,
				thrust::device_pointer_cast(Gaussians_indices_after_removal_host),
				thrust::device_pointer_cast(needsToBeRemoved_host),
				thrust::device_pointer_cast((float4 *)scatterBuffer)
			);
			if (needsToReallocMemory) {
				error_CUDA = cudaFree(params_OptiX.m21);
				if (error_CUDA != cudaSuccess) throw 0;

				error_CUDA = cudaMalloc(&params_OptiX.m21, sizeof(float4) * params_OptiX.tmpArraysGroup1Size);
				if (error_CUDA != cudaSuccess) throw 0;
			}
			cudaMemcpy(params_OptiX.m21, scatterBuffer, sizeof(float4) * numberOfGaussiansNew, cudaMemcpyDeviceToDevice);

			thrust::scatter_if(
				thrust::device_pointer_cast(params_OptiX.m31),
				thrust::device_pointer_cast(params_OptiX.m31) + params_OptiX.numberOfGaussians,
				thrust::device_pointer_cast(Gaussians_indices_after_removal_host),
				thrust::device_pointer_cast(needsToBeRemoved_host),
				thrust::device_pointer_cast((float4 *)scatterBuffer)
			);
			if (needsToReallocMemory) {
				error_CUDA = cudaFree(params_OptiX.m31);
				if (error_CUDA != cudaSuccess) throw 0;

				error_CUDA = cudaMalloc(&params_OptiX.m31, sizeof(float4) * params_OptiX.tmpArraysGroup1Size);
				if (error_CUDA != cudaSuccess) throw 0;
			}
			cudaMemcpy(params_OptiX.m31, scatterBuffer, sizeof(float4) * numberOfGaussiansNew, cudaMemcpyDeviceToDevice);

			thrust::scatter_if(
				thrust::device_pointer_cast(params_OptiX.m41),
				thrust::device_pointer_cast(params_OptiX.m41) + params_OptiX.numberOfGaussians,
				thrust::device_pointer_cast(Gaussians_indices_after_removal_host),
				thrust::device_pointer_cast(needsToBeRemoved_host),
				thrust::device_pointer_cast((float2 *)scatterBuffer)
			);
			if (needsToReallocMemory) {
				error_CUDA = cudaFree(params_OptiX.m41);
				if (error_CUDA != cudaSuccess) throw 0;

				error_CUDA = cudaMalloc(&params_OptiX.m41, sizeof(float2) * params_OptiX.tmpArraysGroup1Size);
				if (error_CUDA != cudaSuccess) throw 0;
			}
			cudaMemcpy(params_OptiX.m41, scatterBuffer, sizeof(float2) * numberOfGaussiansNew, cudaMemcpyDeviceToDevice);

			// Spherical harmonics
			if constexpr (SH_degree >= 1) {
				// m_SH_1
				thrust::scatter_if(
					thrust::device_pointer_cast(params_OptiX.m_SH_1),
					thrust::device_pointer_cast(params_OptiX.m_SH_1) + params_OptiX.numberOfGaussians,
					thrust::device_pointer_cast(Gaussians_indices_after_removal_host),
					thrust::device_pointer_cast(needsToBeRemoved_host),
					thrust::device_pointer_cast((float4 *)scatterBuffer)
				);
				if (needsToReallocMemory) {
					error_CUDA = cudaFree(params_OptiX.m_SH_1);
					if (error_CUDA != cudaSuccess) throw 0;

					error_CUDA = cudaMalloc(&params_OptiX.m_SH_1, sizeof(float4) * params_OptiX.tmpArraysGroup1Size);
					if (error_CUDA != cudaSuccess) throw 0;
				}
				cudaMemcpy(params_OptiX.m_SH_1, scatterBuffer, sizeof(float4) * numberOfGaussiansNew, cudaMemcpyDeviceToDevice);

				// m_SH_2
				thrust::scatter_if(
					thrust::device_pointer_cast(params_OptiX.m_SH_2),
					thrust::device_pointer_cast(params_OptiX.m_SH_2) + params_OptiX.numberOfGaussians,
					thrust::device_pointer_cast(Gaussians_indices_after_removal_host),
					thrust::device_pointer_cast(needsToBeRemoved_host),
					thrust::device_pointer_cast((float4 *)scatterBuffer)
				);
				if (needsToReallocMemory) {
					error_CUDA = cudaFree(params_OptiX.m_SH_2);
					if (error_CUDA != cudaSuccess) throw 0;

					error_CUDA = cudaMalloc(&params_OptiX.m_SH_2, sizeof(float4) * params_OptiX.tmpArraysGroup1Size);
					if (error_CUDA != cudaSuccess) throw 0;
				}
				cudaMemcpy(params_OptiX.m_SH_2, scatterBuffer, sizeof(float4) * numberOfGaussiansNew, cudaMemcpyDeviceToDevice);

				if constexpr (SH_degree >= 2) {
					// m_SH_3
					thrust::scatter_if(
						thrust::device_pointer_cast(params_OptiX.m_SH_3),
						thrust::device_pointer_cast(params_OptiX.m_SH_3) + params_OptiX.numberOfGaussians,
						thrust::device_pointer_cast(Gaussians_indices_after_removal_host),
						thrust::device_pointer_cast(needsToBeRemoved_host),
						thrust::device_pointer_cast((float4 *)scatterBuffer)
					);
					if (needsToReallocMemory) {
						error_CUDA = cudaFree(params_OptiX.m_SH_3);
						if (error_CUDA != cudaSuccess) throw 0;

						error_CUDA = cudaMalloc(&params_OptiX.m_SH_3, sizeof(float4) * params_OptiX.tmpArraysGroup1Size);
						if (error_CUDA != cudaSuccess) throw 0;
					}
					cudaMemcpy(params_OptiX.m_SH_3, scatterBuffer, sizeof(float4) * numberOfGaussiansNew, cudaMemcpyDeviceToDevice);

					// m_SH_4
					thrust::scatter_if(
						thrust::device_pointer_cast(params_OptiX.m_SH_4),
						thrust::device_pointer_cast(params_OptiX.m_SH_4) + params_OptiX.numberOfGaussians,
						thrust::device_pointer_cast(Gaussians_indices_after_removal_host),
						thrust::device_pointer_cast(needsToBeRemoved_host),
						thrust::device_pointer_cast((float4 *)scatterBuffer)
					);
					if (needsToReallocMemory) {
						error_CUDA = cudaFree(params_OptiX.m_SH_4);
						if (error_CUDA != cudaSuccess) throw 0;

						error_CUDA = cudaMalloc(&params_OptiX.m_SH_4, sizeof(float4) * params_OptiX.tmpArraysGroup1Size);
						if (error_CUDA != cudaSuccess) throw 0;
					}
					cudaMemcpy(params_OptiX.m_SH_4, scatterBuffer, sizeof(float4) * numberOfGaussiansNew, cudaMemcpyDeviceToDevice);

					// m_SH_5
					thrust::scatter_if(
						thrust::device_pointer_cast(params_OptiX.m_SH_5),
						thrust::device_pointer_cast(params_OptiX.m_SH_5) + params_OptiX.numberOfGaussians,
						thrust::device_pointer_cast(Gaussians_indices_after_removal_host),
						thrust::device_pointer_cast(needsToBeRemoved_host),
						thrust::device_pointer_cast((float4 *)scatterBuffer)
					);
					if (needsToReallocMemory) {
						error_CUDA = cudaFree(params_OptiX.m_SH_5);
						if (error_CUDA != cudaSuccess) throw 0;

						error_CUDA = cudaMalloc(&params_OptiX.m_SH_5, sizeof(float4) * params_OptiX.tmpArraysGroup1Size);
						if (error_CUDA != cudaSuccess) throw 0;
					}
					cudaMemcpy(params_OptiX.m_SH_5, scatterBuffer, sizeof(float4) * numberOfGaussiansNew, cudaMemcpyDeviceToDevice);

					// m_SH_6
					thrust::scatter_if(
						thrust::device_pointer_cast(params_OptiX.m_SH_6),
						thrust::device_pointer_cast(params_OptiX.m_SH_6) + params_OptiX.numberOfGaussians,
						thrust::device_pointer_cast(Gaussians_indices_after_removal_host),
						thrust::device_pointer_cast(needsToBeRemoved_host),
						thrust::device_pointer_cast((float4 *)scatterBuffer)
					);
					if (needsToReallocMemory) {
						error_CUDA = cudaFree(params_OptiX.m_SH_6);
						if (error_CUDA != cudaSuccess) throw 0;

						error_CUDA = cudaMalloc(&params_OptiX.m_SH_6, sizeof(float4) * params_OptiX.tmpArraysGroup1Size);
						if (error_CUDA != cudaSuccess) throw 0;
					}
					cudaMemcpy(params_OptiX.m_SH_6, scatterBuffer, sizeof(float4) * numberOfGaussiansNew, cudaMemcpyDeviceToDevice);

					if constexpr (SH_degree >= 3) {
						// m_SH_7
						thrust::scatter_if(
							thrust::device_pointer_cast(params_OptiX.m_SH_7),
							thrust::device_pointer_cast(params_OptiX.m_SH_7) + params_OptiX.numberOfGaussians,
							thrust::device_pointer_cast(Gaussians_indices_after_removal_host),
							thrust::device_pointer_cast(needsToBeRemoved_host),
							thrust::device_pointer_cast((float4 *)scatterBuffer)
						);
						if (needsToReallocMemory) {
							error_CUDA = cudaFree(params_OptiX.m_SH_7);
							if (error_CUDA != cudaSuccess) throw 0;

							error_CUDA = cudaMalloc(&params_OptiX.m_SH_7, sizeof(float4) * params_OptiX.tmpArraysGroup1Size);
							if (error_CUDA != cudaSuccess) throw 0;
						}
						cudaMemcpy(params_OptiX.m_SH_7, scatterBuffer, sizeof(float4) * numberOfGaussiansNew, cudaMemcpyDeviceToDevice);

						// m_SH_8
						thrust::scatter_if(
							thrust::device_pointer_cast(params_OptiX.m_SH_8),
							thrust::device_pointer_cast(params_OptiX.m_SH_8) + params_OptiX.numberOfGaussians,
							thrust::device_pointer_cast(Gaussians_indices_after_removal_host),
							thrust::device_pointer_cast(needsToBeRemoved_host),
							thrust::device_pointer_cast((float4 *)scatterBuffer)
						);
						if (needsToReallocMemory) {
							error_CUDA = cudaFree(params_OptiX.m_SH_8);
							if (error_CUDA != cudaSuccess) throw 0;

							error_CUDA = cudaMalloc(&params_OptiX.m_SH_8, sizeof(float4) * params_OptiX.tmpArraysGroup1Size);
							if (error_CUDA != cudaSuccess) throw 0;
						}
						cudaMemcpy(params_OptiX.m_SH_8, scatterBuffer, sizeof(float4) * numberOfGaussiansNew, cudaMemcpyDeviceToDevice);

						// m_SH_9
						thrust::scatter_if(
							thrust::device_pointer_cast(params_OptiX.m_SH_9),
							thrust::device_pointer_cast(params_OptiX.m_SH_9) + params_OptiX.numberOfGaussians,
							thrust::device_pointer_cast(Gaussians_indices_after_removal_host),
							thrust::device_pointer_cast(needsToBeRemoved_host),
							thrust::device_pointer_cast((float4 *)scatterBuffer)
						);
						if (needsToReallocMemory) {
							error_CUDA = cudaFree(params_OptiX.m_SH_9);
							if (error_CUDA != cudaSuccess) throw 0;

							error_CUDA = cudaMalloc(&params_OptiX.m_SH_9, sizeof(float4) * params_OptiX.tmpArraysGroup1Size);
							if (error_CUDA != cudaSuccess) throw 0;
						}
						cudaMemcpy(params_OptiX.m_SH_9, scatterBuffer, sizeof(float4) * numberOfGaussiansNew, cudaMemcpyDeviceToDevice);

						// m_SH_10
						thrust::scatter_if(
							thrust::device_pointer_cast(params_OptiX.m_SH_10),
							thrust::device_pointer_cast(params_OptiX.m_SH_10) + params_OptiX.numberOfGaussians,
							thrust::device_pointer_cast(Gaussians_indices_after_removal_host),
							thrust::device_pointer_cast(needsToBeRemoved_host),
							thrust::device_pointer_cast((float4 *)scatterBuffer)
						);
						if (needsToReallocMemory) {
							error_CUDA = cudaFree(params_OptiX.m_SH_10);
							if (error_CUDA != cudaSuccess) throw 0;

							error_CUDA = cudaMalloc(&params_OptiX.m_SH_10, sizeof(float4) * params_OptiX.tmpArraysGroup1Size);
							if (error_CUDA != cudaSuccess) throw 0;
						}
						cudaMemcpy(params_OptiX.m_SH_10, scatterBuffer, sizeof(float4) * numberOfGaussiansNew, cudaMemcpyDeviceToDevice);

						// m_SH_11
						thrust::scatter_if(
							thrust::device_pointer_cast(params_OptiX.m_SH_11),
							thrust::device_pointer_cast(params_OptiX.m_SH_11) + params_OptiX.numberOfGaussians,
							thrust::device_pointer_cast(Gaussians_indices_after_removal_host),
							thrust::device_pointer_cast(needsToBeRemoved_host),
							thrust::device_pointer_cast((float4 *)scatterBuffer)
						);
						if (needsToReallocMemory) {
							error_CUDA = cudaFree(params_OptiX.m_SH_11);
							if (error_CUDA != cudaSuccess) throw 0;

							error_CUDA = cudaMalloc(&params_OptiX.m_SH_11, sizeof(float4) * params_OptiX.tmpArraysGroup1Size);
							if (error_CUDA != cudaSuccess) throw 0;
						}
						cudaMemcpy(params_OptiX.m_SH_11, scatterBuffer, sizeof(float4) * numberOfGaussiansNew, cudaMemcpyDeviceToDevice);

						if constexpr (SH_degree >= 4) {
							// m_SH_12
							thrust::scatter_if(
								thrust::device_pointer_cast(params_OptiX.m_SH_12),
								thrust::device_pointer_cast(params_OptiX.m_SH_12) + params_OptiX.numberOfGaussians,
								thrust::device_pointer_cast(Gaussians_indices_after_removal_host),
								thrust::device_pointer_cast(needsToBeRemoved_host),
								thrust::device_pointer_cast((float4 *)scatterBuffer)
							);
							if (needsToReallocMemory) {
								error_CUDA = cudaFree(params_OptiX.m_SH_12);
								if (error_CUDA != cudaSuccess) throw 0;

								error_CUDA = cudaMalloc(&params_OptiX.m_SH_12, sizeof(float4) * params_OptiX.tmpArraysGroup1Size);
								if (error_CUDA != cudaSuccess) throw 0;
							}
							cudaMemcpy(params_OptiX.m_SH_12, scatterBuffer, sizeof(float4) * numberOfGaussiansNew, cudaMemcpyDeviceToDevice);

							// m_SH_13
							thrust::scatter_if(
								thrust::device_pointer_cast(params_OptiX.m_SH_13),
								thrust::device_pointer_cast(params_OptiX.m_SH_13) + params_OptiX.numberOfGaussians,
								thrust::device_pointer_cast(Gaussians_indices_after_removal_host),
								thrust::device_pointer_cast(needsToBeRemoved_host),
								thrust::device_pointer_cast((float4 *)scatterBuffer)
							);
							if (needsToReallocMemory) {
								error_CUDA = cudaFree(params_OptiX.m_SH_13);
								if (error_CUDA != cudaSuccess) throw 0;

								error_CUDA = cudaMalloc(&params_OptiX.m_SH_13, sizeof(float4) * params_OptiX.tmpArraysGroup1Size);
								if (error_CUDA != cudaSuccess) throw 0;
							}
							cudaMemcpy(params_OptiX.m_SH_13, scatterBuffer, sizeof(float4) * numberOfGaussiansNew, cudaMemcpyDeviceToDevice);

							// m_SH_14
							thrust::scatter_if(
								thrust::device_pointer_cast(params_OptiX.m_SH_14),
								thrust::device_pointer_cast(params_OptiX.m_SH_14) + params_OptiX.numberOfGaussians,
								thrust::device_pointer_cast(Gaussians_indices_after_removal_host),
								thrust::device_pointer_cast(needsToBeRemoved_host),
								thrust::device_pointer_cast((float4 *)scatterBuffer)
							);
							if (needsToReallocMemory) {
								error_CUDA = cudaFree(params_OptiX.m_SH_14);
								if (error_CUDA != cudaSuccess) throw 0;

								error_CUDA = cudaMalloc(&params_OptiX.m_SH_14, sizeof(float4) * params_OptiX.tmpArraysGroup1Size);
								if (error_CUDA != cudaSuccess) throw 0;
							}
							cudaMemcpy(params_OptiX.m_SH_14, scatterBuffer, sizeof(float4) * numberOfGaussiansNew, cudaMemcpyDeviceToDevice);

							// m_SH_15
							thrust::scatter_if(
								thrust::device_pointer_cast(params_OptiX.m_SH_15),
								thrust::device_pointer_cast(params_OptiX.m_SH_15) + params_OptiX.numberOfGaussians,
								thrust::device_pointer_cast(Gaussians_indices_after_removal_host),
								thrust::device_pointer_cast(needsToBeRemoved_host),
								thrust::device_pointer_cast((float4 *)scatterBuffer)
							);
							if (needsToReallocMemory) {
								error_CUDA = cudaFree(params_OptiX.m_SH_15);
								if (error_CUDA != cudaSuccess) throw 0;

								error_CUDA = cudaMalloc(&params_OptiX.m_SH_15, sizeof(float4) * params_OptiX.tmpArraysGroup1Size);
								if (error_CUDA != cudaSuccess) throw 0;
							}
							cudaMemcpy(params_OptiX.m_SH_15, scatterBuffer, sizeof(float4) * numberOfGaussiansNew, cudaMemcpyDeviceToDevice);

							// m_SH_16
							thrust::scatter_if(
								thrust::device_pointer_cast(params_OptiX.m_SH_16),
								thrust::device_pointer_cast(params_OptiX.m_SH_16) + params_OptiX.numberOfGaussians,
								thrust::device_pointer_cast(Gaussians_indices_after_removal_host),
								thrust::device_pointer_cast(needsToBeRemoved_host),
								thrust::device_pointer_cast((float4 *)scatterBuffer)
							);
							if (needsToReallocMemory) {
								error_CUDA = cudaFree(params_OptiX.m_SH_16);
								if (error_CUDA != cudaSuccess) throw 0;

								error_CUDA = cudaMalloc(&params_OptiX.m_SH_16, sizeof(float4) * params_OptiX.tmpArraysGroup1Size);
								if (error_CUDA != cudaSuccess) throw 0;
							}
							cudaMemcpy(params_OptiX.m_SH_16, scatterBuffer, sizeof(float4) * numberOfGaussiansNew, cudaMemcpyDeviceToDevice);

							// m_SH_17
							thrust::scatter_if(
								thrust::device_pointer_cast(params_OptiX.m_SH_17),
								thrust::device_pointer_cast(params_OptiX.m_SH_17) + params_OptiX.numberOfGaussians,
								thrust::device_pointer_cast(Gaussians_indices_after_removal_host),
								thrust::device_pointer_cast(needsToBeRemoved_host),
								thrust::device_pointer_cast((float4 *)scatterBuffer)
							);
							if (needsToReallocMemory) {
								error_CUDA = cudaFree(params_OptiX.m_SH_17);
								if (error_CUDA != cudaSuccess) throw 0;

								error_CUDA = cudaMalloc(&params_OptiX.m_SH_17, sizeof(float4) * params_OptiX.tmpArraysGroup1Size);
								if (error_CUDA != cudaSuccess) throw 0;
							}
							cudaMemcpy(params_OptiX.m_SH_17, scatterBuffer, sizeof(float4) * numberOfGaussiansNew, cudaMemcpyDeviceToDevice);

							// m_SH_18
							thrust::scatter_if(
								thrust::device_pointer_cast(params_OptiX.m_SH_18),
								thrust::device_pointer_cast(params_OptiX.m_SH_18) + params_OptiX.numberOfGaussians,
								thrust::device_pointer_cast(Gaussians_indices_after_removal_host),
								thrust::device_pointer_cast(needsToBeRemoved_host),
								thrust::device_pointer_cast((float4 *)scatterBuffer)
							);
							if (needsToReallocMemory) {
								error_CUDA = cudaFree(params_OptiX.m_SH_18);
								if (error_CUDA != cudaSuccess) throw 0;

								error_CUDA = cudaMalloc(&params_OptiX.m_SH_18, sizeof(float4) * params_OptiX.tmpArraysGroup1Size);
								if (error_CUDA != cudaSuccess) throw 0;
							}
							cudaMemcpy(params_OptiX.m_SH_18, scatterBuffer, sizeof(float4) * numberOfGaussiansNew, cudaMemcpyDeviceToDevice);
						} else {
							// !!! !!! !!!
							// m_SH_12
							thrust::scatter_if(
								thrust::device_pointer_cast(params_OptiX.m_SH_12),
								thrust::device_pointer_cast(params_OptiX.m_SH_12) + params_OptiX.numberOfGaussians,
								thrust::device_pointer_cast(Gaussians_indices_after_removal_host),
								thrust::device_pointer_cast(needsToBeRemoved_host),
								thrust::device_pointer_cast((float *)scatterBuffer)
							);
							if (needsToReallocMemory) {
								error_CUDA = cudaFree(params_OptiX.m_SH_12);
								if (error_CUDA != cudaSuccess) throw 0;

								error_CUDA = cudaMalloc(&params_OptiX.m_SH_12, sizeof(float) * params_OptiX.tmpArraysGroup1Size);
								if (error_CUDA != cudaSuccess) throw 0;
							}
							cudaMemcpy(params_OptiX.m_SH_12, scatterBuffer, sizeof(float) * numberOfGaussiansNew, cudaMemcpyDeviceToDevice);
						}
					}
				} else {
					// !!! !!! !!!
					// m_SH_3
					thrust::scatter_if(
						thrust::device_pointer_cast(params_OptiX.m_SH_3),
						thrust::device_pointer_cast(params_OptiX.m_SH_3) + params_OptiX.numberOfGaussians,
						thrust::device_pointer_cast(Gaussians_indices_after_removal_host),
						thrust::device_pointer_cast(needsToBeRemoved_host),
						thrust::device_pointer_cast((float *)scatterBuffer)
					);
					if (needsToReallocMemory) {
						error_CUDA = cudaFree(params_OptiX.m_SH_3);
						if (error_CUDA != cudaSuccess) throw 0;

						error_CUDA = cudaMalloc(&params_OptiX.m_SH_3, sizeof(float) * params_OptiX.tmpArraysGroup1Size);
						if (error_CUDA != cudaSuccess) throw 0;
					}
					cudaMemcpy(params_OptiX.m_SH_3, scatterBuffer, sizeof(float) * numberOfGaussiansNew, cudaMemcpyDeviceToDevice);
				}
			}

			// *** *** *** *** ***	   

			// v
			thrust::scatter_if(
				thrust::device_pointer_cast(params_OptiX.v11),
				thrust::device_pointer_cast(params_OptiX.v11) + params_OptiX.numberOfGaussians,
				thrust::device_pointer_cast(Gaussians_indices_after_removal_host),
				thrust::device_pointer_cast(needsToBeRemoved_host),
				thrust::device_pointer_cast((float4 *)scatterBuffer)
			);
			if (needsToReallocMemory) {
				error_CUDA = cudaFree(params_OptiX.v11);
				if (error_CUDA != cudaSuccess) throw 0;

				error_CUDA = cudaMalloc(&params_OptiX.v11, sizeof(float4) * params_OptiX.tmpArraysGroup1Size);
				if (error_CUDA != cudaSuccess) throw 0;
			}
			cudaMemcpy(params_OptiX.v11, scatterBuffer, sizeof(float4) * numberOfGaussiansNew, cudaMemcpyDeviceToDevice);

			thrust::scatter_if(
				thrust::device_pointer_cast(params_OptiX.v21),
				thrust::device_pointer_cast(params_OptiX.v21) + params_OptiX.numberOfGaussians,
				thrust::device_pointer_cast(Gaussians_indices_after_removal_host),
				thrust::device_pointer_cast(needsToBeRemoved_host),
				thrust::device_pointer_cast((float4 *)scatterBuffer)
			);
			if (needsToReallocMemory) {
				error_CUDA = cudaFree(params_OptiX.v21);
				if (error_CUDA != cudaSuccess) throw 0;

				error_CUDA = cudaMalloc(&params_OptiX.v21, sizeof(float4) * params_OptiX.tmpArraysGroup1Size);
				if (error_CUDA != cudaSuccess) throw 0;
			}
			cudaMemcpy(params_OptiX.v21, scatterBuffer, sizeof(float4) * numberOfGaussiansNew, cudaMemcpyDeviceToDevice);

			thrust::scatter_if(
				thrust::device_pointer_cast(params_OptiX.v31),
				thrust::device_pointer_cast(params_OptiX.v31) + params_OptiX.numberOfGaussians,
				thrust::device_pointer_cast(Gaussians_indices_after_removal_host),
				thrust::device_pointer_cast(needsToBeRemoved_host),
				thrust::device_pointer_cast((float4 *)scatterBuffer)
			);
			if (needsToReallocMemory) {
				error_CUDA = cudaFree(params_OptiX.v31);
				if (error_CUDA != cudaSuccess) throw 0;

				error_CUDA = cudaMalloc(&params_OptiX.v31, sizeof(float4) * params_OptiX.tmpArraysGroup1Size);
				if (error_CUDA != cudaSuccess) throw 0;
			}
			cudaMemcpy(params_OptiX.v31, scatterBuffer, sizeof(float4) * numberOfGaussiansNew, cudaMemcpyDeviceToDevice);

			thrust::scatter_if(
				thrust::device_pointer_cast(params_OptiX.v41),
				thrust::device_pointer_cast(params_OptiX.v41) + params_OptiX.numberOfGaussians,
				thrust::device_pointer_cast(Gaussians_indices_after_removal_host),
				thrust::device_pointer_cast(needsToBeRemoved_host),
				thrust::device_pointer_cast((float2 *)scatterBuffer)
			);
			if (needsToReallocMemory) {
				error_CUDA = cudaFree(params_OptiX.v41);
				if (error_CUDA != cudaSuccess) throw 0;

				error_CUDA = cudaMalloc(&params_OptiX.v41, sizeof(float2) * params_OptiX.tmpArraysGroup1Size);
				if (error_CUDA != cudaSuccess) throw 0;
			}
			cudaMemcpy(params_OptiX.v41, scatterBuffer, sizeof(float2) * numberOfGaussiansNew, cudaMemcpyDeviceToDevice);

			// Spherical harmonics
			if constexpr (SH_degree >= 1) {
				// v_SH_1
				thrust::scatter_if(
					thrust::device_pointer_cast(params_OptiX.v_SH_1),
					thrust::device_pointer_cast(params_OptiX.v_SH_1) + params_OptiX.numberOfGaussians,
					thrust::device_pointer_cast(Gaussians_indices_after_removal_host),
					thrust::device_pointer_cast(needsToBeRemoved_host),
					thrust::device_pointer_cast((float4 *)scatterBuffer)
				);
				if (needsToReallocMemory) {
					error_CUDA = cudaFree(params_OptiX.v_SH_1);
					if (error_CUDA != cudaSuccess) throw 0;

					error_CUDA = cudaMalloc(&params_OptiX.v_SH_1, sizeof(float4) * params_OptiX.tmpArraysGroup1Size);
					if (error_CUDA != cudaSuccess) throw 0;
				}
				cudaMemcpy(params_OptiX.v_SH_1, scatterBuffer, sizeof(float4) * numberOfGaussiansNew, cudaMemcpyDeviceToDevice);

				// v_SH_2
				thrust::scatter_if(
					thrust::device_pointer_cast(params_OptiX.v_SH_2),
					thrust::device_pointer_cast(params_OptiX.v_SH_2) + params_OptiX.numberOfGaussians,
					thrust::device_pointer_cast(Gaussians_indices_after_removal_host),
					thrust::device_pointer_cast(needsToBeRemoved_host),
					thrust::device_pointer_cast((float4 *)scatterBuffer)
				);
				if (needsToReallocMemory) {
					error_CUDA = cudaFree(params_OptiX.v_SH_2);
					if (error_CUDA != cudaSuccess) throw 0;

					error_CUDA = cudaMalloc(&params_OptiX.v_SH_2, sizeof(float4) * params_OptiX.tmpArraysGroup1Size);
					if (error_CUDA != cudaSuccess) throw 0;
				}
				cudaMemcpy(params_OptiX.v_SH_2, scatterBuffer, sizeof(float4) * numberOfGaussiansNew, cudaMemcpyDeviceToDevice);

				if constexpr (SH_degree >= 2) {
					// v_SH_3
					thrust::scatter_if(
						thrust::device_pointer_cast(params_OptiX.v_SH_3),
						thrust::device_pointer_cast(params_OptiX.v_SH_3) + params_OptiX.numberOfGaussians,
						thrust::device_pointer_cast(Gaussians_indices_after_removal_host),
						thrust::device_pointer_cast(needsToBeRemoved_host),
						thrust::device_pointer_cast((float4 *)scatterBuffer)
					);
					if (needsToReallocMemory) {
						error_CUDA = cudaFree(params_OptiX.v_SH_3);
						if (error_CUDA != cudaSuccess) throw 0;

						error_CUDA = cudaMalloc(&params_OptiX.v_SH_3, sizeof(float4) * params_OptiX.tmpArraysGroup1Size);
						if (error_CUDA != cudaSuccess) throw 0;
					}
					cudaMemcpy(params_OptiX.v_SH_3, scatterBuffer, sizeof(float4) * numberOfGaussiansNew, cudaMemcpyDeviceToDevice);

					// v_SH_4
					thrust::scatter_if(
						thrust::device_pointer_cast(params_OptiX.v_SH_4),
						thrust::device_pointer_cast(params_OptiX.v_SH_4) + params_OptiX.numberOfGaussians,
						thrust::device_pointer_cast(Gaussians_indices_after_removal_host),
						thrust::device_pointer_cast(needsToBeRemoved_host),
						thrust::device_pointer_cast((float4 *)scatterBuffer)
					);
					if (needsToReallocMemory) {
						error_CUDA = cudaFree(params_OptiX.v_SH_4);
						if (error_CUDA != cudaSuccess) throw 0;

						error_CUDA = cudaMalloc(&params_OptiX.v_SH_4, sizeof(float4) * params_OptiX.tmpArraysGroup1Size);
						if (error_CUDA != cudaSuccess) throw 0;
					}
					cudaMemcpy(params_OptiX.v_SH_4, scatterBuffer, sizeof(float4) * numberOfGaussiansNew, cudaMemcpyDeviceToDevice);

					// v_SH_5
					thrust::scatter_if(
						thrust::device_pointer_cast(params_OptiX.v_SH_5),
						thrust::device_pointer_cast(params_OptiX.v_SH_5) + params_OptiX.numberOfGaussians,
						thrust::device_pointer_cast(Gaussians_indices_after_removal_host),
						thrust::device_pointer_cast(needsToBeRemoved_host),
						thrust::device_pointer_cast((float4 *)scatterBuffer)
					);
					if (needsToReallocMemory) {
						error_CUDA = cudaFree(params_OptiX.v_SH_5);
						if (error_CUDA != cudaSuccess) throw 0;

						error_CUDA = cudaMalloc(&params_OptiX.v_SH_5, sizeof(float4) * params_OptiX.tmpArraysGroup1Size);
						if (error_CUDA != cudaSuccess) throw 0;
					}
					cudaMemcpy(params_OptiX.v_SH_5, scatterBuffer, sizeof(float4) * numberOfGaussiansNew, cudaMemcpyDeviceToDevice);

					// v_SH_6
					thrust::scatter_if(
						thrust::device_pointer_cast(params_OptiX.v_SH_6),
						thrust::device_pointer_cast(params_OptiX.v_SH_6) + params_OptiX.numberOfGaussians,
						thrust::device_pointer_cast(Gaussians_indices_after_removal_host),
						thrust::device_pointer_cast(needsToBeRemoved_host),
						thrust::device_pointer_cast((float4 *)scatterBuffer)
					);
					if (needsToReallocMemory) {
						error_CUDA = cudaFree(params_OptiX.v_SH_6);
						if (error_CUDA != cudaSuccess) throw 0;

						error_CUDA = cudaMalloc(&params_OptiX.v_SH_6, sizeof(float4) * params_OptiX.tmpArraysGroup1Size);
						if (error_CUDA != cudaSuccess) throw 0;
					}
					cudaMemcpy(params_OptiX.v_SH_6, scatterBuffer, sizeof(float4) * numberOfGaussiansNew, cudaMemcpyDeviceToDevice);

					if constexpr (SH_degree >= 3) {
						// v_SH_7
						thrust::scatter_if(
							thrust::device_pointer_cast(params_OptiX.v_SH_7),
							thrust::device_pointer_cast(params_OptiX.v_SH_7) + params_OptiX.numberOfGaussians,
							thrust::device_pointer_cast(Gaussians_indices_after_removal_host),
							thrust::device_pointer_cast(needsToBeRemoved_host),
							thrust::device_pointer_cast((float4 *)scatterBuffer)
						);
						if (needsToReallocMemory) {
							error_CUDA = cudaFree(params_OptiX.v_SH_7);
							if (error_CUDA != cudaSuccess) throw 0;

							error_CUDA = cudaMalloc(&params_OptiX.v_SH_7, sizeof(float4) * params_OptiX.tmpArraysGroup1Size);
							if (error_CUDA != cudaSuccess) throw 0;
						}
						cudaMemcpy(params_OptiX.v_SH_7, scatterBuffer, sizeof(float4) * numberOfGaussiansNew, cudaMemcpyDeviceToDevice);

						// v_SH_8
						thrust::scatter_if(
							thrust::device_pointer_cast(params_OptiX.v_SH_8),
							thrust::device_pointer_cast(params_OptiX.v_SH_8) + params_OptiX.numberOfGaussians,
							thrust::device_pointer_cast(Gaussians_indices_after_removal_host),
							thrust::device_pointer_cast(needsToBeRemoved_host),
							thrust::device_pointer_cast((float4 *)scatterBuffer)
						);
						if (needsToReallocMemory) {
							error_CUDA = cudaFree(params_OptiX.v_SH_8);
							if (error_CUDA != cudaSuccess) throw 0;

							error_CUDA = cudaMalloc(&params_OptiX.v_SH_8, sizeof(float4) * params_OptiX.tmpArraysGroup1Size);
							if (error_CUDA != cudaSuccess) throw 0;
						}
						cudaMemcpy(params_OptiX.v_SH_8, scatterBuffer, sizeof(float4) * numberOfGaussiansNew, cudaMemcpyDeviceToDevice);

						// v_SH_9
						thrust::scatter_if(
							thrust::device_pointer_cast(params_OptiX.v_SH_9),
							thrust::device_pointer_cast(params_OptiX.v_SH_9) + params_OptiX.numberOfGaussians,
							thrust::device_pointer_cast(Gaussians_indices_after_removal_host),
							thrust::device_pointer_cast(needsToBeRemoved_host),
							thrust::device_pointer_cast((float4 *)scatterBuffer)
						);
						if (needsToReallocMemory) {
							error_CUDA = cudaFree(params_OptiX.v_SH_9);
							if (error_CUDA != cudaSuccess) throw 0;

							error_CUDA = cudaMalloc(&params_OptiX.v_SH_9, sizeof(float4) * params_OptiX.tmpArraysGroup1Size);
							if (error_CUDA != cudaSuccess) throw 0;
						}
						cudaMemcpy(params_OptiX.v_SH_9, scatterBuffer, sizeof(float4) * numberOfGaussiansNew, cudaMemcpyDeviceToDevice);

						// v_SH_10
						thrust::scatter_if(
							thrust::device_pointer_cast(params_OptiX.v_SH_10),
							thrust::device_pointer_cast(params_OptiX.v_SH_10) + params_OptiX.numberOfGaussians,
							thrust::device_pointer_cast(Gaussians_indices_after_removal_host),
							thrust::device_pointer_cast(needsToBeRemoved_host),
							thrust::device_pointer_cast((float4 *)scatterBuffer)
						);
						if (needsToReallocMemory) {
							error_CUDA = cudaFree(params_OptiX.v_SH_10);
							if (error_CUDA != cudaSuccess) throw 0;

							error_CUDA = cudaMalloc(&params_OptiX.v_SH_10, sizeof(float4) * params_OptiX.tmpArraysGroup1Size);
							if (error_CUDA != cudaSuccess) throw 0;
						}
						cudaMemcpy(params_OptiX.v_SH_10, scatterBuffer, sizeof(float4) * numberOfGaussiansNew, cudaMemcpyDeviceToDevice);

						// v_SH_11
						thrust::scatter_if(
							thrust::device_pointer_cast(params_OptiX.v_SH_11),
							thrust::device_pointer_cast(params_OptiX.v_SH_11) + params_OptiX.numberOfGaussians,
							thrust::device_pointer_cast(Gaussians_indices_after_removal_host),
							thrust::device_pointer_cast(needsToBeRemoved_host),
							thrust::device_pointer_cast((float4 *)scatterBuffer)
						);
						if (needsToReallocMemory) {
							error_CUDA = cudaFree(params_OptiX.v_SH_11);
							if (error_CUDA != cudaSuccess) throw 0;

							error_CUDA = cudaMalloc(&params_OptiX.v_SH_11, sizeof(float4) * params_OptiX.tmpArraysGroup1Size);
							if (error_CUDA != cudaSuccess) throw 0;
						}
						cudaMemcpy(params_OptiX.v_SH_11, scatterBuffer, sizeof(float4) * numberOfGaussiansNew, cudaMemcpyDeviceToDevice);

						if constexpr (SH_degree >= 4) {
							// v_SH_12
							thrust::scatter_if(
								thrust::device_pointer_cast(params_OptiX.v_SH_12),
								thrust::device_pointer_cast(params_OptiX.v_SH_12) + params_OptiX.numberOfGaussians,
								thrust::device_pointer_cast(Gaussians_indices_after_removal_host),
								thrust::device_pointer_cast(needsToBeRemoved_host),
								thrust::device_pointer_cast((float4 *)scatterBuffer)
							);
							if (needsToReallocMemory) {
								error_CUDA = cudaFree(params_OptiX.v_SH_12);
								if (error_CUDA != cudaSuccess) throw 0;

								error_CUDA = cudaMalloc(&params_OptiX.v_SH_12, sizeof(float4) * params_OptiX.tmpArraysGroup1Size);
								if (error_CUDA != cudaSuccess) throw 0;
							}
							cudaMemcpy(params_OptiX.v_SH_12, scatterBuffer, sizeof(float4) * numberOfGaussiansNew, cudaMemcpyDeviceToDevice);

							// v_SH_13
							thrust::scatter_if(
								thrust::device_pointer_cast(params_OptiX.v_SH_13),
								thrust::device_pointer_cast(params_OptiX.v_SH_13) + params_OptiX.numberOfGaussians,
								thrust::device_pointer_cast(Gaussians_indices_after_removal_host),
								thrust::device_pointer_cast(needsToBeRemoved_host),
								thrust::device_pointer_cast((float4 *)scatterBuffer)
							);
							if (needsToReallocMemory) {
								error_CUDA = cudaFree(params_OptiX.v_SH_13);
								if (error_CUDA != cudaSuccess) throw 0;

								error_CUDA = cudaMalloc(&params_OptiX.v_SH_13, sizeof(float4) * params_OptiX.tmpArraysGroup1Size);
								if (error_CUDA != cudaSuccess) throw 0;
							}
							cudaMemcpy(params_OptiX.v_SH_13, scatterBuffer, sizeof(float4) * numberOfGaussiansNew, cudaMemcpyDeviceToDevice);

							// v_SH_14
							thrust::scatter_if(
								thrust::device_pointer_cast(params_OptiX.v_SH_14),
								thrust::device_pointer_cast(params_OptiX.v_SH_14) + params_OptiX.numberOfGaussians,
								thrust::device_pointer_cast(Gaussians_indices_after_removal_host),
								thrust::device_pointer_cast(needsToBeRemoved_host),
								thrust::device_pointer_cast((float4 *)scatterBuffer)
							);
							if (needsToReallocMemory) {
								error_CUDA = cudaFree(params_OptiX.v_SH_14);
								if (error_CUDA != cudaSuccess) throw 0;

								error_CUDA = cudaMalloc(&params_OptiX.v_SH_14, sizeof(float4) * params_OptiX.tmpArraysGroup1Size);
								if (error_CUDA != cudaSuccess) throw 0;
							}
							cudaMemcpy(params_OptiX.v_SH_14, scatterBuffer, sizeof(float4) * numberOfGaussiansNew, cudaMemcpyDeviceToDevice);

							// v_SH_15
							thrust::scatter_if(
								thrust::device_pointer_cast(params_OptiX.v_SH_15),
								thrust::device_pointer_cast(params_OptiX.v_SH_15) + params_OptiX.numberOfGaussians,
								thrust::device_pointer_cast(Gaussians_indices_after_removal_host),
								thrust::device_pointer_cast(needsToBeRemoved_host),
								thrust::device_pointer_cast((float4 *)scatterBuffer)
							);
							if (needsToReallocMemory) {
								error_CUDA = cudaFree(params_OptiX.v_SH_15);
								if (error_CUDA != cudaSuccess) throw 0;

								error_CUDA = cudaMalloc(&params_OptiX.v_SH_15, sizeof(float4) * params_OptiX.tmpArraysGroup1Size);
								if (error_CUDA != cudaSuccess) throw 0;
							}
							cudaMemcpy(params_OptiX.v_SH_15, scatterBuffer, sizeof(float4) * numberOfGaussiansNew, cudaMemcpyDeviceToDevice);

							// v_SH_16
							thrust::scatter_if(
								thrust::device_pointer_cast(params_OptiX.v_SH_16),
								thrust::device_pointer_cast(params_OptiX.v_SH_16) + params_OptiX.numberOfGaussians,
								thrust::device_pointer_cast(Gaussians_indices_after_removal_host),
								thrust::device_pointer_cast(needsToBeRemoved_host),
								thrust::device_pointer_cast((float4 *)scatterBuffer)
							);
							if (needsToReallocMemory) {
								error_CUDA = cudaFree(params_OptiX.v_SH_16);
								if (error_CUDA != cudaSuccess) throw 0;

								error_CUDA = cudaMalloc(&params_OptiX.v_SH_16, sizeof(float4) * params_OptiX.tmpArraysGroup1Size);
								if (error_CUDA != cudaSuccess) throw 0;
							}
							cudaMemcpy(params_OptiX.v_SH_16, scatterBuffer, sizeof(float4) * numberOfGaussiansNew, cudaMemcpyDeviceToDevice);

							// v_SH_17
							thrust::scatter_if(
								thrust::device_pointer_cast(params_OptiX.v_SH_17),
								thrust::device_pointer_cast(params_OptiX.v_SH_17) + params_OptiX.numberOfGaussians,
								thrust::device_pointer_cast(Gaussians_indices_after_removal_host),
								thrust::device_pointer_cast(needsToBeRemoved_host),
								thrust::device_pointer_cast((float4 *)scatterBuffer)
							);
							if (needsToReallocMemory) {
								error_CUDA = cudaFree(params_OptiX.v_SH_17);
								if (error_CUDA != cudaSuccess) throw 0;

								error_CUDA = cudaMalloc(&params_OptiX.v_SH_17, sizeof(float4) * params_OptiX.tmpArraysGroup1Size);
								if (error_CUDA != cudaSuccess) throw 0;
							}
							cudaMemcpy(params_OptiX.v_SH_17, scatterBuffer, sizeof(float4) * numberOfGaussiansNew, cudaMemcpyDeviceToDevice);

							// v_SH_18
							thrust::scatter_if(
								thrust::device_pointer_cast(params_OptiX.v_SH_18),
								thrust::device_pointer_cast(params_OptiX.v_SH_18) + params_OptiX.numberOfGaussians,
								thrust::device_pointer_cast(Gaussians_indices_after_removal_host),
								thrust::device_pointer_cast(needsToBeRemoved_host),
								thrust::device_pointer_cast((float4 *)scatterBuffer)
							);
							if (needsToReallocMemory) {
								error_CUDA = cudaFree(params_OptiX.v_SH_18);
								if (error_CUDA != cudaSuccess) throw 0;

								error_CUDA = cudaMalloc(&params_OptiX.v_SH_18, sizeof(float4) * params_OptiX.tmpArraysGroup1Size);
								if (error_CUDA != cudaSuccess) throw 0;
							}
							cudaMemcpy(params_OptiX.v_SH_18, scatterBuffer, sizeof(float4) * numberOfGaussiansNew, cudaMemcpyDeviceToDevice);
						} else {
							// !!! !!! !!!
							// v_SH_12
							thrust::scatter_if(
								thrust::device_pointer_cast(params_OptiX.v_SH_12),
								thrust::device_pointer_cast(params_OptiX.v_SH_12) + params_OptiX.numberOfGaussians,
								thrust::device_pointer_cast(Gaussians_indices_after_removal_host),
								thrust::device_pointer_cast(needsToBeRemoved_host),
								thrust::device_pointer_cast((float *)scatterBuffer)
							);
							if (needsToReallocMemory) {
								error_CUDA = cudaFree(params_OptiX.v_SH_12);
								if (error_CUDA != cudaSuccess) throw 0;

								error_CUDA = cudaMalloc(&params_OptiX.v_SH_12, sizeof(float) * params_OptiX.tmpArraysGroup1Size);
								if (error_CUDA != cudaSuccess) throw 0;
							}
							cudaMemcpy(params_OptiX.v_SH_12, scatterBuffer, sizeof(float) * numberOfGaussiansNew, cudaMemcpyDeviceToDevice);
						}
					}
				} else {
					// !!! !!! !!!
					// v_SH_3
					thrust::scatter_if(
						thrust::device_pointer_cast(params_OptiX.v_SH_3),
						thrust::device_pointer_cast(params_OptiX.v_SH_3) + params_OptiX.numberOfGaussians,
						thrust::device_pointer_cast(Gaussians_indices_after_removal_host),
						thrust::device_pointer_cast(needsToBeRemoved_host),
						thrust::device_pointer_cast((float *)scatterBuffer)
					);
					if (needsToReallocMemory) {
						error_CUDA = cudaFree(params_OptiX.v_SH_3);
						if (error_CUDA != cudaSuccess) throw 0;

						error_CUDA = cudaMalloc(&params_OptiX.v_SH_3, sizeof(float) * params_OptiX.tmpArraysGroup1Size);
						if (error_CUDA != cudaSuccess) throw 0;
					}
					cudaMemcpy(params_OptiX.v_SH_3, scatterBuffer, sizeof(float) * numberOfGaussiansNew, cudaMemcpyDeviceToDevice);
				}
			}

			// *** *** *** *** ***

			// !!! !!! !!!
			if (needsToReallocMemory) {
				error_CUDA = cudaFree(needsToBeRemoved_host);
				if (error_CUDA != cudaSuccess) throw 0;

				error_CUDA = cudaMalloc(&needsToBeRemoved_host, sizeof(int) * params_OptiX.tmpArraysGroup1Size);
				if (error_CUDA != cudaSuccess) throw 0;

				error_CUDA = cudaMemcpyToSymbol(needsToBeRemoved, &needsToBeRemoved_host, sizeof(int *));
				if (error_CUDA != cudaSuccess) throw 0;
			} else {
				if (!next_densification) {
					error_CUDA = cudaFree(Gaussians_indices_after_removal_host);
					if (error_CUDA != cudaSuccess) throw 0;

					// !!! !!! !!!
					error_CUDA = cudaFree(scatterBuffer);
					if (error_CUDA != cudaSuccess) throw 0;

					error_CUDA = cudaMalloc(&scatterBuffer, sizeof(float) * 6 * ((numberOfGaussiansNew + 31) & -32));
					if (error_CUDA != cudaSuccess) throw 0;
					// !!! !!! !!!

					error_CUDA = cudaFree(needsToBeRemoved_host);
					if (error_CUDA != cudaSuccess) throw 0;
				}
			}
			// !!! !!! !!!

			// *** *** *** *** ***

			params_OptiX.numberOfGaussians = numberOfGaussiansNew;
			// !!! !!! !!! EXPERIMENTAL !!! !!! !!!
		}

		// ************************************************************************************************

		GenerateInstances<<<(params_OptiX.numberOfGaussians + 63) >> 6, 64, ((20 * 64) + 7) << 2>>>(
			params_OptiX.GC_part_1_1, params_OptiX.GC_part_2_1, params_OptiX.GC_part_3_1, params_OptiX.GC_part_4_1,
			params_OptiX.numberOfGaussians,
			params_OptiX.GAS,
			(float *)params_OptiX.instancesBuffer
			);
		error_CUDA = cudaGetLastError();
		if (error_CUDA != cudaSuccess) throw 0;

		error_CUDA = cudaDeviceSynchronize();
		if (error_CUDA != cudaSuccess) throw 0;
		
		// ************************************************************************************************

		ComputeAABBs<<<(params_OptiX.numberOfGaussians + 63) >> 6, 64, ((6 * 64) + 3) << 2>>>(
			params_OptiX.GC_part_1_1, params_OptiX.GC_part_2_1, params_OptiX.GC_part_3_1, params_OptiX.GC_part_4_1,
			params_OptiX.numberOfGaussians,
			(float *)scatterBuffer
		);
		error_CUDA = cudaGetLastError();
		if (error_CUDA != cudaSuccess) throw 0;

		error_CUDA = cudaDeviceSynchronize();
		if (error_CUDA != cudaSuccess) throw 0;

		error_CUDA = cudaMemcpy(&params_OptiX.loss_host, params_OptiX.loss_device, sizeof(double) * 1, cudaMemcpyDeviceToHost);
		if (error_CUDA != cudaSuccess) throw 0;

		// ************************************************************************************************

		OptixAabb aabb_scene;
		OptixAabb aabb_scene_initial;

		aabb_scene_initial.minX = INFINITY;
		aabb_scene_initial.minY = INFINITY;
		aabb_scene_initial.minZ = INFINITY;

		aabb_scene_initial.maxX = -INFINITY;
		aabb_scene_initial.maxY = -INFINITY;
		aabb_scene_initial.maxZ = -INFINITY;

		try {
			aabb_scene = thrust::reduce(
				thrust::device_pointer_cast((OptixAabb *)scatterBuffer),
				thrust::device_pointer_cast((OptixAabb *)scatterBuffer) + params_OptiX.numberOfGaussians,
				aabb_scene_initial,
				SReductionOperator_OptixAabb()
			);
		} catch (...) {
			throw 0;
		}

		float dX = aabb_scene.maxX - aabb_scene.minX;
		float dY = aabb_scene.maxY - aabb_scene.minY;
		float dZ = aabb_scene.maxZ - aabb_scene.minZ;

		float scene_extent_local = sqrtf((dX * dX) + (dY * dY) + (dZ * dZ));

		error_CUDA = cudaMemcpyToSymbol(scene_extent, &scene_extent_local, sizeof(float) * 1);
		if (error_CUDA != cudaSuccess) throw 0;

		// ************************************************************************************************

		OptixAccelBuildOptions accel_options = {};
		accel_options.buildFlags = OPTIX_BUILD_FLAG_ALLOW_COMPACTION;
		accel_options.operation  = OPTIX_BUILD_OPERATION_BUILD;

		OptixBuildInput instances_input = {};
		instances_input.type                         = OPTIX_BUILD_INPUT_TYPE_INSTANCES;
		instances_input.instanceArray.instances      = (CUdeviceptr)params_OptiX.instancesBuffer;
		instances_input.instanceArray.numInstances   = params_OptiX.numberOfGaussians;
		instances_input.instanceArray.instanceStride = 0;

		// *********************************************************************************************

		OptixAccelBufferSizes blasBufferSizes;
		error_OptiX = optixAccelComputeMemoryUsage(
			params_OptiX.optixContext,
			&accel_options,
			&instances_input,
			1,
			&blasBufferSizes
		);
		if (error_OptiX != OPTIX_SUCCESS) throw 0;

		// *********************************************************************************************

		OptixAccelEmitDesc emitDesc;
		emitDesc.type   = OPTIX_PROPERTY_TYPE_COMPACTED_SIZE;
		emitDesc.result = (CUdeviceptr)params_OptiX.compactedSizeBuffer;

		if (blasBufferSizes.tempSizeInBytes > params_OptiX.tempBufferSize) {
			params_OptiX.tempBufferSize = blasBufferSizes.tempSizeInBytes;

			error_CUDA = cudaFree(params_OptiX.tempBuffer);
			if (error_CUDA != cudaSuccess) throw 0;

			error_CUDA = cudaMalloc(&params_OptiX.tempBuffer, params_OptiX.tempBufferSize);
			if (error_CUDA != cudaSuccess) throw 0;
		}

		if (blasBufferSizes.outputSizeInBytes > params_OptiX.outputBufferSize) {
			params_OptiX.outputBufferSize = blasBufferSizes.outputSizeInBytes;

			error_CUDA = cudaFree(params_OptiX.outputBuffer);
			if (error_CUDA != cudaSuccess) throw 0;

			error_CUDA = cudaMalloc(&params_OptiX.outputBuffer, params_OptiX.outputBufferSize);
			if (error_CUDA != cudaSuccess) throw 0;
		}

		// *********************************************************************************************

		error_OptiX = optixAccelBuild(
			params_OptiX.optixContext,
			0,
			&accel_options,
			&instances_input,
			1,  
			(CUdeviceptr)params_OptiX.tempBuffer,
			blasBufferSizes.tempSizeInBytes,
			(CUdeviceptr)params_OptiX.outputBuffer,
			blasBufferSizes.outputSizeInBytes,
			&params_OptiX.IAS,
			&emitDesc,
			1
		);
		if (error_OptiX != OPTIX_SUCCESS) throw 0;

		cudaDeviceSynchronize();
		error_CUDA = cudaGetLastError();
		if (error_CUDA != cudaSuccess) throw 0;

		unsigned long long compactedSize;
		error_CUDA = cudaMemcpy(&compactedSize, params_OptiX.compactedSizeBuffer, 8, cudaMemcpyDeviceToHost);
		if (error_CUDA != cudaSuccess) throw 0;

		if (compactedSize > params_OptiX.IASBufferSize) {
			error_CUDA = cudaFree(params_OptiX.IASBuffer);
			if (error_CUDA != cudaSuccess) throw 0;

			params_OptiX.IASBufferSize = compactedSize; // !!! !!! !!! 
			error_CUDA = cudaMalloc(&params_OptiX.IASBuffer, params_OptiX.IASBufferSize);
			if (error_CUDA != cudaSuccess) throw 0;
		}

		error_OptiX = optixAccelCompact(
			params_OptiX.optixContext,
			0,
			params_OptiX.IAS,
			(CUdeviceptr)params_OptiX.IASBuffer,
			compactedSize,
			&params_OptiX.IAS
		);
		if (error_OptiX != OPTIX_SUCCESS) throw 0;

		cudaDeviceSynchronize();
		error_CUDA = cudaGetLastError();
		if (error_CUDA != cudaSuccess) throw 0;

		// *********************************************************************************************

		return true;
	} catch (...) {
		return false;
	}
}

// *************************************************************************************************

template bool UpdateGradientOptiX<0>(SOptiXRenderParams<0> &params_OptiX);

// *************************************************************************************************

template bool UpdateGradientOptiX<1>(SOptiXRenderParams<1> &params_OptiX);

// *************************************************************************************************

template bool UpdateGradientOptiX<2>(SOptiXRenderParams<2> &params_OptiX);

// *************************************************************************************************

template bool UpdateGradientOptiX<3>(SOptiXRenderParams<3> &params_OptiX);

// *************************************************************************************************

template bool UpdateGradientOptiX<4>(SOptiXRenderParams<4> &params_OptiX);