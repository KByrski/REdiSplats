#include "Header.cuh"

// !!! !!! !!!
#include "optix_function_table_definition.h"
// !!! !!! !!!

// *** *** *** *** ***

int *needsToBeRemoved_host;
__device__ int *needsToBeRemoved;
int *Gaussians_indices_after_removal_host;
int *scatterBuffer;

// *** *** *** *** ***

__constant__ float scene_extent;

// *** *** *** *** ***

float bg_color_R_host;
float bg_color_G_host;
float bg_color_B_host;
int number_of_sides_host;
int densification_frequency_host;
int densification_start_epoch_host;
int densification_end_epoch_host;
float min_s_coefficients_clipping_threshold_host;
float max_s_coefficients_clipping_threshold_host;
float ray_termination_T_threshold_host;
float last_significant_Gauss_alpha_gradient_precision_host;
float chi_square_squared_radius_host; 
int max_Gaussians_per_ray_host;
int max_Gaussians_per_model_host;
double tmp_arrays_growth_factor_host;

__constant__ float bg_color_R;
__constant__ float bg_color_G;
__constant__ float bg_color_B;

__constant__ float lr_SH0;
__constant__ float lr_SH0_exponential_decay_coefficient;
__constant__ float lr_SH0_final;

__constant__ int   SH1_activation_iter;
__constant__ float lr_SH1;
__constant__ float lr_SH1_exponential_decay_coefficient;
__constant__ float lr_SH1_final;

__constant__ int   SH2_activation_iter;
__constant__ float lr_SH2;
__constant__ float lr_SH2_exponential_decay_coefficient;
__constant__ float lr_SH2_final;

__constant__ int   SH3_activation_iter;
__constant__ float lr_SH3;
__constant__ float lr_SH3_exponential_decay_coefficient;
__constant__ float lr_SH3_final;

__constant__ int   SH4_activation_iter;
__constant__ float lr_SH4;
__constant__ float lr_SH4_exponential_decay_coefficient;
__constant__ float lr_SH4_final;

__constant__ float lr_alpha;
__constant__ float lr_alpha_exponential_decay_coefficient;
__constant__ float lr_alpha_final;

__constant__ float lr_m;
__constant__ float lr_m_exponential_decay_coefficient;
__constant__ float lr_m_final;

__constant__ float lr_s;
__constant__ float lr_s_exponential_decay_coefficient;
__constant__ float lr_s_final;

__constant__ float lr_q;
__constant__ float lr_q_exponential_decay_coefficient;
__constant__ float lr_q_final;

__constant__ int densification_frequency;
__constant__ int densification_start_epoch;
__constant__ int densification_end_epoch;
__constant__ float alpha_threshold_for_Gauss_removal;
__constant__ float min_s_coefficients_clipping_threshold;
__constant__ float max_s_coefficients_clipping_threshold;
__constant__ float min_s_norm_threshold_for_Gauss_removal;
__constant__ float max_s_norm_threshold_for_Gauss_removal;
__constant__ float mu_grad_norm_threshold_for_densification;
__constant__ float s_norm_threshold_for_split_strategy;
__constant__ float split_ratio;
__constant__ float lambda;
__constant__ float ray_termination_T_threshold;
__constant__ float chi_square_squared_radius; 
__constant__ int max_Gaussians_per_ray;
__constant__ int max_Gaussians_per_model;