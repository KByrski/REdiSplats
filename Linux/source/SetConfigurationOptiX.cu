#include "Header.cuh"

// *************************************************************************************************

bool SetConfigurationOptiX(SOptiXRenderConfig& config_OptiX) {
	cudaMemcpyToSymbol(bg_color_R, &config_OptiX.bg_color_R, sizeof(float));
	bg_color_R_host = config_OptiX.bg_color_R;
	cudaMemcpyToSymbol(bg_color_G, &config_OptiX.bg_color_G, sizeof(float));
	bg_color_G_host = config_OptiX.bg_color_G;
	cudaMemcpyToSymbol(bg_color_B, &bg_color_B, sizeof(float));
	bg_color_B_host = config_OptiX.bg_color_B;

	number_of_sides_host = config_OptiX.number_of_sides;

	cudaMemcpyToSymbol(lr_SH0, &config_OptiX.lr_SH0, sizeof(float));
	cudaMemcpyToSymbol(lr_SH0_exponential_decay_coefficient, &config_OptiX.lr_SH0_exponential_decay_coefficient, sizeof(float));
	cudaMemcpyToSymbol(lr_SH0_final, &config_OptiX.lr_SH0_final, sizeof(float));

	cudaMemcpyToSymbol(SH1_activation_iter, &config_OptiX.SH1_activation_iter, sizeof(float));
	cudaMemcpyToSymbol(lr_SH1, &config_OptiX.lr_SH1, sizeof(float));
	cudaMemcpyToSymbol(lr_SH1_exponential_decay_coefficient, &config_OptiX.lr_SH1_exponential_decay_coefficient, sizeof(float));
	cudaMemcpyToSymbol(lr_SH1_final, &config_OptiX.lr_SH1_final, sizeof(float));

	cudaMemcpyToSymbol(SH2_activation_iter, &config_OptiX.SH2_activation_iter, sizeof(float));
	cudaMemcpyToSymbol(lr_SH2, &config_OptiX.lr_SH2, sizeof(float));
	cudaMemcpyToSymbol(lr_SH2_exponential_decay_coefficient, &config_OptiX.lr_SH2_exponential_decay_coefficient, sizeof(float));
	cudaMemcpyToSymbol(lr_SH2_final, &config_OptiX.lr_SH2_final, sizeof(float));

	cudaMemcpyToSymbol(SH3_activation_iter, &config_OptiX.SH3_activation_iter, sizeof(float));
	cudaMemcpyToSymbol(lr_SH3, &config_OptiX.lr_SH3, sizeof(float));
	cudaMemcpyToSymbol(lr_SH3_exponential_decay_coefficient, &config_OptiX.lr_SH3_exponential_decay_coefficient, sizeof(float));
	cudaMemcpyToSymbol(lr_SH3_final, &config_OptiX.lr_SH3_final, sizeof(float));

	cudaMemcpyToSymbol(SH4_activation_iter, &config_OptiX.SH4_activation_iter, sizeof(float));
	cudaMemcpyToSymbol(lr_SH4, &config_OptiX.lr_SH4, sizeof(float));
	cudaMemcpyToSymbol(lr_SH4_exponential_decay_coefficient, &config_OptiX.lr_SH4_exponential_decay_coefficient, sizeof(float));
	cudaMemcpyToSymbol(lr_SH4_final, &config_OptiX.lr_SH4_final, sizeof(float));

	cudaMemcpyToSymbol(lr_alpha, &config_OptiX.lr_alpha, sizeof(float));
	cudaMemcpyToSymbol(lr_alpha_exponential_decay_coefficient, &config_OptiX.lr_alpha_exponential_decay_coefficient, sizeof(float));
	cudaMemcpyToSymbol(lr_alpha_final, &config_OptiX.lr_alpha_final, sizeof(float));

	cudaMemcpyToSymbol(lr_m, &config_OptiX.lr_m, sizeof(float));
	cudaMemcpyToSymbol(lr_m_exponential_decay_coefficient, &config_OptiX.lr_m_exponential_decay_coefficient, sizeof(float));
	cudaMemcpyToSymbol(lr_m_final, &config_OptiX.lr_m_final, sizeof(float));

	cudaMemcpyToSymbol(lr_s, &config_OptiX.lr_s, sizeof(float));
	cudaMemcpyToSymbol(lr_s_exponential_decay_coefficient, &config_OptiX.lr_s_exponential_decay_coefficient, sizeof(float));
	cudaMemcpyToSymbol(lr_s_final, &config_OptiX.lr_s_final, sizeof(float));

	cudaMemcpyToSymbol(lr_q, &config_OptiX.lr_q, sizeof(float));
	cudaMemcpyToSymbol(lr_q_exponential_decay_coefficient, &config_OptiX.lr_q_exponential_decay_coefficient, sizeof(float));
	cudaMemcpyToSymbol(lr_q_final, &config_OptiX.lr_q_final, sizeof(float));

	cudaMemcpyToSymbol(densification_frequency, &config_OptiX.densification_frequency, sizeof(int));
	densification_frequency_host = config_OptiX.densification_frequency;

	cudaMemcpyToSymbol(densification_start_epoch, &config_OptiX.densification_start_epoch, sizeof(int));
	densification_start_epoch_host = config_OptiX.densification_start_epoch;

	cudaMemcpyToSymbol(densification_end_epoch, &config_OptiX.densification_end_epoch, sizeof(int));
	densification_end_epoch_host = config_OptiX.densification_end_epoch;

	cudaMemcpyToSymbol(alpha_threshold_for_Gauss_removal, &config_OptiX.alpha_threshold_for_Gauss_removal, sizeof(float));

	cudaMemcpyToSymbol(min_s_coefficients_clipping_threshold, &config_OptiX.min_s_coefficients_clipping_threshold, sizeof(float));
	min_s_coefficients_clipping_threshold_host = config_OptiX.min_s_coefficients_clipping_threshold;

	cudaMemcpyToSymbol(max_s_coefficients_clipping_threshold, &config_OptiX.max_s_coefficients_clipping_threshold, sizeof(float));
	max_s_coefficients_clipping_threshold_host = config_OptiX.max_s_coefficients_clipping_threshold;

	cudaMemcpyToSymbol(min_s_norm_threshold_for_Gauss_removal, &config_OptiX.min_s_norm_threshold_for_Gauss_removal, sizeof(float));
	cudaMemcpyToSymbol(max_s_norm_threshold_for_Gauss_removal, &config_OptiX.max_s_norm_threshold_for_Gauss_removal, sizeof(float));
	cudaMemcpyToSymbol(mu_grad_norm_threshold_for_densification, &config_OptiX.mu_grad_norm_threshold_for_densification, sizeof(float));
	cudaMemcpyToSymbol(s_norm_threshold_for_split_strategy, &config_OptiX.s_norm_threshold_for_split_strategy, sizeof(float));
	cudaMemcpyToSymbol(split_ratio, &config_OptiX.split_ratio, sizeof(float));
	cudaMemcpyToSymbol(lambda, &config_OptiX.lambda, sizeof(float));

	cudaMemcpyToSymbol(ray_termination_T_threshold, &config_OptiX.ray_termination_T_threshold, sizeof(float));
	ray_termination_T_threshold_host = config_OptiX.ray_termination_T_threshold;

	last_significant_Gauss_alpha_gradient_precision_host = config_OptiX.last_significant_Gauss_alpha_gradient_precision;

	cudaMemcpyToSymbol(chi_square_squared_radius, &config_OptiX.chi_square_squared_radius, sizeof(float));
	chi_square_squared_radius_host = config_OptiX.chi_square_squared_radius;

	cudaMemcpyToSymbol(max_Gaussians_per_ray, &config_OptiX.max_Gaussians_per_ray, sizeof(int));
	max_Gaussians_per_ray_host = config_OptiX.max_Gaussians_per_ray;

	cudaMemcpyToSymbol(max_Gaussians_per_model, &config_OptiX.max_Gaussians_per_model, sizeof(int));
	max_Gaussians_per_model_host = config_OptiX.max_Gaussians_per_model;

	tmp_arrays_growth_factor_host = config_OptiX.tmp_arrays_growth_factor;

	return true;
}