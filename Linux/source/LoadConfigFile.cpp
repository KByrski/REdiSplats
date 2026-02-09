#include "Header.cuh"

// *** *** *** *** ***

void LoadConfigFile(const char* fName, SOptiXRenderConfig &config) {
	FILE *f;

	char buf[256];
	char tmp[256];
	int pos;

	// *********************************************************************************************

	f = fopen(fName, "rt");

	// *********************************************************************************************

	// ESSENTIAL
	fgets(buf, 256, f);
	fgets(buf, 256, f);
	fgets(buf, 256, f);

	// *********************************************************************************************

	// Data directory path
	fgets(buf, 256, f);
	pos = (strstr(buf, "<") - buf);
	--pos;
	while (buf[pos] == ' ') --pos;
	config.data_path = (char *)malloc(sizeof(char) * (pos + 2));
	strncpy(config.data_path, buf, pos + 1);
	config.data_path[pos + 1] = 0;

	// Pretrained 3DGS model path
	fgets(buf, 256, f);
	pos = (strstr(buf, "<") - buf);
	--pos;
	while (buf[pos] == ' ') --pos;
	config.pretrained_model_path = (char *)malloc(sizeof(char) * (pos + 2));
	strncpy(config.pretrained_model_path, buf, pos + 1);
	config.pretrained_model_path[pos + 1] = 0;

	// Data format
	fgets(buf, 256, f);
	pos = (strstr(buf, "<") - buf);
	--pos;
	while (buf[pos] == ' ') --pos;
	config.data_format = (char *)malloc(sizeof(char) * (pos + 2));
	strncpy(config.data_format, buf, pos + 1);
	config.data_format[pos + 1] = 0;

	// *********************************************************************************************

	fgets(buf, 256, f);

	// *********************************************************************************************

	// Start iteration
	fgets(buf, 256, f);
	sscanf(buf, "%d", &config.start_epoch);

	// End iteration
	fgets(buf, 256, f);
	sscanf(buf, "%d", &config.end_epoch);

	// *********************************************************************************************

	fgets(buf, 256, f);

	// *********************************************************************************************

	// Spherical harmonics degree
	fgets(buf, 256, f);
	sscanf(buf, "%d", &config.SH_degree);

	// *********************************************************************************************

	fgets(buf, 256, f);

	// *********************************************************************************************

	// Background color R component
	fgets(buf, 256, f);
	sscanf(buf, "%f", &config.bg_color_R);

	// Background color G component
	fgets(buf, 256, f);
	sscanf(buf, "%f", &config.bg_color_G);

	// Background color B component
	fgets(buf, 256, f);
	sscanf(buf, "%f", &config.bg_color_B);

	// *********************************************************************************************

	fgets(buf, 256, f);

	// *********************************************************************************************

	// PERFORMANCE
	fgets(buf, 256, f);
	fgets(buf, 256, f);
	fgets(buf, 256, f);

	// *********************************************************************************************

	// Number of sides of the polygon approximating each of the Gaussians
	fgets(buf, 256, f);
	sscanf(buf, "%d", &config.number_of_sides);

	// Epsilon value set as the Gaussians scale parameter on the OX axis
	fgets(buf, 256, f);
	sscanf(buf, "%f", &config.epsilon);

	// *********************************************************************************************

	fgets(buf, 256, f);

	// *********************************************************************************************

	// Chi-square squared radius for the Gaussian ellipsoid of confidence
	fgets(buf, 256, f);
	sscanf(buf, "%f", &config.chi_square_squared_radius);

	// Ray termination T threshold
	fgets(buf, 256, f);
	sscanf(buf, "%f", &config.ray_termination_T_threshold);

	// Ray termination T threshold for inference
	fgets(buf, 256, f);
	sscanf(buf, "%f", &config.ray_termination_T_threshold_inference);

	// Opacity threshold for Gauss removal
	fgets(buf, 256, f);
	sscanf(buf, "%f", &config.alpha_threshold_for_Gauss_removal);

	// Maximum number of Gaussians per ray
	fgets(buf, 256, f);
	sscanf(buf, "%d", &config.max_Gaussians_per_ray);

	// *********************************************************************************************

	fgets(buf, 256, f);

	// *********************************************************************************************

	// Densification frequency
	fgets(buf, 256, f);
	sscanf(buf, "%d", &config.densification_frequency);

	// Densification start iteration
	fgets(buf, 256, f);
	sscanf(buf, "%d", &config.densification_start_epoch);

	// Densification end iteration
	fgets(buf, 256, f);
	sscanf(buf, "%d", &config.densification_end_epoch);

	// Maximum number of Gaussians per model threshold
	fgets(buf, 256, f);
	sscanf(buf, "%d", &config.max_Gaussians_per_model);

	// *********************************************************************************************

	fgets(buf, 256, f);

	// *********************************************************************************************

	// mu gradient norm threshold for densification
	fgets(buf, 256, f);
	sscanf(buf, "%f", &config.mu_grad_norm_threshold_for_densification);

	// s gradient norm threshold for Gaussian split streategy
	fgets(buf, 256, f);
	sscanf(buf, "%f", &config.s_norm_threshold_for_split_strategy);

	// Split ratio for Gaussian split strategy
	fgets(buf, 256, f);
	sscanf(buf, "%f", &config.split_ratio);

	// *********************************************************************************************

	fgets(buf, 256, f);

	// *********************************************************************************************

	// Minimum s norm threshold for Gauss removal
	fgets(buf, 256, f);
	sscanf(buf, "%f", &config.min_s_norm_threshold_for_Gauss_removal);

	// Minimum s coefficients clipping threshold
	fgets(buf, 256, f);
	sscanf(buf, "%f", &config.min_s_coefficients_clipping_threshold);

	// Maximum s norm threshold for Gauss removal
	fgets(buf, 256, f);
	sscanf(buf, "%f", &config.max_s_norm_threshold_for_Gauss_removal);

	// *********************************************************************************************

	fgets(buf, 256, f);

	// *********************************************************************************************

	// Lambda parameter for the cost function
	fgets(buf, 256, f);
	sscanf(buf, "%f", &config.lambda);

	// *********************************************************************************************

	fgets(buf, 256, f);

	// *********************************************************************************************

	// OUTPUT
	fgets(buf, 256, f);
	fgets(buf, 256, f);
	fgets(buf, 256, f);

	// *********************************************************************************************

	// Model parameters saving frequency
	fgets(buf, 256, f);
	sscanf(buf, "%d", &config.saving_frequency);

	// Model parameters saving iteration modulo frequency
	fgets(buf, 256, f);
	sscanf(buf, "%d", &config.saving_iter);

	// *********************************************************************************************

	fgets(buf, 256, f);

	// *********************************************************************************************

	// PLY file saving frequency
	fgets(buf, 256, f);
	sscanf(buf, "%d", &config.saving_frequency_PLY);

	// PLY file saving iteration modulo frequency
	fgets(buf, 256, f);
	sscanf(buf, "%d", &config.saving_iter_PLY);

	// *********************************************************************************************

	fgets(buf, 256, f);

	// *********************************************************************************************

	// Model evaluation on startup for train data set
	fgets(buf, 256, f);
	pos = (strstr(buf, "<") - buf);
	--pos;
	while (buf[pos] == ' ') --pos;
	buf[pos + 1] = 0;
	config.evaluation_on_startup_train = (strcmp(buf, "Yes") == 0);

	// Model evaluation frequency for train data set
	fgets(buf, 256, f);
	sscanf(buf, "%d", &config.evaluation_frequency_train);

	// Model evaluation iteration for train data set modulo frequency
	fgets(buf, 256, f);
	sscanf(buf, "%d", &config.evaluation_iter_train);

	// Model evaluation on finish for train data set
	fgets(buf, 256, f);
	pos = (strstr(buf, "<") - buf);
	--pos;
	while (buf[pos] == ' ') --pos;
	buf[pos + 1] = 0;
	config.evaluation_on_finish_train = (strcmp(buf, "Yes") == 0);

	// *********************************************************************************************

	fgets(buf, 256, f);

	// *********************************************************************************************

	// Model evaluation on startup for test data set
	fgets(buf, 256, f);
	pos = (strstr(buf, "<") - buf);
	--pos;
	while (buf[pos] == ' ') --pos;
	buf[pos + 1] = 0;
	config.evaluation_on_startup_test = (strcmp(buf, "Yes") == 0);

	// Model evaluation frequency for test data set
	fgets(buf, 256, f);
	sscanf(buf, "%d", &config.evaluation_frequency_test);

	// Model evaluation iteration for test data set modulo frequency
	fgets(buf, 256, f);
	sscanf(buf, "%d", &config.evaluation_iter_test);

	// Model evaluation on finish for test data set
	fgets(buf, 256, f);
	pos = (strstr(buf, "<") - buf);
	--pos;
	while (buf[pos] == ' ') --pos;
	buf[pos + 1] = 0;
	config.evaluation_on_finish_test = (strcmp(buf, "Yes") == 0);

	// *********************************************************************************************

	fgets(buf, 256, f);

	// *********************************************************************************************

	// Model visualization on startup for train data set
	fgets(buf, 256, f);
	pos = (strstr(buf, "<") - buf);
	--pos;
	while (buf[pos] == ' ') --pos;
	buf[pos + 1] = 0;
	config.visualization_on_startup_train = (strcmp(buf, "Yes") == 0);

	// Model visualization frequency for train data set
	fgets(buf, 256, f);
	sscanf(buf, "%d", &config.visualization_frequency_train);

	// Model visualization iteration for train data set modulo frequency
	fgets(buf, 256, f);
	sscanf(buf, "%d", &config.visualization_iter_train);

	// Model visualization on finish for train data set
	fgets(buf, 256, f);
	pos = (strstr(buf, "<") - buf);
	--pos;
	while (buf[pos] == ' ') --pos;
	buf[pos + 1] = 0;
	config.visualization_on_finish_train = (strcmp(buf, "Yes") == 0);

	// *********************************************************************************************

	fgets(buf, 256, f);

	// *********************************************************************************************

	// Model visualization on startup for test data set
	fgets(buf, 256, f);
	pos = (strstr(buf, "<") - buf);
	--pos;
	while (buf[pos] == ' ') --pos;
	buf[pos + 1] = 0;
	config.visualization_on_startup_test = (strcmp(buf, "Yes") == 0);

	// Model visualization frequency for test data set
	fgets(buf, 256, f);
	sscanf(buf, "%d", &config.visualization_frequency_test);

	// Model visualization iteration for test data set modulo frequency
	fgets(buf, 256, f);
	sscanf(buf, "%d", &config.visualization_iter_test);

	// Model visualization on finish for test data set
	fgets(buf, 256, f);
	pos = (strstr(buf, "<") - buf);
	--pos;
	while (buf[pos] == ' ') --pos;
	buf[pos + 1] = 0;
	config.visualization_on_finish_test = (strcmp(buf, "Yes") == 0);

	// *********************************************************************************************

	fgets(buf, 256, f);

	// *********************************************************************************************
	
	// LEARNING RATES
	fgets(buf, 256, f);
	fgets(buf, 256, f);
	fgets(buf, 256, f);

	// *********************************************************************************************

	// Learning rate for Gaussian spherical harmonics of degree 0
	fgets(buf, 256, f);
	sscanf(buf, "%f", &config.lr_SH0);

	// Exponential decay coefficient for learning rate for Gaussian Gaussian spherical harmonics of degree 0
	fgets(buf, 256, f);
	sscanf(buf, "%f", &config.lr_SH0_exponential_decay_coefficient);

	// Final value of learning rate for Gaussian spherical harmonics of degree 0
	fgets(buf, 256, f);
	sscanf(buf, "%f", &config.lr_SH0_final);

	// *********************************************************************************************

	fgets(buf, 256, f);

	// *********************************************************************************************

	// Spherical harmonics of degree 1 activation iteration
	fgets(buf, 256, f);
	sscanf(buf, "%d", &config.SH1_activation_iter);

	// Learning rate for Gaussian spherical harmonics of degree 1
	fgets(buf, 256, f);
	sscanf(buf, "%f", &config.lr_SH1);

	// Exponential decay coefficient for learning rate for Gaussian Gaussian spherical harmonics of degree 1
	fgets(buf, 256, f);
	sscanf(buf, "%f", &config.lr_SH1_exponential_decay_coefficient);

	// Final value of learning rate for Gaussian spherical harmonics of degree 1
	fgets(buf, 256, f);
	sscanf(buf, "%f", &config.lr_SH1_final);

	// *********************************************************************************************

	fgets(buf, 256, f);

	// *********************************************************************************************

	// Spherical harmonics of degree 2 activation iteration
	fgets(buf, 256, f);
	sscanf(buf, "%d", &config.SH2_activation_iter);

	// Learning rate for Gaussian spherical harmonics of degree 2
	fgets(buf, 256, f);
	sscanf(buf, "%f", &config.lr_SH2);

	// Exponential decay coefficient for learning rate for Gaussian Gaussian spherical harmonics of degree 2
	fgets(buf, 256, f);
	sscanf(buf, "%f", &config.lr_SH2_exponential_decay_coefficient);

	// Final value of learning rate for Gaussian spherical harmonics of degree 2
	fgets(buf, 256, f);
	sscanf(buf, "%f", &config.lr_SH2_final);

	// *********************************************************************************************

	fgets(buf, 256, f);

	// *********************************************************************************************

	// Spherical harmonics of degree 3 activation iteration
	fgets(buf, 256, f);
	sscanf(buf, "%d", &config.SH3_activation_iter);

	// Learning rate for Gaussian spherical harmonics of degree 3
	fgets(buf, 256, f);
	sscanf(buf, "%f", &config.lr_SH3);

	// Exponential decay coefficient for learning rate for Gaussian Gaussian spherical harmonics of degree 3
	fgets(buf, 256, f);
	sscanf(buf, "%f", &config.lr_SH3_exponential_decay_coefficient);

	// Final value of learning rate for Gaussian spherical harmonics of degree 3
	fgets(buf, 256, f);
	sscanf(buf, "%f", &config.lr_SH3_final);

	// *********************************************************************************************

	fgets(buf, 256, f);

	// *********************************************************************************************

	// Spherical harmonics of degree 4 activation iteration
	fgets(buf, 256, f);
	sscanf(buf, "%d", &config.SH4_activation_iter);

	// Learning rate for Gaussian spherical harmonics of degree 4
	fgets(buf, 256, f);
	sscanf(buf, "%f", &config.lr_SH4);

	// Exponential decay coefficient for learning rate for Gaussian Gaussian spherical harmonics of degree 4
	fgets(buf, 256, f);
	sscanf(buf, "%f", &config.lr_SH4_exponential_decay_coefficient);

	// Final value of learning rate for Gaussian spherical harmonics of degree 4
	fgets(buf, 256, f);
	sscanf(buf, "%f", &config.lr_SH4_final);

	// *********************************************************************************************

	fgets(buf, 256, f);

	// *********************************************************************************************

	// Learning rate for Gaussian opacities
	fgets(buf, 256, f);
	sscanf(buf, "%f", &config.lr_alpha);

	// Exponential decay coefficient for learning rate for Gaussian opacities
	fgets(buf, 256, f);
	sscanf(buf, "%f", &config.lr_alpha_exponential_decay_coefficient);

	// Final value of learning rate for Gaussian opacities
	fgets(buf, 256, f);
	sscanf(buf, "%f", &config.lr_alpha_final);

	// *********************************************************************************************

	fgets(buf, 256, f);

	// *********************************************************************************************

	// Learning rate for Gaussian means
	fgets(buf, 256, f);
	sscanf(buf, "%f", &config.lr_m);

	// Exponential decay coefficient for learning rate for Gaussian means
	fgets(buf, 256, f);
	sscanf(buf, "%f", &config.lr_m_exponential_decay_coefficient);

	// Final value of learning rate for Gaussian means
	fgets(buf, 256, f);
	sscanf(buf, "%f", &config.lr_m_final);

	// *********************************************************************************************

	fgets(buf, 256, f);

	// *********************************************************************************************

	// Learning rate for Gaussian scales
	fgets(buf, 256, f);
	sscanf(buf, "%f", &config.lr_s);

	// Exponential decay coefficient for learning rate for Gaussian scales
	fgets(buf, 256, f);
	sscanf(buf, "%f", &config.lr_s_exponential_decay_coefficient);

	// Final value of learning rate for Gaussian scales
	fgets(buf, 256, f);
	sscanf(buf, "%f", &config.lr_s_final);

	// *********************************************************************************************

	fgets(buf, 256, f);

	// *********************************************************************************************

	// Learning rate for Gaussian quaternions
	fgets(buf, 256, f);
	sscanf(buf, "%f", &config.lr_q);

	// Exponential decay coefficient for learning rate for Gaussian quaternions
	fgets(buf, 256, f);
	sscanf(buf, "%f", &config.lr_q_exponential_decay_coefficient);

	// Final value of learning rate for Gaussian quaternions
	fgets(buf, 256, f);
	sscanf(buf, "%f", &config.lr_q_final);

	// *********************************************************************************************

	fgets(buf, 256, f);

	// *********************************************************************************************

	// MEMORY MANAGEMENT
	fgets(buf, 256, f);
	fgets(buf, 256, f);
	fgets(buf, 256, f);

	// *********************************************************************************************

	// Temporary arrays growth factor
	fgets(buf, 256, f);
	sscanf(buf, "%lf", &config.tmp_arrays_growth_factor);

	// *********************************************************************************************

	fclose(f);
}