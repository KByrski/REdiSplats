#include "Header.cuh"

// *** *** *** *** ***

template<int SH_degree>
void LoadPLYFile(SOptiXRenderConfig &config, SGaussianComponent<SH_degree> **GC_ptr, int &numberOfGaussians) {
	FILE *f;

	f = fopen(config.pretrained_model_path, "rb");

	char buffer[256];
	int numberOfProperties = 0;
	do {
		fgets(buffer, 256, f);

		char *str = strstr(buffer, "element vertex");
		if (str != NULL)
			sscanf(str, "element vertex %d", &numberOfGaussians);

		str = strstr(buffer, "property");
		if (str != NULL) ++numberOfProperties;

	} while (strstr(buffer, "end_header") == NULL);

	int SH_degree_source = ((int)sqrt((numberOfProperties - 13) / 3)) - 1;

	// *** *** ***

	float *pfs = (float *)malloc(sizeof(float) * numberOfProperties);

	*GC_ptr = (SGaussianComponent<SH_degree> *)malloc(sizeof(SGaussianComponent<SH_degree>) * numberOfGaussians);
	SGaussianComponent<SH_degree> *GC = *GC_ptr;
	for (int i = 0; i < numberOfGaussians; ++i) {
		fread(pfs, sizeof(float) * numberOfProperties, 1, f);

		GC[i].mX = pfs[0];
		GC[i].mY = pfs[1];
		GC[i].mZ = pfs[2];

		// !!! !!! !!!
		GC[i].sX = log(config.epsilon);
		// !!! !!! !!!
		
		GC[i].sY = pfs[numberOfProperties - 6];
		GC[i].sZ = pfs[numberOfProperties - 5];

		// *** *** *** *** ***

		double qr = pfs[numberOfProperties - 4];
		double qi = pfs[numberOfProperties - 3];
		double qj = pfs[numberOfProperties - 2];
		double qk = pfs[numberOfProperties - 1];
		double invNorm = 1.0 / sqrt((qr * qr) + (qi * qi) + (qj * qj) + (qk * qk));
		qr = qr * invNorm;
		qi = qi * invNorm;
		qj = qj * invNorm;
		qk = qk * invNorm;

		GC[i].qr = qr;
		GC[i].qi = qi;
		GC[i].qj = qj;
		GC[i].qk = qk;

		// *** *** *** *** ***

		GC[i].R = pfs[6];
		GC[i].G = pfs[7];
		GC[i].B = pfs[8];

		if constexpr (SH_degree > 0) {
			int ind_src = 0;
			for (int j = 0; j < 3; ++j) {
				if (SH_degree <= SH_degree_source) {
					for (int k = 1; k <= SH_degree_source; ++k) {
						for (int l = 0; l < (2 * k) + 1; ++l) {
							if (k <= SH_degree) {
								int ind_dest = ((((k * k) - 1) + l) * 3) + j;
								GC[i].RGB_SH_higher_order[ind_dest] = pfs[8 + ind_src];
							}
													
							++ind_src;
						}
					}
				} else {
					for (int k = 1; k <= SH_degree; ++k) {
						for (int l = 0; l < (2 * k) + 1; ++l) {
							int ind_dest = ((((k * k) - 1) + l) * 3) + j;

							if (k <= SH_degree_source) {
								GC[i].RGB_SH_higher_order[ind_dest] = pfs[8 + ind_src];
								++ind_src;
							} else
								GC[i].RGB_SH_higher_order[ind_dest] = 0.0f; // !!! !!! !!!
						}
					}
				}
			}
		}
		
		GC[i].alpha = pfs[numberOfProperties - 7];
	}

	fclose(f);
}

// *** *** *** *** ***

template void LoadPLYFile<0>(SOptiXRenderConfig &config, SGaussianComponent<0> **GC_ptr, int &numberOfGaussians);
template void LoadPLYFile<1>(SOptiXRenderConfig &config, SGaussianComponent<1> **GC_ptr, int &numberOfGaussians);
template void LoadPLYFile<2>(SOptiXRenderConfig &config, SGaussianComponent<2> **GC_ptr, int &numberOfGaussians);
template void LoadPLYFile<3>(SOptiXRenderConfig &config, SGaussianComponent<3> **GC_ptr, int &numberOfGaussians);
template void LoadPLYFile<4>(SOptiXRenderConfig &config, SGaussianComponent<4> **GC_ptr, int &numberOfGaussians);