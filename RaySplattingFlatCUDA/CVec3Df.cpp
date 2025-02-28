#include "CVec3Df.h"

CVec3Df operator*(float k, const CVec3Df &v) {
    return CVec3Df(v.X * k, v.Y * k, v.Z * k);
}

CVec3Df CrossProduct(const CVec3Df &v1, const CVec3Df &v2) {
    return CVec3Df(
        (v1.Y * v2.Z) - (v1.Z * v2.Y),
        (v1.Z * v2.X) - (v1.X * v2.Z),
        (v1.X * v2.Y) - (v1.Y * v2.X)
    );
}

CVec3Df ReflectVector(CVec3Df &N, CVec3Df &v) {
	return v - (N * (2.0f * (N * v)));
}

bool RefractVector(CVec3Df &N, CVec3Df &v, float n1, float n2, CVec3Df &t) {
	float tmp1 = n1 / n2;
	float tmp2 = N * v;
	float tmp3 = 1.0f - ((tmp1 * tmp1) * (1.0f - (tmp2 * tmp2)));
	if (tmp3 < 0.0f) return false;
	else {
		if (tmp2 <= 0.0f) t = (v * tmp1) - (N * ((tmp2 * tmp1) + sqrt(tmp3)));
		else
			t = (v * tmp1) - (N * ((tmp2 * tmp1) - sqrt(tmp3)));
		return true;
	}
}

CVec3Df IntersectRayPlane(CVec3Df &N, float D, CVec3Df &P, CVec3Df &v) {
    float t = (-(N * P) - D) / (N * v);
    return P + (v * t);
}

