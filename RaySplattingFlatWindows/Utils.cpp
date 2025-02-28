#include "CVec3Df.h"
#include "Utils.h"
#include <cmath>
#include <windows.h>

int Log2i(int x) {
	int tmp = 0;
    if (x >= 65536) { tmp += 16; x >>= 16; }
    if (x >= 256) { tmp += 8; x >>= 8; }
    if (x >= 16) { tmp += 4; x >>= 4; }
    if (x >= 4) { tmp += 2; x >>= 2; }
    if (x >= 2) ++tmp;
    return tmp;
}

//   -*-   -*-   -*-

int GetTmpArrSizeForMergeSort(int len) {
	if (len < 2) return 0;
	else {
		int log2Len = Log2i(len);
		int tmp = 1 << (log2Len - 1);
		return max(tmp, len & ((tmp << 1) - 1));
	}
}

//   -*-   -*-   -*-

int GetAllocSize(int size, int granularity) {
	return (((size - 1) / granularity) + 1) * granularity;
}

//   -*-   -*-   -*-

bool IntersectRayBox(CVec3Df &P, CVec3Df &v, SAABB &box, float &t1, float &t2) {
    float tL = (box.lB - P.X) / v.X;
    float tR = (box.rB - P.X) / v.X;
    float tU = (box.uB - P.Y) / v.Y;
    float tD = (box.dB - P.Y) / v.Y;
    float tB = (box.bB - P.Z) / v.Z;
    float tF = (box.fB - P.Z) / v.Z;
    t1 = max(max((v.X != 0.0f) ? min(tL, tR) : -INFINITY, (v.Y != 0.0f) ? min(tU, tD) : -INFINITY), (v.Z != 0.0f) ? min(tB, tF) : -INFINITY);
    t2 = min(min((v.X != 0.0f) ? max(tL, tR) : INFINITY, (v.Y != 0.0f) ? max(tU, tD) : INFINITY), (v.Z != 0.0f) ? max(tB, tF) : INFINITY);
    return ((t2 >= 0.0f) && (t1 <= t2));
}

//   -*-   -*-   -*-

float IntersectRayTri(CVec3Df &P, CVec3Df &v, STriangle &tri) {
	CVec3Df P1 = tri.P1;
	CVec3Df P2 = tri.P2;
	CVec3Df P3 = tri.P3;
	CVec3Df U(P2.X - P1.X, P2.Y - P1.Y, P2.Z - P1.Z);
	CVec3Df V(P3.X - P1.X, P3.Y - P1.Y, P3.Z - P1.Z);
	CVec3Df W(P.X - P1.X, P.Y - P1.Y, P.Z - P1.Z);
	
	CVec3Df WxU = CrossProduct(W, U);
	CVec3Df vxV = CrossProduct(v, V);
	float D = 1.0f / (vxV * U);
	float t = (WxU * V) * D;
	if (t >= 0.0f) {
        float uu = (vxV * W) * D;
        float vv = (WxU * v) * D;
        if ((uu >= 0.0f) && (vv >= 0.0f) && (uu + vv <= 1.0f)) return t;
		else
			return INFINITY;
    } else
        return INFINITY;	
}
