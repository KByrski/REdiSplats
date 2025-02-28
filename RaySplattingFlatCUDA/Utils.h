#pragma once

#include "CVec3Df.h"

class CCamera {
    public:
        CVec3Df O, R, D, F;
};

struct STriangle {
    CVec3Df P1, P2, P3;
        
    STriangle() {
    } 
        
    STriangle(CVec3Df P1, CVec3Df P2, CVec3Df P3) : P1(P1), P2(P2), P3(P3) {
    }
};

struct SMaterial {
	float Ra, Ga, Ba;
	float Rd, Gd, Bd;
	float shininess;
	float kt;
	float n;

	SMaterial() {
	}

	SMaterial(float Ra, float Ga, float Ba, float Rd, float Gd, float Bd, float shininess, float kt, float n) :
		Ra(Ra), Ga(Ga), Ba(Ba), Rd(Rd), Gd(Gd), Bd(Bd), shininess(shininess), kt(kt), n(n)
	{
	}
};

struct CFace {
    public:
        unsigned P1, P2, P3;
		unsigned N;
        unsigned N1, N2, N3;
		int mat;
		
		CFace() {
        }
		
		CFace(unsigned P1, unsigned P2, unsigned P3, unsigned N, unsigned N1, unsigned N2, unsigned N3, int mat) : P1(P1), P2(P2), P3(P3), N(N), N1(N1), N2(N2), N3(N3), mat(mat) {
        }
};

struct SAABB {
	float lB, rB;
	float uB, dB;
	float bB, fB;
};

struct SKDTreeNode {
	char info;
	float splitPos;
    int lNode, rNode;
    int lInd, uInd;
    float lB, rB, uB, dB, bB, fB;
    int lN, rN, uN, dN, bN, fN;
};

// New data structure - !!! !!! !!!
struct SKDTreeBisNode {
	char info;
	float splitPos;
    int lNode;
	int rNode;
	int kdTreeNodeInd;
};

struct SBVHTreeNode {
    char info;
    int lNode, rNode;
    int lInd, uInd;
    SAABB aabb;
};

enum EKDTreeNodeInfo { YZ_SUBDIV_SURF, XZ_SUBDIV_SURF, XY_SUBDIV_SURF, LEAF };

enum ETriSide { LEFT_ONLY, RIGHT_ONLY, BOTH };
enum EEvtType { END, PLANAR, START };

int Log2i(int x);
int GetTmpArrSizeForMergeSort(int len);
int GetAllocSize(int size, int granularity);

bool IntersectRayBox(CVec3Df &P, CVec3Df &v, SAABB &box, float &t1, float &t2);
float IntersectRayTri(CVec3Df &P, CVec3Df &v, STriangle &tri);
