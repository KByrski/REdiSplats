#pragma once

#include "CMat4Df.h"
#include "CVec3Df.h"
#include "Utils.h"

struct SLight {
	float Ox, Oy, Oz;
	float R, G, B;
};

class C3DScene {
    //friend class CKDTree;
    //friend class CBVHTree;
    
    public:
		SLight light;

        CFace *fBuf;
        CVec3Df *nBuf;
        CVec3Df *vBuf;
		SMaterial *mBuf;
        int fCnt, fSz;
        int nCnt, nSz;
        int vCnt, vSz;
		int mCnt, mSz;

		int vAllocGranularity;
		int nAllocGranularity;
		int fAllocGranularity;
		int mAllocGranularity;
        
        C3DScene(int vAllocGranularity = 1048576, int nAllocGranularity = 1048576, int fAllocGranularity = 1048576, int mAllocGranularity = 16);
		~C3DScene();
		int AddVertex(CVec3Df v);
		int AddFace(CFace f);
		int AddNormal(CVec3Df N);
		int AddMaterial(SMaterial m);

		bool LoadOBJFile(const char *name, int mat);

		SAABB GetAABB(int lFInd, int uFInd);
        void Transform(CMat4Df &m, int lVInd, int uVInd, int lFInd, int uFInd);    
        
		void ClipTriangleToBoxAndGetAABB(int ind, SAABB &aabb_in, float epsilon, SAABB &aabb_out);
};
