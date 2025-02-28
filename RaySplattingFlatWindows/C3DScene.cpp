#include "C3DScene.h"
#include "CMat4Df.h"
#include "CStringTokenizer.h"
#include "CVec3Df.h"
#include "Utils.h"
#include <stdio.h>
#include <cmath>
#include <windows.h>

C3DScene::C3DScene(int vAllocGranularity, int nAllocGranularity, int fAllocGranularity, int mAllocGranularity) {
    fBuf = (CFace *)malloc(sizeof(CFace) * fAllocGranularity); fCnt = 0; fSz = fAllocGranularity;
    nBuf = (CVec3Df *)malloc(sizeof(CVec3Df) * nAllocGranularity); nCnt = 0; nSz = nAllocGranularity;
    vBuf = (CVec3Df *)malloc(sizeof(CVec3Df) * vAllocGranularity); vCnt = 0; vSz = vAllocGranularity;
	mBuf = (SMaterial *)malloc(sizeof(SMaterial) * mAllocGranularity); mCnt = 0; mSz = mAllocGranularity;
	this->fAllocGranularity = fAllocGranularity;
	this->nAllocGranularity = nAllocGranularity;
	this->vAllocGranularity = vAllocGranularity;
	this->mAllocGranularity = mAllocGranularity;
}

C3DScene::~C3DScene() {
	free(fBuf);
	free(nBuf);
	free(vBuf);
	free(mBuf);
}
        
//   -*-   -*-   -*-

int C3DScene::AddVertex(CVec3Df v) {
	if (vCnt == vSz) {
		vSz += vAllocGranularity;
		vBuf = (CVec3Df *)realloc(vBuf, sizeof(CVec3Df) * vSz);
		if (vBuf == NULL) { printf("vBuf realloc error... .\n"); }
	}
	vBuf[vCnt] = v;
	return vCnt++;
}

//   -*-   -*-   -*-

int C3DScene::AddNormal(CVec3Df N) {
	if (nCnt == nSz) {
		nSz += nAllocGranularity;
		nBuf = (CVec3Df *)realloc(nBuf, sizeof(CVec3Df) * nSz);
		if (nBuf == NULL) { printf("nBuf realloc error... .\n"); }
	}
	nBuf[nCnt] = N;
	return nCnt++;
}

//   -*-   -*-   -*-

int C3DScene::AddFace(CFace f) {
	if (fCnt == fSz) {
		fSz += fAllocGranularity;
		fBuf = (CFace *)realloc(fBuf, sizeof(CFace) * fSz);
		if (fBuf == NULL) { printf("fBuf realloc error... .\n"); }
	}
	fBuf[fCnt] = f;
	return fCnt++;
}

//   -*-   -*-   -*-

int C3DScene::AddMaterial(SMaterial m) {
	if (mCnt == mSz) {
		mSz += mAllocGranularity;
		mBuf = (SMaterial *)realloc(mBuf, sizeof(SMaterial) * mSz);
		if (mBuf == NULL) { printf("mBuf realloc error... .\n"); }
	}
	mBuf[mCnt] = m;
	return mCnt++;
}

//   -*-   -*-   -*-

SAABB C3DScene::GetAABB(int lFInd, int uFInd) {
	SAABB res = { INFINITY, -INFINITY, INFINITY, -INFINITY, INFINITY, -INFINITY };
	for (int i = lFInd; i <= uFInd; ++i) {
		CFace f = fBuf[i];
		CVec3Df P1 = vBuf[f.P1];
		CVec3Df P2 = vBuf[f.P2];
		CVec3Df P3 = vBuf[f.P3];
		SAABB aabbF = {
			min(P1.X, min(P2.X, P3.X)), max(P1.X, max(P2.X, P3.X)),
			min(P1.Y, min(P2.Y, P3.Y)), max(P1.Y, max(P2.Y, P3.Y)),
			min(P1.Z, min(P2.Z, P3.Z)), max(P1.Z, max(P2.Z, P3.Z))
		};
		if (aabbF.lB < res.lB) res.lB = aabbF.lB;
		if (aabbF.rB > res.rB) res.rB = aabbF.rB;
		if (aabbF.uB < res.uB) res.uB = aabbF.uB;
		if (aabbF.dB > res.dB) res.dB = aabbF.dB;
		if (aabbF.bB < res.bB) res.bB = aabbF.bB;
		if (aabbF.fB > res.fB) res.fB = aabbF.fB;
	}
	return res;
}

//   -*-    -*-   -*-

bool C3DScene::LoadOBJFile(const char *name, int mat) {
	char *buf;
	char *line;
	FILE *f;
	fopen_s(&f, name, "rb+");
	float XMin = INFINITY, XMax = -INFINITY;
	float YMin = INFINITY, YMax = -INFINITY;
	float ZMin = INFINITY, ZMax = -INFINITY;
	long long int fSize;
	
	int pVCnt = vCnt, pNCnt = nCnt, pFCnt = fCnt;

	fseek(f, 0, SEEK_END);
	fSize = _ftelli64(f);
	buf = (char *)malloc(sizeof(char) * (fSize + 1));
	_fseeki64(f, 0, SEEK_SET);
	fread(buf, 1, fSize, f);
	buf[fSize] = 0;
	fclose(f);
	CStringTokenizer st1(buf);
	line = st1.NextTokenIE("\n\r");
	while (line != NULL) {
		char *token1;
		char *tokens1 = _strdup(line);
		CStringTokenizer st2(tokens1);

		token1 = st2.NextTokenIE(" ");
		if (token1 != NULL) {
			if (strcmp(token1, "v") == 0) {
				CVec3Df v;

				sscanf_s(st2.NextTokenIE(" "), "%f", &v.X);
				sscanf_s(st2.NextTokenIE(" "), "%f", &v.Y);
				sscanf_s(st2.NextTokenIE(" "), "%f", &v.Z);
				XMin = min(v.X, XMin);
				XMax = max(v.X, XMax);
				YMin = min(v.Y, YMin);
				YMax = max(v.Y, YMax);
				ZMin = min(v.Z, ZMin);
				ZMax = max(v.Z, ZMax);
				AddVertex(v);
			}
			if (strcmp(token1, "vn") == 0) {
				CVec3Df N;

				sscanf_s(st2.NextTokenIE(" "), "%f", &N.X);
				sscanf_s(st2.NextTokenIE(" "), "%f", &N.Y);
				sscanf_s(st2.NextTokenIE(" "), "%f", &N.Z);
				nBuf[AddNormal(N)].Normalize();
			}
		}
		if (strcmp(token1, "f") == 0) {
			CFace f;
			int *v = (int *)malloc(sizeof(int) * 1);
			int *tv = (int *)malloc(sizeof(int) * 1);
			int *n = (int *)malloc(sizeof(int) * 1);
			int cnt = 0;
			int sz = 1;
			token1 = st2.NextTokenIE(" ");
			while (token1 != NULL) {
				char *tokens2 = _strdup(token1);
				char *token2;
				CStringTokenizer st3(tokens2);

				if (cnt == sz) {
					sz <<= 1;
					v = (int *)realloc(v, sizeof(int) * sz);
					tv = (int *)realloc(tv, sizeof(int) * sz);
					n = (int *)realloc(n, sizeof(int) * sz);
				}
				sscanf_s(st3.NextTokenIE("/"), "%d", &v[cnt]);
				token2 = st3.NextTokenAE("/");
				if (strlen(token2) > 0) sscanf_s(token2, "%d", &tv[cnt]);
				token2 = st3.NextTokenAE("/");
				if (token2 != NULL) sscanf_s(token2, "%d", &n[cnt]);
				token1 = st2.NextTokenIE(" ");
				++cnt;
			}
			for (int i = 0; i < cnt - 2; ++i) {
				f.P1 = pVCnt + v[0] - 1; f.P2 = pVCnt + v[i + 1] - 1; f.P3 = pVCnt + v[i + 2] - 1;				
				f.N1 = pNCnt + n[0] - 1; f.N2 = pNCnt + n[i + 1] - 1; f.N3 = pNCnt + n[i + 2] - 1;
				f.mat = mat;
				AddFace(f);
			}
			free(v);
			free(tv);
			free(n);
		}
		line = st1.NextTokenIE("\n\r");
		free(tokens1);
	}
	free(buf);
	return true;
}

//   -*-   -*-   -*-

void C3DScene::ClipTriangleToBoxAndGetAABB(int ind, SAABB &aabb_in, float epsilon, SAABB &aabb_out) {
	CFace f = fBuf[ind];
	CVec3Df P1 = vBuf[f.P1];
	CVec3Df P2 = vBuf[f.P2];
	CVec3Df P3 = vBuf[f.P3];
	int pointsCnt[2];
	pointsCnt[0] = 3;
	int inInd = 0, outInd = 1;
	CVec3Df points[2][9];
	points[0][0] = P1; points[0][1] = P2; points[0][2] = P3;
	struct {
		CVec3Df N;
		float D;
	} box[6] = {
		{ CVec3Df(-1.0f, 0.0f, 0.0f), aabb_in.lB },
		{ CVec3Df(1.0f, 0.0f, 0.0f), -aabb_in.rB },
		{ CVec3Df(0.0f, -1.0f, 0.0f), aabb_in.uB },
		{ CVec3Df(0.0f, 1.0f, 0.0f), -aabb_in.dB },
		{ CVec3Df(0.0f, 0.0f, -1.0f), aabb_in.bB },
		{ CVec3Df(0.0f, 0.0f, 1.0f), -aabb_in.fB }
	};
	for (int i = 0; i < 6; ++i) {
		pointsCnt[outInd] = 0;
		P1 = points[inInd][pointsCnt[inInd] - 1];
		float D = box[i].D;
		float dotP1 = (box[i].N.X * P1.X) + (box[i].N.Y * P1.Y) + (box[i].N.Z * P1.Z) + D;
		for (int j = 0; j < pointsCnt[inInd]; ++j) {
			P2 = points[inInd][j];
			float dotP2 = (box[i].N.X * P2.X) + (box[i].N.Y * P2.Y) + (box[i].N.Z * P2.Z) + D;
			if (dotP2 <= 0.0f) {
				if (dotP1 > 0.0f) {
					float t = -dotP1 / (dotP2 - dotP1);
					CVec3Df PHit;
					if (t != -INFINITY) PHit = CVec3Df(P1.X + ((P2.X - P1.X) * t), P1.Y + ((P2.Y - P1.Y) * t), P1.Z + ((P2.Z - P1.Z) * t));
					else
						PHit = P1;
					switch (i) {
						case 0 : { PHit.X = aabb_in.lB; break; }
						case 1 : { PHit.X = aabb_in.rB; break; }
						case 2 : { PHit.Y = aabb_in.uB; break; }
						case 3 : { PHit.Y = aabb_in.dB; break; }
						case 4 : { PHit.Z = aabb_in.bB; break; }
						case 5 : { PHit.Z = aabb_in.fB; break; }
					}
					points[outInd][pointsCnt[outInd]++] = PHit;
				}
				points[outInd][pointsCnt[outInd]++] = P2;
			} else {
				if (dotP1 <= 0.0f) {
					float t = -dotP2 / (dotP1 - dotP2);
					CVec3Df PHit;
					if (t != -INFINITY) PHit = CVec3Df(P2.X + ((P1.X - P2.X) * t), P2.Y + ((P1.Y - P2.Y) * t), P2.Z + ((P1.Z - P2.Z) * t));
					else
						PHit = P2;
					switch (i) {
						case 0 : { PHit.X = aabb_in.lB; break; }
						case 1 : { PHit.X = aabb_in.rB; break; }
						case 2 : { PHit.Y = aabb_in.uB; break; }
						case 3 : { PHit.Y = aabb_in.dB; break; }
						case 4 : { PHit.Z = aabb_in.bB; break; }
						case 5 : { PHit.Z = aabb_in.fB; break; }
					}
					points[outInd][pointsCnt[outInd]++] = PHit;
				}
			}
			P1 = P2;
			dotP1 = dotP2;
		}
		int tmp = inInd; inInd = outInd; outInd = tmp;
	}
	aabb_out.lB = INFINITY; aabb_out.rB = -INFINITY;
	aabb_out.uB = INFINITY; aabb_out.dB = -INFINITY;
	aabb_out.bB = INFINITY; aabb_out.fB = -INFINITY;
	for (int i = 0; i < pointsCnt[inInd]; ++i) {
		aabb_out.lB = min(aabb_out.lB, points[inInd][i].X); aabb_out.rB = max(aabb_out.rB, points[inInd][i].X);
		aabb_out.uB = min(aabb_out.uB, points[inInd][i].Y); aabb_out.dB = max(aabb_out.dB, points[inInd][i].Y);
		aabb_out.bB = min(aabb_out.bB, points[inInd][i].Z); aabb_out.fB = max(aabb_out.fB, points[inInd][i].Z);
	}
}

//   -*-   -*-   -*-

void C3DScene::Transform(CMat4Df &m, int lVInd, int uVInd, int lNInd, int uNInd) {
    CMat4Df m4n = m.Convert2M4Normals();

	for (int i = lVInd; i <= uVInd; ++i) vBuf[i] = m * vBuf[i];
	for (int i = lNInd; i <= uNInd; ++i) {
		nBuf[i] = m4n * nBuf[i];
		nBuf[i].Normalize();
	}
}
