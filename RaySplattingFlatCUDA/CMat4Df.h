#pragma once

#include <cmath>
#include "CVec3Df.h"

class CMat4Df {
    public:
        float a11, a12, a13, a14;
        float a21, a22, a23, a24;
        float a31, a32, a33, a34;
        
        CMat4Df() {
            a11 = 0.0f; a12 = 0.0f; a13 = 0.0f; a14 = 0.0f;
            a21 = 0.0f; a22 = 0.0f; a23 = 0.0f; a24 = 0.0f;
            a31 = 0.0f; a32 = 0.0f; a33 = 0.0f; a34 = 0.0f;
        }
        
        static CMat4Df Translation(const CVec3Df &v) {
            CMat4Df m;
            
            m.a11 = 1.0f; m.a14 = v.X;
            m.a22 = 1.0f; m.a24 = v.Y;
            m.a33 = 1.0f; m.a34 = v.Z;
            return m;
        }
        
        static CMat4Df Scaling(const CVec3Df &s) {
            CMat4Df m;
            
            m.a11 = s.X; m.a22 = s.Y; m.a33 = s.Z;
            return m;
        }

		static CMat4Df OXRotation(float angle) {
			CMat4Df m;

			m.a11 = 1.0f;
			m.a22 = cos(angle);
			m.a23 = -sin(angle);
			m.a32 = sin(angle);
			m.a33 = cos(angle);
			return m;
		}

		static CMat4Df OYRotation(float angle) {
			CMat4Df m;

			m.a11 = cos(angle);
			m.a13 = -sin(angle);
			m.a22 = 1.0f;
			m.a31 = sin(angle);
			m.a33 = cos(angle);
			return m;
		}

		static CMat4Df OZRotation(float angle) {
			CMat4Df m;

			m.a11 = cos(angle);
			m.a12 = -sin(angle);
			m.a21 = sin(angle);
			m.a22 = cos(angle);
			m.a33 = 1.0f;
			return m;
		}
        
        CMat4Df operator*(const CMat4Df &m) {
            CMat4Df res;
            
            res.a11 = (a11 * m.a11) + (a12 * m.a21) + (a13 * m.a31);
            res.a12 = (a11 * m.a12) + (a12 * m.a22) + (a13 * m.a32);
            res.a13 = (a11 * m.a13) + (a12 * m.a23) + (a13 * m.a33);
            res.a14 = (a11 * m.a14) + (a12 * m.a24) + (a13 * m.a34) + a14;
            
            res.a21 = (a21 * m.a11) + (a22 * m.a21) + (a23 * m.a31);
            res.a22 = (a21 * m.a12) + (a22 * m.a22) + (a23 * m.a32);
            res.a23 = (a21 * m.a13) + (a22 * m.a23) + (a23 * m.a33);
            res.a24 = (a21 * m.a14) + (a22 * m.a24) + (a23 * m.a34) + a24;
            
            res.a31 = (a31 * m.a11) + (a32 * m.a21) + (a33 * m.a31);
            res.a32 = (a31 * m.a12) + (a32 * m.a22) + (a33 * m.a32);
            res.a33 = (a31 * m.a13) + (a32 * m.a23) + (a33 * m.a33);
            res.a34 = (a31 * m.a14) + (a32 * m.a24) + (a33 * m.a34) + a34;
            
            return res;
        }
        
        CVec3Df operator*(const CVec3Df &v) {
            return CVec3Df(
                (a11 * v.X) + (a12 * v.Y) + (a13 * v.Z) + a14,
                (a21 * v.X) + (a22 * v.Y) + (a23 * v.Z) + a24,
                (a31 * v.X) + (a32 * v.Y) + (a33 * v.Z) + a34
            );
        }

		CMat4Df Convert2M4Normals() {
            CMat4Df tmp;

            tmp.a11 = (a22 * a33) - (a23 * a32);
            tmp.a12 = (a23 * a31) - (a21 * a33);
            tmp.a13 = (a21 * a32) - (a22 * a31);
            
            tmp.a21 = (a32 * a13) - (a33 * a12);
            tmp.a22 = (a33 * a11) - (a31 * a13);
            tmp.a23 = (a31 * a12) - (a32 * a11);
            
            tmp.a31 = (a12 * a23) - (a13 * a22);
            tmp.a32 = (a13 * a21) - (a11 * a23);
            tmp.a33 = (a11 * a22) - (a12 * a21);

            return tmp;
		}
};
