#pragma once

#include <cmath>

class CVec3Df {
    public:
        float X, Y, Z;
        
        CVec3Df() : X(0.0f), Y(0.0f), Z(0.0f) {
        }
        
        CVec3Df(float X, float Y, float Z) : X(X), Y(Y), Z(Z) {
        }
        
        CVec3Df operator-() {
            return CVec3Df(-X, -Y, -Z);
        }
        
        CVec3Df operator+(const CVec3Df &v) {
            return CVec3Df(X + v.X, Y + v.Y, Z + v.Z);
        }
        
        CVec3Df operator-(const CVec3Df &v) {
            return CVec3Df(X - v.X, Y - v.Y, Z - v.Z);
        }
        
        CVec3Df operator*(float k) {
            return CVec3Df(X * k, Y * k, Z * k);
        }
        
        float operator*(const CVec3Df &v) {
            return (X * v.X) + (Y * v.Y) + (Z * v.Z);
        }
        
        void Normalize() {
            float tmp = 1.0f / sqrt((X * X) + (Y * Y) + (Z * Z));
            X *= tmp;
            Y *= tmp;
            Z *= tmp;
        }
};

CVec3Df operator*(float k, const CVec3Df &v);
CVec3Df CrossProduct(const CVec3Df &v1, const CVec3Df &v2);
CVec3Df ReflectVector(CVec3Df &N, CVec3Df &v);
bool RefractVector(CVec3Df &N, CVec3Df &v, float n1, float n2, CVec3Df &t);
CVec3Df IntersectRayPlane(CVec3Df &N, float D, CVec3Df &P, CVec3Df &v);
