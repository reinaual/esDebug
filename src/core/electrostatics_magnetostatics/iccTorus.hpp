#ifndef CORE_ICCSHAPE_TORUS_HPP
#define CORE_ICCSHAPE_TORUS_HPP

#include "Vector.hpp"
#include "iccShape.hpp"
#include <queue>
#include <tuple>

class iccTorus : public iccShape {
public:
        iccTorus(Vector3d center, Vector3d axis, double length, double radius, double smoothingRadius, Vector3d cutoff, bool useTrans, double * transMatrix, double * invMatrix);
        void splitExt(const Particle & p, std::queue<std::vector<NewParticle>> &newParticleData);
        void reduceExt(NewParticle & reducedPart);
        std::tuple<Vector3d, Vector3d, double> calcTorusPart(double phi, double z, Vector3d displace);
        double calcArea(double z);
        Vector3d center;
        Vector3d axis;
        double innerLengthHalf;
        double length;
        double radius;
        double radiusOuter;
        double direction;
        double smoothingRadius;
    private:
        int newParticles = 3;
};

#endif
