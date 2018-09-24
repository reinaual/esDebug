#ifndef CORE_ICCSHAPE_INTERFACE_HPP
#define CORE_ICCSHAPE_INTERFACE_HPP

#include "Vector.hpp"
#include "iccShape.hpp"
#include <queue>
#include <tuple>

class iccInterface : public iccShape {
public:
        iccInterface(Vector3d center, double radius, double smoothingRadius, Vector3d cutoff, bool useTrans, double * transMatrix, double * invMatrix);
        void splitExt(const Particle & p, std::queue<std::vector<NewParticle>> &newParticleData);
        void reduceExt(NewParticle & reducedPart);
        double calcArea(double x0, double x1);
        Vector3d center;
        double radius;
        double radiusOuter;
        double radiusOuter2;
        double smoothingRadius;
    private:
        int newParticles = 1;
};

#endif
