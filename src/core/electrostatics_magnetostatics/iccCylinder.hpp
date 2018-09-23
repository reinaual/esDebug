#ifndef CORE_ICCSHAPE_CYLINDER_HPP
#define CORE_ICCSHAPE_CYLINDER_HPP

#include "Vector.hpp"
#include "iccShape.hpp"
#include <queue>
#include <utility>

class iccCylinder : public iccShape {
    public:
        iccCylinder(Vector3d center, Vector3d axis, double length, double radius, double direction, Vector3d cutoff, bool useTrans, double * transMatrix, double * invMatrix);
        void splitExt(const Particle & p, std::queue<std::vector<NewParticle>> &newParticleData);
        void reduceExt(NewParticle & reducedPart);
        std::pair<Vector3d, Vector3d> calcCylPart(double phi, double z);
        Vector3d center;
        Vector3d axis;
        double length;
        double radius;
        double direction;
    private:
        int newParticles = 3;
};

#endif
