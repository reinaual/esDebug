#ifndef CORE_ICCSHAPE_WALL_HPP
#define CORE_ICCSHAPE_WALL_HPP

#include "Vector.hpp"
#include "iccShape.hpp"
#include <queue>

class iccWall : public iccShape {
    public:
        iccWall(Vector3d normal, double dist, Vector3d cutoff, bool useTrans, double * transMatrix, double * invMatrix);
        void splitExt(const Particle * p, std::queue<std::vector<NewParticle>> &newParticleData);
        void reduceExt(NewParticle & reducedPart);
    private:
        Vector3d normal;
        double dist;
        int newParticles = 3;
};

#endif
