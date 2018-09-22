#ifndef CORE_ICCSHAPE_HPP
#define CORE_ICCSHAPE_HPP

#include "particle_data.hpp"
#include "Vector.hpp"
#include <queue>


struct NewParticle {
  int parentID;
  int iccTypeID;
  int typeID;
  Vector3d pos;
  Vector3d normal;
  Vector3d displace;
  double area;
  double eps;
  double sigma;
  double charge;
};


class iccShape {
    public:
        virtual void splitExt(const Particle * p, std::queue<std::vector<NewParticle>> &newParticleData) = 0;
        virtual void reduceExt(const Particle * p) = 0;
        Vector3d cutoff;
    protected:
        iccShape(Vector3d cutoff, bool useTrans, double * transMatrix, double * invMatrix) {
            this->useTrans = useTrans;
            this->cutoff = cutoff;
            for (int i = 0; i < 9; i++) {
                this->transMatrix[i] = transMatrix[i];
                this->invMatrix[i] = invMatrix[i];
            }
        };
        bool useTrans;
        double transMatrix[9];
        double invMatrix[9];
};

#endif
