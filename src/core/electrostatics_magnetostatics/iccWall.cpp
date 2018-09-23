
#include "Vector.hpp"
#include "iccShape.hpp"
#include "iccWall.hpp"

#include <queue>

void iccWall::reduceExt(NewParticle & reducedPart) {
    Vector3d t = {reducedPart.displace[0], reducedPart.displace[1], 0.0};
    reducedPart.pos -= t;
    reducedPart.normal = normal;
    reducedPart.displace = reducedPart.displace * 2.;
}

void iccWall::splitExt(const Particle * p, std::queue<std::vector<NewParticle>> &newParticleData) {
    // split to 4 new particles
    const Vector3d temp = matrixMul(p->r.p, invMatrix) ? useTrans : p->r.p;
    const double chargedensity = p->p.q / p->adapICC.area;
    const Vector3d newdisplace = p->adapICC.displace / 2.0;
    const double newArea = 4.0 * newdisplace[0] * newdisplace[1];

    std::vector<NewParticle> newP(newParticles + 1);

    newP[0].parentID = p->p.identity;
    newP[0].iccTypeID = p->adapICC.iccTypeID;
    newP[0].typeID = p->p.type;
    newP[0].normal = normal;
    newP[0].displace = newdisplace;
    newP[0].eps = p->adapICC.eps;
    newP[0].sigma = p->adapICC.sigma;
    newP[0].area = newArea;
    newP[0].charge = chargedensity * newArea;

    for (int i = 1; i < newParticles + 1; i++) {
        newP[i].parentID = 0;
        newP[i].iccTypeID = p->adapICC.iccTypeID;
        newP[i].typeID = p->p.type;
        newP[i].normal = normal;
        newP[i].displace = newdisplace;
        newP[i].eps = p->adapICC.eps;
        newP[i].sigma = p->adapICC.sigma;
        newP[i].area = newArea;
        newP[i].charge = newP[0].charge;
    }
    Vector3d t = {newdisplace[0], newdisplace[1], 0.0};
    newP[0].pos = matrixMul(temp + t, transMatrix) ? useTrans : temp + t;
    t = {-newdisplace[0], newdisplace[1], 0.0};
    newP[1].pos = _systemmatrixMul(temp + t, transMatrix) ? useTrans : temp + t;
    t = {newdisplace[0], -newdisplace[1], 0.0};
    newP[2].pos = _systemmatrixMul(temp + t, transMatrix) ? useTrans : temp + t;
    t = {-newdisplace[0], -newdisplace[1], 0.0};
    newP[3].pos = _systemmatrixMul(temp + t, transMatrix) ? useTrans : temp + t;

    newParticleData.push(newP);
}

iccWall::iccWall(Vector3d normal, double dist, Vector3d cutoff, bool useTrans, double * transMatrix, double * invMatrix) : iccShape::iccShape(cutoff, useTrans, transMatrix, invMatrix) {
    this->normal = normal.normalize();
    this->dist = dist;
}
