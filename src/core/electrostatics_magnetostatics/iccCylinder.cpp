
#include "math.h"
#include "Vector.hpp"
#include "iccShape.hpp"
#include "iccCylinder.hpp"
#include <boost/algorithm/clamp.hpp>

#include <queue>
#include <utility>

void iccCylinder::reduceExt(NewParticle & reducedPart) {
    // position is in cylindrical coordinates
    Vector3d pos = useTrans ? matrixMul(reducedPart.pos - center, invMatrix) : reducedPart.pos - center;

    // in python clipping was necessary because of numerical errors
    double phi = atan2(boost::algorithm::clamp(pos[1] / radius, -1.0, 1.0), boost::algorithm::clamp(pos[0] / radius, -1.0, 1.0));

    std::pair<Vector3d, Vector3d> result;
    result = calcCylPart(phi - reducedPart.displace[1], pos[2] - reducedPart.displace[2]);
    reducedPart.pos = (useTrans ? matrixMul(result.first, transMatrix) : result.first) + center;
    reducedPart.normal = (useTrans ? matrixMul(result.second, transMatrix) : result.second) + center;
    reducedPart.displace = reducedPart.displace * 2.;
}

void iccCylinder::splitExt(const Particle & p, std::queue<std::vector<NewParticle>> &newParticleData) {
    // position is in cylindrical coordinates
    Vector3d pos = useTrans ? matrixMul(p.r.p - center, invMatrix) : p.r.p - center;
    const double chargedensity = p.p.q / p.adapICC.area;
    const Vector3d newdisplace = p.adapICC.displace / 2.0;
    const double newArea = 4.0 * newdisplace[1] * newdisplace[2];

    // in python clipping was necessary because of numerical errors
    double phi = atan2(boost::algorithm::clamp(pos[1] / radius, -1.0, 1.0), boost::algorithm::clamp(pos[0] / radius, -1.0, 1.0));

    std::vector<NewParticle> newP(newParticles + 1);

    newP[0].parentID = p.p.identity;
    newP[0].iccTypeID = p.adapICC.iccTypeID;
    newP[0].typeID = p.p.type;
    newP[0].displace = newdisplace;
    newP[0].eps = p.adapICC.eps;
    newP[0].sigma = p.adapICC.sigma;
    newP[0].area = newArea;
    newP[0].charge = chargedensity * newArea;

    for (int i = 1; i < newParticles + 1; i++) {
        newP[i].parentID = 0;
        newP[i].iccTypeID = p.adapICC.iccTypeID;
        newP[i].typeID = p.p.type;
        newP[i].displace = newdisplace;
        newP[i].eps = p.adapICC.eps;
        newP[i].sigma = p.adapICC.sigma;
        newP[i].area = newArea;
        newP[i].charge = newP[0].charge;
    }
    std::pair<Vector3d, Vector3d> result;
    result = calcCylPart(phi + newdisplace[1], pos[2] + newdisplace[2]);
    newP[0].pos = (useTrans ? matrixMul(result.first, transMatrix) : result.first) + center;
    newP[0].normal = useTrans ? matrixMul(result.second, transMatrix) : result.second;

    result = calcCylPart(phi + newdisplace[1], pos[2] - newdisplace[2]);
    newP[1].pos = (useTrans ? matrixMul(result.first, transMatrix) : result.first) + center;
    newP[1].normal = useTrans ? matrixMul(result.second, transMatrix) : result.second;

    result = calcCylPart(phi - newdisplace[1], pos[2] + newdisplace[2]);
    newP[2].pos = (useTrans ? matrixMul(result.first, transMatrix) : result.first) + center;
    newP[2].normal = useTrans ? matrixMul(result.second, transMatrix) : result.second;

    result = calcCylPart(phi - newdisplace[1], pos[2] - newdisplace[2]);
    newP[3].pos = (useTrans ? matrixMul(result.first, transMatrix) : result.first) + center;
    newP[3].normal = useTrans ? matrixMul(result.second, transMatrix) : result.second;

    newParticleData.push(newP);
}

std::pair<Vector3d, Vector3d> iccCylinder::calcCylPart(double phi, double z) {
    Vector3d pos = {radius * cos(phi), radius * sin(phi), z};
    Vector3d norm = direction * pos;
    norm[2] = 0.0;
    return std::make_pair(pos, norm);
}

iccCylinder::iccCylinder(Vector3d center, Vector3d axis, double length, double radius, double direction, Vector3d cutoff, bool useTrans, double * transMatrix, double * invMatrix) : iccShape::iccShape(cutoff, useTrans, transMatrix, invMatrix) {
    this->center = center;
    this->axis = axis;
    this->length = length;
    this->radius = radius;
    this->direction = direction;
}
