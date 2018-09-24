
#include "math.h"
#include "Vector.hpp"
#include "iccShape.hpp"
#include "iccInterface.hpp"

#include <queue>

void iccInterface::reduceExt(NewParticle & reducedPart) {
    // position is in cylindrical coordinates
    Vector3d pos = useTrans ? matrixMul(reducedPart.pos, invMatrix) : reducedPart.pos;

    if (reducedPart.displace[0] > 0.) {
        // horizontal displacement
        pos[0] -= reducedPart.displace[0];
        double temp = radiusOuter2 - pos[0] * pos[0];
        temp = temp > 0. ? sqrt(temp) : 0.;

        // sign
        if (pos[1] > 0.) {
            pos[1] = 0.5 * (radiusOuter + temp);
        } else {
            pos[1] = - 0.5 * (radiusOuter + temp);
        }
    } else {
        // vertical displacement
        pos[1] -= reducedPart.displace[1];
        double temp = radiusOuter2 - pos[1] * pos[1];
        temp = temp > 0. ? sqrt(temp) : 0.;

        // sign
        if (pos[0] > 0.) {
            pos[0] = 0.5 * (radiusOuter + temp);
        } else {
            pos[0] = - 0.5 * (radiusOuter + temp);
        }
    }

    reducedPart.pos = (useTrans ? matrixMul(pos, transMatrix) : pos);
    // normal stays the same
    reducedPart.displace = reducedPart.displace * 2.;
}

void iccInterface::splitExt(const Particle & p, std::queue<std::vector<NewParticle>> &newParticleData) {
    // to cylindrical coordinates with ez as axis
    Vector3d pos = useTrans ? matrixMul(p.r.p - center, invMatrix) : p.r.p - center;
    const double chargedensity = p.p.q / p.adapICC.area;
    const Vector3d newdisplace = p.adapICC.displace / 2.0;

    std::vector<NewParticle> newP(newParticles + 1);

    if (newdisplace[0] > 0.) {
        // horizontal displacement
        newP[0].pos[0] = pos[0] + newdisplace[0];
        newP[0].pos[2] = pos[2];

        double temp = radiusOuter2 - newP[0].pos[0] * newP[0].pos[0];
        temp = temp > 0. ? sqrt(temp) : 0.;

        // sign
        if (pos[1] > 0.) {
            newP[0].pos[1] = 0.5 * (radiusOuter + temp);
        } else {
            newP[0].pos[1] = - 0.5 * (radiusOuter + temp);
        }

        newP[1].pos[0] = pos[0] - newdisplace[0];
        newP[1].pos[2] = pos[2];

        temp = radiusOuter2 - newP[1].pos[0] * newP[1].pos[0];
        temp = temp > 0. ? sqrt(temp) : 0.;
        // sign
        if (pos[1] > 0.) {
            newP[1].pos[1] = 0.5 * (radiusOuter + temp);
        } else {
            newP[1].pos[1] = - 0.5 * (radiusOuter + temp);
        }

        newP[0].area = calcArea(pos[0], pos[0] + p.adapICC.displace[0]);
        newP[1].area = calcArea(pos[0] - p.adapICC.displace[0], pos[0]);
    } else {
        // vertical displacement
        newP[0].pos[1] = pos[1] + newdisplace[1];
        newP[0].pos[2] = pos[2];

        double temp = radiusOuter2 - newP[0].pos[1] * newP[0].pos[1];
        temp = temp > 0. ? sqrt(temp) : 0.;

        // sign
        if (pos[0] > 0.) {
            newP[0].pos[0] = 0.5 * (radiusOuter + temp);
        } else {
            newP[0].pos[0] = - 0.5 * (radiusOuter + temp);
        }

        // second
        newP[1].pos[1] = pos[1] - newdisplace[1];
        newP[1].pos[2] = pos[2];

        temp = radiusOuter2 - newP[1].pos[1] * newP[1].pos[1];
        temp = temp > 0. ? sqrt(temp) : 0.;

        // sign
        if (pos[0] > 0.) {
            newP[1].pos[0] = 0.5 * (radiusOuter + temp);
        } else {
            newP[1].pos[0] = - 0.5 * (radiusOuter + temp);
        }

        newP[0].area = calcArea(pos[1], pos[1] + p.adapICC.displace[1]);
        newP[1].area = calcArea(pos[1] - p.adapICC.displace[1], pos[1]);
    }

    newP[0].parentID = p.p.identity;
    newP[0].iccTypeID = p.adapICC.iccTypeID;
    newP[0].typeID = p.p.type;
    newP[0].displace = newdisplace;
    newP[0].eps = p.adapICC.eps;
    newP[0].sigma = p.adapICC.sigma;

    for (int i = 1; i < newParticles + 1; i++) {
        newP[i].parentID = 0;
        newP[i].iccTypeID = p.adapICC.iccTypeID;
        newP[i].typeID = p.p.type;
        newP[i].displace = newdisplace;
        newP[i].eps = p.adapICC.eps;
        newP[i].sigma = p.adapICC.sigma;
        newP[i].pos = newP[i].pos + center;
    }

    newP[0].normal = p.adapICC.normal;
    newP[1].normal = p.adapICC.normal;

    newParticleData.push(newP);
}

double iccInterface::calcArea(double x0, double x1) {
    if (x0 > x1) {
        std::swap(x0, x1);
    }

    double out = radiusOuter * (x1 - x0);
    double temp1 = radiusOuter2 - x0 * x0;
    double temp2 = radiusOuter2 - x1 * x1;

    if (temp1 > 0.) {
        temp1 = sqrt(temp1);
        out += 0.5 * (x0 * temp1 + radiusOuter2 * atan(x0 / temp1));
    } else {
        out += 0.25 * radiusOuter2 * PI;
    }

    if (temp2 > 0.) {
        temp2 = sqrt(temp2);
        out -= 0.5 * (x1 * temp2 + radiusOuter2 * atan(x1 / temp2));
    } else {
        out -= 0.25 * radiusOuter2 * PI;
    }
    return out;
}

iccInterface::iccInterface(Vector3d center, double radius, double smoothingRadius, Vector3d cutoff, bool useTrans, double * transMatrix, double * invMatrix) : iccShape::iccShape(cutoff, useTrans, transMatrix, invMatrix) {
    this->center = center;
    this->radius = radius;
    this->smoothingRadius = smoothingRadius;
    this->radiusOuter = radius + smoothingRadius;
    this->radiusOuter2 = radiusOuter * radiusOuter;
}
