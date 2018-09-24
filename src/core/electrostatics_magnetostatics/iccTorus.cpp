
#include <cmath>

#include "Vector.hpp"
#include "iccShape.hpp"
#include "iccTorus.hpp"
#include <boost/algorithm/clamp.hpp>
#include "utils/constants.hpp"
#include "utils.hpp"

#include <queue>
#include <tuple>

void iccTorus::reduceExt(NewParticle & reducedPart) {
    // position is in cylindrical coordinates
    Vector3d pos = useTrans ? matrixMul(reducedPart.pos - center, invMatrix) : reducedPart.pos - center;

    double temp = smoothingRadius * smoothingRadius - pow((std::abs(pos[2]) - innerLengthHalf), 2.);
    double rad;
    if (temp > 0.) {
        rad = radiusOuter - sqrt(temp);
    } else {
        rad = radiusOuter;
    }

    double phi = atan2(pos[1] / rad, pos[0] / rad);

    std::tuple<Vector3d, Vector3d, double> result;
    result = calcTorusPart(phi - reducedPart.displace[1], pos[2] - reducedPart.displace[2], 2. * reducedPart.displace);
    reducedPart.pos = (useTrans ? matrixMul(std::get<0>(result), transMatrix) : std::get<0>(result)) + center;
    reducedPart.normal = (useTrans ? matrixMul(std::get<1>(result), transMatrix) : std::get<1>(result));
    reducedPart.displace = reducedPart.displace * 2.;
}

void iccTorus::splitExt(const Particle & p, std::queue<std::vector<NewParticle>> &newParticleData) {
    // to cylindrical coordinates with ez as axis
    const Vector3d pos = useTrans ? matrixMul(p.r.p - center, invMatrix) : p.r.p - center;
    const double chargedensity = p.p.q / p.adapICC.area;
    const Vector3d newdisplace = p.adapICC.displace / 2.0;

    double temp1 = smoothingRadius * smoothingRadius - pow((std::abs(pos[2]) - innerLengthHalf), 2.);

    double rad;
    if (temp1 > 0.) {
        rad = radiusOuter - sqrt(temp1);
    } else {
        rad = radiusOuter;
    }

    double phi = atan2(pos[1] / rad, pos[0] / rad);

    std::vector<NewParticle> newP(newParticles + 1);

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
    }

    std::tuple<Vector3d, Vector3d, double> result;

    result = calcTorusPart(phi + newdisplace[1], pos[2] + newdisplace[2], newdisplace);
    newP[0].pos = (useTrans ? matrixMul(std::get<0>(result), transMatrix) : std::get<0>(result)) + center;
    newP[0].normal = (useTrans ? matrixMul(std::get<1>(result), transMatrix) : std::get<1>(result));
    newP[0].area = std::get<2>(result);
    newP[0].charge = newP[0].area * chargedensity;

    result = calcTorusPart(phi + newdisplace[1], pos[2] - newdisplace[2], newdisplace);
    newP[1].pos = (useTrans ? matrixMul(std::get<0>(result), transMatrix) : std::get<0>(result)) + center;
    newP[1].normal = (useTrans ? matrixMul(std::get<1>(result), transMatrix) : std::get<1>(result));
    newP[1].area = std::get<2>(result);
    newP[1].charge = newP[1].area * chargedensity;

    result = calcTorusPart(phi - newdisplace[1], pos[2] + newdisplace[2], newdisplace);
    newP[2].pos = (useTrans ? matrixMul(std::get<0>(result), transMatrix) : std::get<0>(result)) + center;
    newP[2].normal = (useTrans ? matrixMul(std::get<1>(result), transMatrix) : std::get<1>(result));
    newP[2].area = std::get<2>(result);
    newP[2].charge = newP[2].area * chargedensity;

    result = calcTorusPart(phi - newdisplace[1], pos[2] - newdisplace[2], newdisplace);
    newP[3].pos = (useTrans ? matrixMul(std::get<0>(result), transMatrix) : std::get<0>(result)) + center;
    newP[3].normal = (useTrans ? matrixMul(std::get<1>(result), transMatrix) : std::get<1>(result));
    newP[3].area = std::get<2>(result);
    newP[3].charge = newP[3].area * chargedensity;

    newParticleData.push(newP);
}

std::tuple<Vector3d, Vector3d, double> iccTorus::calcTorusPart(double phi, double z, Vector3d displace) {
    double temp = smoothingRadius * smoothingRadius - pow(std::abs(z) - innerLengthHalf, 2);
    temp = temp > 0. ? sqrt(temp) : 0.;
    double radiusTorus = radiusOuter - temp;
    Vector3d newPos = {radiusTorus * cos(phi), radiusTorus * sin(phi), z};
    Vector3d normal;

    if (temp > 0.) {
        // particle on torus with finite partial derivative
        for (int i = 0; i < 3; i++) {
            normal[i] = - newPos[i];
        }
        normal[2] = Utils::sgn(z) * (std::abs(z) - innerLengthHalf) * radiusTorus / temp;
    } else {
        // infinite partial derivative
        int t = Utils::sgn(z);
        for (int i = 0; i < 3; i++) {
            normal[i] = t * axis[i];
        }
    }
    return std::make_tuple(newPos, normal, 2. * (calcArea(std::abs(z) + displace[2]) - calcArea(std::abs(z) - displace[2])) * displace[1]);
}

double iccTorus::calcArea(double z) {

    double zprime = std::abs(z) - innerLengthHalf;
    double temp = smoothingRadius * smoothingRadius - zprime * zprime;

    if (temp > 0.) {
        temp = sqrt(temp);
        return 2. * radius * std::abs(z) - 0.5 * (zprime * temp + smoothingRadius * smoothingRadius * atan(zprime / temp));
    } else {
        return 2. * radius * std::abs(z) - 0.25 * smoothingRadius * smoothingRadius * PI;
    }
}

iccTorus::iccTorus(Vector3d center, Vector3d axis, double length, double radius, double smoothingRadius, Vector3d cutoff, bool useTrans, double * transMatrix, double * invMatrix) : iccShape::iccShape(cutoff, useTrans, transMatrix, invMatrix) {
    this->center = center;
    this->axis = axis;
    this->length = length;
    this->radius = radius;
    this->smoothingRadius = smoothingRadius;
    this->innerLengthHalf = length / 2. - smoothingRadius;
    this->radiusOuter = radius + smoothingRadius;
}
