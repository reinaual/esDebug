
#include "Vector.hpp"
#include "iccShape.hpp"


Vector3d matrixMul(const Vector3d & vec, double * transMatrix) {
  Vector3d out = {0., 0., 0.};

  for (int j = 0; j < 3; j++) {
    for (int i = 0; i < 3; i++) {
      out[j] += vec[i] * transMatrix[i + j*3];
    }
  }

  return out;
}
