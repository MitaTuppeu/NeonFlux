#ifndef NEONFLUX_VECTOR_MATH_H
#define NEONFLUX_VECTOR_MATH_H

#include "vector.h"

namespace neonflux {

void add(const FloatVector &a, const FloatVector &b, FloatVector &result);
void sub(const FloatVector &a, const FloatVector &b, FloatVector &result);
void mul(const FloatVector &a, const FloatVector &b, FloatVector &result);
void scalar_mul(const FloatVector &a, float scalar, FloatVector &result);

}

#endif 
