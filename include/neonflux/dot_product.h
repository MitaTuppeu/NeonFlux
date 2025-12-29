#ifndef NEONFLUX_DOT_PRODUCT_H
#define NEONFLUX_DOT_PRODUCT_H

#include "vector.h"

namespace neonflux {

float dot_naive(const FloatVector &a, const FloatVector &b);
float dot_unrolled4(const FloatVector &a, const FloatVector &b);

} 

#endif 
