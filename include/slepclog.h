/*
    Defines profile/logging in SLEPc.
*/

#if !defined(__SLEPCLOG_H)
#define __SLEPCLOG_H
#include "slepc.h"  
PETSC_EXTERN_CXX_BEGIN

/*
  Lists all SLEPC events for profiling.
*/

extern PetscEvent EPS_SetUp, EPS_Solve, ST_SetUp, ST_Apply, ST_ApplyB, ST_ApplyNoB, EPS_Orthogonalize, ST_InnerProduct,EPS_ReverseProjection;

PETSC_EXTERN_CXX_END
#endif
