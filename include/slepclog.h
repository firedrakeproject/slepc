/*
    Defines profile/logging in SLEPc.
*/

#if !defined(__SLEPCLOG_H)
#define __SLEPCLOG_H
#include "slepc.h"  

/*
  Lists all SLEPC events for profiling.
*/

extern int EPS_SetUp, EPS_Solve, ST_SetUp, ST_Apply, ST_ApplyB, ST_ApplyNoB, EPS_Orthogonalization;

#endif
