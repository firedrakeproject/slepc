/*
   This is the main SLEPc include file (for C and C++).  It is included
   by all other SLEPc include files, so it almost never has to be 
   specifically included.
*/
#if !defined(__SLEPC_H)
#define __SLEPC_H

/* ========================================================================== */
/* 
   Current SLEPc version number and release date
*/
#include "slepcversion.h"

/* ========================================================================== */
/* 
   SLEPc logging/profiling information
*/
#include "slepclog.h"
#include <limits.h>
#include <float.h>

/* ========================================================================== */
/* 
   The PETSc include files. 
*/
#include "petsc.h"
#include "petscvec.h"
#include "petscmat.h"

/*
    Initialization of SLEPc and other system routines
*/
extern int SlepcInitialize(int*,char***,char[],const char[]);
extern int SlepcFinalize(void);
extern int SlepcInitializeFortran(void);

extern int SlepcVecSetRandom(Vec);
extern int SlepcIsHermitian(Mat,PetscTruth*);

#endif

