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
   The PETSc include files. 
*/
#include "petsc.h"
#include "petscvec.h"
#include "petscmat.h"

/* ========================================================================== */
/* 
   SLEPc logging/profiling information
*/
#include "slepclog.h"

PETSC_EXTERN_CXX_BEGIN
/*
    Initialization of SLEPc and other system routines
*/
EXTERN PetscErrorCode SlepcInitialize(int*,char***,char[],const char[]);
EXTERN PetscErrorCode SlepcFinalize(void);
EXTERN PetscErrorCode SlepcInitializeFortran(void);

EXTERN PetscErrorCode SlepcVecSetRandom(Vec);
EXTERN PetscErrorCode SlepcIsHermitian(Mat,PetscTruth*);
#if !defined(PETSC_USE_COMPLEX)
EXTERN PetscReal SlepcAbsEigenvalue(PetscScalar,PetscScalar);
#else
#define SlepcAbsEigenvalue(x,y) PetscAbsScalar(x)
#endif
EXTERN PetscErrorCode SlepcMatConvertSeqDense(Mat,Mat*);
EXTERN PetscErrorCode SlepcQuietErrorHandler(int,const char*,const char*,const char*,PetscErrorCode,int,const char*,void*);

PETSC_EXTERN_CXX_END
#endif

