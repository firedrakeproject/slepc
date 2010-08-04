/*
   This is the main SLEPc include file (for C and C++).  It is included
   by all other SLEPc include files, so it almost never has to be 
   specifically included.

   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
   SLEPc - Scalable Library for Eigenvalue Problem Computations
   Copyright (c) 2002-2010, Universidad Politecnica de Valencia, Spain

   This file is part of SLEPc.
      
   SLEPc is free software: you can redistribute it and/or modify it under  the
   terms of version 3 of the GNU Lesser General Public License as published by
   the Free Software Foundation.

   SLEPc  is  distributed in the hope that it will be useful, but WITHOUT  ANY 
   WARRANTY;  without even the implied warranty of MERCHANTABILITY or  FITNESS 
   FOR  A  PARTICULAR PURPOSE. See the GNU Lesser General Public  License  for 
   more details.

   You  should have received a copy of the GNU Lesser General  Public  License
   along with SLEPc. If not, see <http://www.gnu.org/licenses/>.
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
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

PETSC_EXTERN_CXX_BEGIN
/*
    Initialization of SLEPc and other system routines
*/
EXTERN PetscErrorCode SlepcInitialize(int*,char***,char[],const char[]);
EXTERN PetscErrorCode SlepcFinalize(void);
EXTERN PetscErrorCode SlepcInitializeFortran(void);

EXTERN PetscErrorCode SlepcVecSetRandom(Vec,PetscRandom);
EXTERN PetscErrorCode SlepcIsHermitian(Mat,PetscTruth*);
#if !defined(PETSC_USE_COMPLEX)
EXTERN PetscReal SlepcAbsEigenvalue(PetscScalar,PetscScalar);
#else
#define SlepcAbsEigenvalue(x,y) PetscAbsScalar(x)
#endif
EXTERN PetscErrorCode SlepcVecNormalize(Vec,Vec,PetscTruth,PetscReal*);
EXTERN PetscErrorCode SlepcMatConvertSeqDense(Mat,Mat*);
EXTERN PetscErrorCode SlepcCheckOrthogonality(Vec*,PetscInt,Vec *,PetscInt,Mat,PetscScalar*);
EXTERN PetscErrorCode SlepcUpdateVectors(PetscInt,Vec*,PetscInt,PetscInt,const PetscScalar*,PetscInt,PetscTruth);
EXTERN PetscErrorCode SlepcUpdateStrideVectors(PetscInt n_,Vec *V,PetscInt s,PetscInt d,PetscInt e,const PetscScalar *Q,PetscInt ldq_,PetscTruth qtrans);
EXTERN PetscErrorCode SlepcVecMAXPBY(Vec,PetscScalar,PetscScalar,PetscInt,PetscScalar*,Vec*);
 
extern PetscTruth SlepcInitializeCalled;

PETSC_EXTERN_CXX_END
#endif

