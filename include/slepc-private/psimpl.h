/*
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
   SLEPc - Scalable Library for Eigenvalue Problem Computations
   Copyright (c) 2002-2011, Universitat Politecnica de Valencia, Spain

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

#ifndef _PSIMPL
#define _PSIMPL

#include <slepcps.h>

PETSC_EXTERN PetscLogEvent PS_Solve,PS_Vectors,PS_Other;
PETSC_EXTERN const char *PSMatName[];

typedef struct _PSOps *PSOps;

struct _PSOps {
  PetscErrorCode (*allocate)(PS,PetscInt);
  PetscErrorCode (*view)(PS,PetscViewer);
  PetscErrorCode (*vectors)(PS,PSMatType,PetscInt*,PetscReal*);
  PetscErrorCode (*solve[PS_MAX_SOLVE])(PS,PetscScalar*,PetscScalar*);
  PetscErrorCode (*truncate)(PS,PetscInt);
  PetscErrorCode (*cond)(PS,PetscReal*);
  PetscErrorCode (*transharm)(PS,PetscScalar,PetscReal,PetscBool,PetscScalar*,PetscReal*);
  PetscErrorCode (*transrks)(PS,PetscScalar);
  PetscErrorCode (*normalize)(PS,PSMatType,PetscInt);
};

struct _p_PS {
  PETSCHEADER(struct _PSOps);
  PetscInt       method;             /* identifies the variant to be used */
  PetscBool      compact;            /* whether the matrices are stored in compact form */
  PetscBool      refined;            /* get refined vectors instead of regular vectors */
  PetscBool      extrarow;           /* assume the matrix dimension is (n+1) x n */
  PetscInt       ld;                 /* leading dimension */
  PetscInt       l;                  /* number of locked (inactive) leading columns */
  PetscInt       n;                  /* current dimension */
  PetscInt       m;                  /* current column dimension (for SVD only) */
  PetscInt       k;                  /* intermediate dimension (e.g. position of arrow) */
  PSStateType    state;              /* the current state */
  PetscScalar    *mat[PS_NUM_MAT];   /* the matrices */
  PetscReal      *rmat[PS_NUM_MAT];  /* the matrices (real) */
  PetscInt       *perm;              /* permutation */
  PetscScalar    *work;
  PetscReal      *rwork;
  PetscBLASInt   *iwork;
  PetscInt       lwork,lrwork,liwork;
  PetscErrorCode (*comp_fun)(PetscScalar,PetscScalar,PetscScalar,PetscScalar,PetscInt*,void*);
  void           *comp_ctx;
};

PETSC_EXTERN PetscErrorCode PSAllocateMat_Private(PS,PSMatType);
PETSC_EXTERN PetscErrorCode PSAllocateMatReal_Private(PS,PSMatType);
PETSC_EXTERN PetscErrorCode PSAllocateWork_Private(PS,PetscInt,PetscInt,PetscInt);
PETSC_EXTERN PetscErrorCode PSViewMat_Private(PS,PetscViewer,PSMatType);
PETSC_EXTERN PetscErrorCode PSSortEigenvaluesReal_Private(PS,PetscInt,PetscInt,PetscReal*,PetscInt*);
PETSC_EXTERN PetscErrorCode PSPermuteColumns_Private(PS,PetscInt,PetscInt,PSMatType,PetscInt*);
PETSC_EXTERN PetscErrorCode PSCopyMatrix_Private(PS,PSMatType,PSMatType);

#endif
