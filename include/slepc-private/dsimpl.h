/*
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
   SLEPc - Scalable Library for Eigenvalue Problem Computations
   Copyright (c) 2002-2012, Universitat Politecnica de Valencia, Spain

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

#ifndef _DSIMPL
#define _DSIMPL

#include <slepcds.h>

PETSC_EXTERN PetscLogEvent DS_Solve,DS_Vectors,DS_Other;
PETSC_EXTERN const char *DSMatName[];

typedef struct _DSOps *DSOps;

struct _DSOps {
  PetscErrorCode (*allocate)(DS,PetscInt);
  PetscErrorCode (*view)(DS,PetscViewer);
  PetscErrorCode (*vectors)(DS,DSMatType,PetscInt*,PetscReal*);
  PetscErrorCode (*solve[DS_MAX_SOLVE])(DS,PetscScalar*,PetscScalar*);
  PetscErrorCode (*sort)(DS,PetscScalar*,PetscScalar*,PetscScalar*,PetscScalar*,PetscInt*);
  PetscErrorCode (*truncate)(DS,PetscInt);
  PetscErrorCode (*update)(DS);
  PetscErrorCode (*cond)(DS,PetscReal*);
  PetscErrorCode (*transharm)(DS,PetscScalar,PetscReal,PetscBool,PetscScalar*,PetscReal*);
  PetscErrorCode (*transrks)(DS,PetscScalar);
  PetscErrorCode (*normalize)(DS,DSMatType,PetscInt);
};

struct _p_DS {
  PETSCHEADER(struct _DSOps);
  PetscInt       method;             /* identifies the variant to be used */
  PetscBool      compact;            /* whether the matrices are stored in compact form */
  PetscBool      refined;            /* get refined vectors instead of regular vectors */
  PetscBool      extrarow;           /* assume the matrix dimension is (n+1) x n */
  PetscInt       ld;                 /* leading dimension */
  PetscInt       l;                  /* number of locked (inactive) leading columns */
  PetscInt       n;                  /* current dimension */
  PetscInt       m;                  /* current column dimension (for SVD only) */
  PetscInt       k;                  /* intermediate dimension (e.g. position of arrow) */
  DSStateType    state;              /* the current state */
  PetscScalar    *mat[DS_NUM_MAT];   /* the matrices */
  PetscReal      *rmat[DS_NUM_MAT];  /* the matrices (real) */
  PetscInt       *perm;              /* permutation */
  PetscScalar    *work;
  PetscReal      *rwork;
  PetscBLASInt   *iwork;
  PetscInt       lwork,lrwork,liwork;
  PetscErrorCode (*comp_fun)(PetscScalar,PetscScalar,PetscScalar,PetscScalar,PetscInt*,void*);
  void           *comp_ctx;
};

PETSC_EXTERN PetscErrorCode DSAllocateMat_Private(DS,DSMatType);
PETSC_EXTERN PetscErrorCode DSAllocateMatReal_Private(DS,DSMatType);
PETSC_EXTERN PetscErrorCode DSAllocateWork_Private(DS,PetscInt,PetscInt,PetscInt);
PETSC_EXTERN PetscErrorCode DSViewMat_Private(DS,PetscViewer,DSMatType);
PETSC_EXTERN PetscErrorCode DSSortEigenvalues_Private(DS,PetscScalar*,PetscScalar*,PetscInt*,PetscBool);
PETSC_EXTERN PetscErrorCode DSSortEigenvaluesReal_Private(DS,PetscReal*,PetscInt*);
PETSC_EXTERN PetscErrorCode DSPermuteColumns_Private(DS,PetscInt,PetscInt,DSMatType,PetscInt*);
PETSC_EXTERN PetscErrorCode DSPermuteRows_Private(DS,PetscInt,PetscInt,DSMatType,PetscInt*);
PETSC_EXTERN PetscErrorCode DSPermuteBoth_Private(DS,PetscInt,PetscInt,DSMatType,DSMatType,PetscInt*);
PETSC_EXTERN PetscErrorCode DSCopyMatrix_Private(DS,DSMatType,DSMatType);

#endif
