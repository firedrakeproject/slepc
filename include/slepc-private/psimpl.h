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

extern PetscLogEvent PS_Solve,PS_Sort,PS_Vectors,PS_Other;
extern const char *PSMatName[];

typedef struct _PSOps *PSOps;

struct _PSOps {
  PetscErrorCode (*allocate)(PS,PetscInt);
  PetscErrorCode (*view)(PS,PetscViewer);
  PetscErrorCode (*vectors)(PS,PSMatType,PetscInt*,PetscReal*);
  PetscErrorCode (*solve)(PS,PetscScalar*,PetscScalar*);
  PetscErrorCode (*sort)(PS,PetscScalar*,PetscScalar*,PetscErrorCode(*)(PetscScalar,PetscScalar,PetscScalar,PetscScalar,PetscInt*,void*),void*);
  PetscErrorCode (*cond)(PS,PetscReal*);
  PetscErrorCode (*transharm)(PS,PetscScalar,PetscReal,PetscBool,PetscScalar*,PetscReal*);
  PetscErrorCode (*transrks)(PS,PetscScalar);
};

struct _p_PS {
  PETSCHEADER(struct _PSOps);
  PetscInt     method;             /* identifies the variant to be used */
  PetscInt     nmeth;              /* number of methods available in this ps */
  PetscBool    compact;            /* whether the matrices are stored in compact form */
  PetscBool    refined;            /* get refined vectors instead of regular vectors */
  PetscInt     ld;                 /* leading dimension */
  PetscInt     l;                  /* number of locked (inactive) leading columns */
  PetscInt     n;                  /* current dimension */
  PetscInt     k;                  /* intermediate dimension (e.g. position of arrow) */
  PSStateType  state;              /* the current state */
  PetscScalar  *mat[PS_NUM_MAT];   /* the matrices */
  PetscReal    *rmat[PS_NUM_MAT];  /* the matrices (real) */
  PetscScalar  *work;
  PetscReal    *rwork;
  PetscBLASInt *iwork;
  PetscInt     lwork,lrwork,liwork;
};

extern PetscErrorCode PSAllocateMat_Private(PS,PSMatType);
extern PetscErrorCode PSAllocateMatReal_Private(PS,PSMatType);
extern PetscErrorCode PSAllocateWork_Private(PS,PetscInt,PetscInt,PetscInt);
extern PetscErrorCode PSViewMat_Private(PS,PetscViewer,PSMatType);
extern PetscErrorCode PSSortEigenvaluesReal_Private(PS,PetscInt,PetscInt,PetscReal*,PetscInt*,PetscErrorCode (*)(PetscScalar,PetscScalar,PetscScalar,PetscScalar,PetscInt*,void*),void*);
extern PetscErrorCode PSPermuteColumns_Private(PS,PetscInt,PetscInt,PSMatType,PetscInt*);
extern PetscErrorCode PSCopyMatrix_Private(PS,PSMatType,PSMatType);

#endif
