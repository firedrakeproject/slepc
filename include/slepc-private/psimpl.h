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

extern PetscLogEvent PS_Solve,PS_Sort;

typedef struct _PSOps *PSOps;

struct _PSOps {
  PetscErrorCode (*allocate)(PS,PetscInt);
  PetscErrorCode (*computevector)(PS,PetscInt,PSMatType,PetscBool*);
  PetscErrorCode (*solve)(PS);
  PetscErrorCode (*sort)(PS);
};

struct _p_PS {
  PETSCHEADER(struct _PSOps);
  PetscInt    ld;                 /* leading dimension */
  PetscInt    l;                  /* number of locked (inactive) leading columns */
  PetscInt    n;                  /* current dimension */
  PetscInt    k;                  /* intermediate dimension (e.g. position of arrow) */
  PSStateType state;              /* the current state */
  PetscScalar *mat[PS_NUM_MAT];   /* the matrices */
  PetscReal   *rmat[PS_NUM_MAT];  /* the matrices (real) */
  PetscScalar *work;
  PetscReal   *rwork;
};

extern PetscErrorCode PSAllocateMat_Private(PS,PSMatType);

#endif
