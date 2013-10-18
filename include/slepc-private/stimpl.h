/*
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
   SLEPc - Scalable Library for Eigenvalue Problem Computations
   Copyright (c) , Universitat Politecnica de Valencia, Spain

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

#if !defined(_STIMPL)
#define _STIMPL

#include <slepcst.h>
#include <slepc-private/slepcimpl.h>

PETSC_EXTERN PetscLogEvent ST_SetUp,ST_Apply,ST_ApplyTranspose;

typedef struct _STOps *STOps;

struct _STOps {
  PetscErrorCode (*setup)(ST);
  PetscErrorCode (*apply)(ST,Vec,Vec);
  PetscErrorCode (*getbilinearform)(ST,Mat*);
  PetscErrorCode (*applytrans)(ST,Vec,Vec);
  PetscErrorCode (*setshift)(ST,PetscScalar);
  PetscErrorCode (*setfromoptions)(ST);
  PetscErrorCode (*postsolve)(ST);
  PetscErrorCode (*backtransform)(ST,PetscInt,PetscScalar*,PetscScalar*);
  PetscErrorCode (*destroy)(ST);
  PetscErrorCode (*reset)(ST);
  PetscErrorCode (*view)(ST,PetscViewer);
  PetscErrorCode (*checknullspace)(ST,PetscInt,const Vec[]);
};

struct _p_ST {
  PETSCHEADER(struct _STOps);
  /*------------------------- User parameters --------------------------*/
  Mat          *A;               /* Matrices which define the eigensystem */
  PetscInt     *Astate;          /* State (to identify the original matrices) */
  Mat          *T;               /* Matrices resulting from transformation */
  PetscInt     nmat;             /* Number of matrices */
  PetscScalar  sigma;            /* Value of the shift */
  PetscBool    sigma_set;        /* whether the user provided the shift or not */
  PetscScalar  defsigma;         /* Default value of the shift */
  STMatMode    shift_matrix;
  MatStructure str;              /* whether matrices have the same pattern or not */
  PetscScalar  gamma,delta;      /* scaling factors */
  PetscBool    userscale;        /* flag indicating the user passed gamma,delta */

  /*------------------------- Misc data --------------------------*/
  KSP          ksp;
  PetscInt     kspidx;           /* which T matrix is associated to ksp */
  Vec          w;
  Vec          D;                /* diagonal matrix for balancing */
  Vec          wb;               /* balancing requires an extra work vector */
  void         *data;
  PetscInt     setupcalled;
  PetscInt     lineariterations;
  PetscInt     applys;
};

PETSC_INTERN PetscErrorCode STGetBilinearForm_Default(ST,Mat*);
PETSC_INTERN PetscErrorCode STCheckNullSpace_Default(ST,PetscInt,const Vec[]);
PETSC_INTERN PetscErrorCode STMatShellCreate(ST,PetscScalar,PetscInt,PetscInt*,Mat*);
PETSC_INTERN PetscErrorCode STMatShellShift(Mat,PetscScalar);
PETSC_INTERN PetscErrorCode STMatSetHermitian(ST,Mat);
PETSC_INTERN PetscErrorCode STMatGAXPY_Private(ST,PetscScalar,PetscScalar,PetscInt,PetscInt,PetscBool);
PETSC_INTERN PetscErrorCode STComputeScaleFactors(ST);

#endif

