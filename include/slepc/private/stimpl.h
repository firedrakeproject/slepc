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
#include <slepc/private/slepcimpl.h>

PETSC_EXTERN PetscBool STRegisterAllCalled;
PETSC_EXTERN PetscErrorCode STRegisterAll(void);
PETSC_EXTERN PetscLogEvent ST_SetUp,ST_Apply,ST_ApplyTranspose,ST_MatSetUp,ST_MatMult,ST_MatMultTranspose,ST_MatSolve,ST_MatSolveTranspose;

typedef struct _STOps *STOps;

struct _STOps {
  PetscErrorCode (*setup)(ST);
  PetscErrorCode (*apply)(ST,Vec,Vec);
  PetscErrorCode (*getbilinearform)(ST,Mat*);
  PetscErrorCode (*applytrans)(ST,Vec,Vec);
  PetscErrorCode (*setshift)(ST,PetscScalar);
  PetscErrorCode (*setfromoptions)(PetscOptionItems*,ST);
  PetscErrorCode (*postsolve)(ST);
  PetscErrorCode (*backtransform)(ST,PetscInt,PetscScalar*,PetscScalar*);
  PetscErrorCode (*destroy)(ST);
  PetscErrorCode (*reset)(ST);
  PetscErrorCode (*view)(ST,PetscViewer);
  PetscErrorCode (*checknullspace)(ST,BV);
};

/*
     'Updated' state means STSetUp must be called because matrices have been
     modified, but the pattern is the same (hence reuse symbolic factorization)
*/
typedef enum { ST_STATE_INITIAL,
               ST_STATE_SETUP,
               ST_STATE_UPDATED } STStateType;

struct _p_ST {
  PETSCHEADER(struct _STOps);
  /*------------------------- User parameters --------------------------*/
  Mat              *A;               /* matrices that define the eigensystem */
  PetscObjectState *Astate;          /* state (to identify the original matrices) */
  Mat              *T;               /* matrices resulting from transformation */
  Mat              P;                /* matrix from which preconditioner is built */
  PetscInt         nmat;             /* number of matrices */
  PetscScalar      sigma;            /* value of the shift */
  PetscBool        sigma_set;        /* whether the user provided the shift or not */
  PetscScalar      defsigma;         /* default value of the shift */
  STMatMode        shift_matrix;
  MatStructure     str;              /* whether matrices have the same pattern or not */
  PetscBool        transform;        /* whether transformed matrices are computed */

  /*------------------------- Misc data --------------------------*/
  KSP              ksp;
  Vec              w;                /* work vector used in apply operation */
  Vec              D;                /* diagonal matrix for balancing */
  Vec              wb;               /* balancing requires an extra work vector */
  void             *data;
  STStateType      state;            /* initial -> setup -> with updated matrices */
};

#undef __FUNCT__
#define __FUNCT__ "ST_AllocateWorkVec"
/*
  ST_AllocateWorkVec - Allocate work vector for the STApply operation.
*/
PETSC_STATIC_INLINE PetscErrorCode ST_AllocateWorkVec(ST st)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (!st->w) {
    ierr = MatCreateVecs(st->A[0],&st->w,NULL);CHKERRQ(ierr);
    ierr = PetscLogObjectParent((PetscObject)st,(PetscObject)st->w);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

/*
    Macros to test valid ST arguments
*/
#if !defined(PETSC_USE_DEBUG)

#define STCheckMatrices(h,arg) do {} while (0)

#else

#define STCheckMatrices(h,arg) \
  do { \
    if (!h->A) SETERRQ1(PetscObjectComm((PetscObject)h),PETSC_ERR_ARG_WRONGSTATE,"ST matrices have not been set: Parameter #%d",arg); \
  } while (0)

#endif

PETSC_INTERN PetscErrorCode STGetBilinearForm_Default(ST,Mat*);
PETSC_INTERN PetscErrorCode STCheckNullSpace_Default(ST,BV);
PETSC_INTERN PetscErrorCode STMatShellCreate(ST,PetscScalar,PetscInt,PetscInt*,PetscScalar*,Mat*);
PETSC_INTERN PetscErrorCode STMatShellShift(Mat,PetscScalar);
PETSC_INTERN PetscErrorCode STMatSetHermitian(ST,Mat);
PETSC_INTERN PetscErrorCode STCheckFactorPackage(ST);
PETSC_INTERN PetscErrorCode STMatMAXPY_Private(ST,PetscScalar,PetscScalar,PetscInt,PetscScalar*,PetscBool,Mat*);
PETSC_INTERN PetscErrorCode STCoeffs_Monomial(ST,PetscScalar*);

#endif
