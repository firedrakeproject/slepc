/*
      Implements the shift-and-invert technique for eigenvalue problems.

   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
   SLEPc - Scalable Library for Eigenvalue Problem Computations
   Copyright (c) 2002-2016, Universitat Politecnica de Valencia, Spain

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

#include <slepc/private/stimpl.h>

PetscErrorCode STApply_Sinvert(ST st,Vec x,Vec y)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (st->nmat>1) {
    /* generalized eigenproblem: y = (A - sB)^-1 B x */
    ierr = MatMult(st->T[0],x,st->w);CHKERRQ(ierr);
    ierr = STMatSolve(st,st->w,y);CHKERRQ(ierr);
  } else {
    /* standard eigenproblem: y = (A - sI)^-1 x */
    ierr = STMatSolve(st,x,y);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

PetscErrorCode STApplyTranspose_Sinvert(ST st,Vec x,Vec y)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (st->nmat>1) {
    /* generalized eigenproblem: y = B^T (A - sB)^-T x */
    ierr = STMatSolveTranspose(st,x,st->w);CHKERRQ(ierr);
    ierr = MatMultTranspose(st->T[0],st->w,y);CHKERRQ(ierr);
  } else {
    /* standard eigenproblem: y = (A - sI)^-T x */
    ierr = STMatSolveTranspose(st,x,y);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

PetscErrorCode STBackTransform_Sinvert(ST st,PetscInt n,PetscScalar *eigr,PetscScalar *eigi)
{
  PetscInt    j;
#if !defined(PETSC_USE_COMPLEX)
  PetscScalar t;
#endif

  PetscFunctionBegin;
#if !defined(PETSC_USE_COMPLEX)
  for (j=0;j<n;j++) {
    if (eigi[j] == 0) eigr[j] = 1.0 / eigr[j] + st->sigma;
    else {
      t = eigr[j] * eigr[j] + eigi[j] * eigi[j];
      eigr[j] = eigr[j] / t + st->sigma;
      eigi[j] = - eigi[j] / t;
    }
  }
#else
  for (j=0;j<n;j++) {
    eigr[j] = 1.0 / eigr[j] + st->sigma;
  }
#endif
  PetscFunctionReturn(0);
}

PetscErrorCode STPostSolve_Sinvert(ST st)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (st->shift_matrix == ST_MATMODE_INPLACE) {
    if (st->nmat>1) {
      ierr = MatAXPY(st->A[0],st->sigma,st->A[1],st->str);CHKERRQ(ierr);
    } else {
      ierr = MatShift(st->A[0],st->sigma);CHKERRQ(ierr);
    }
    st->Astate[0] = ((PetscObject)st->A[0])->state;
    st->state = ST_STATE_INITIAL;
  }
  PetscFunctionReturn(0);
}

PetscErrorCode STSetUp_Sinvert(ST st)
{
  PetscErrorCode ierr;
  PetscInt       k,nc,nmat=PetscMax(st->nmat,2);
  PetscScalar    *coeffs=NULL;

  PetscFunctionBegin;
  if (st->nmat>1) {
    ierr = ST_AllocateWorkVec(st);CHKERRQ(ierr);
  }
  /* if the user did not set the shift, use the target value */
  if (!st->sigma_set) st->sigma = st->defsigma;
  if (st->transform) {
    if (nmat>2) {
      nc = (nmat*(nmat+1))/2;
      ierr = PetscMalloc1(nc,&coeffs);CHKERRQ(ierr);
      /* Compute coeffs */
      ierr = STCoeffs_Monomial(st,coeffs);CHKERRQ(ierr);
    }
    /* T[0] = A_n */
    k = nmat-1;
    ierr = PetscObjectReference((PetscObject)st->A[k]);CHKERRQ(ierr);
    ierr = MatDestroy(&st->T[0]);CHKERRQ(ierr);
    st->T[0] = st->A[k];
    for (k=1;k<nmat;k++) {
      ierr = STMatMAXPY_Private(st,nmat>2?st->sigma:-st->sigma,0.0,nmat-k-1,coeffs?coeffs+(k*(k+1))/2:NULL,PetscNot(st->state==ST_STATE_UPDATED),&st->T[k]);CHKERRQ(ierr);
    }
    if (nmat>2) { ierr = PetscFree(coeffs);CHKERRQ(ierr); }
    ierr = PetscObjectReference((PetscObject)st->T[nmat-1]);CHKERRQ(ierr);
    ierr = MatDestroy(&st->P);CHKERRQ(ierr);
    st->P = st->T[nmat-1];
  } else {
    for (k=0;k<nmat;k++) {
      ierr = PetscObjectReference((PetscObject)st->A[k]);CHKERRQ(ierr);
      ierr = MatDestroy(&st->T[k]);CHKERRQ(ierr);
      st->T[k] = st->A[k];
    }
  }
  if (st->P) {
    if (!st->ksp) { ierr = STGetKSP(st,&st->ksp);CHKERRQ(ierr); }
    ierr = STCheckFactorPackage(st);CHKERRQ(ierr);
    ierr = KSPSetOperators(st->ksp,st->P,st->P);CHKERRQ(ierr);
    ierr = KSPSetUp(st->ksp);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

PetscErrorCode STSetShift_Sinvert(ST st,PetscScalar newshift)
{
  PetscErrorCode ierr;
  PetscInt       nmat=PetscMax(st->nmat,2),k,nc;
  PetscScalar    *coeffs=NULL;

  PetscFunctionBegin;
  if (st->transform) {
    if (st->shift_matrix == ST_MATMODE_COPY && nmat>2) {
      nc = (nmat*(nmat+1))/2;
      ierr = PetscMalloc1(nc,&coeffs);CHKERRQ(ierr);
      /* Compute coeffs */
      ierr = STCoeffs_Monomial(st,coeffs);CHKERRQ(ierr);
    }
    for (k=1;k<nmat;k++) {
      ierr = STMatMAXPY_Private(st,nmat>2?newshift:-newshift,nmat>2?st->sigma:-st->sigma,nmat-k-1,coeffs?coeffs+(k*(k+1))/2:NULL,PETSC_FALSE,&st->T[k]);CHKERRQ(ierr);
    }
    if (st->shift_matrix == ST_MATMODE_COPY && nmat>2) {
      ierr = PetscFree(coeffs);CHKERRQ(ierr);
    }
    if (st->P!=st->T[nmat-1]) {
      ierr = MatDestroy(&st->P);CHKERRQ(ierr);
      st->P = st->T[nmat-1];
      ierr = PetscObjectReference((PetscObject)st->P);CHKERRQ(ierr);
    }
  }
  if (st->P) {
    if (!st->ksp) { ierr = STGetKSP(st,&st->ksp);CHKERRQ(ierr); }
    ierr = KSPSetOperators(st->ksp,st->P,st->P);CHKERRQ(ierr);
    ierr = KSPSetUp(st->ksp);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

PETSC_EXTERN PetscErrorCode STCreate_Sinvert(ST st)
{
  PetscFunctionBegin;
  st->ops->apply           = STApply_Sinvert;
  st->ops->getbilinearform = STGetBilinearForm_Default;
  st->ops->applytrans      = STApplyTranspose_Sinvert;
  st->ops->postsolve       = STPostSolve_Sinvert;
  st->ops->backtransform   = STBackTransform_Sinvert;
  st->ops->setup           = STSetUp_Sinvert;
  st->ops->setshift        = STSetShift_Sinvert;
  st->ops->checknullspace  = STCheckNullSpace_Default;
  st->ops->setdefaultksp   = STSetDefaultKSP_Default;
  PetscFunctionReturn(0);
}
