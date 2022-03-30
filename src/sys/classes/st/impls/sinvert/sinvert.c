/*
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
   SLEPc - Scalable Library for Eigenvalue Problem Computations
   Copyright (c) 2002-, Universitat Politecnica de Valencia, Spain

   This file is part of SLEPc.
   SLEPc is distributed under a 2-clause BSD license (see LICENSE).
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
*/
/*
   Implements the shift-and-invert technique for eigenvalue problems
*/

#include <slepc/private/stimpl.h>

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
  PetscFunctionBegin;
  if (st->matmode == ST_MATMODE_INPLACE) {
    if (st->nmat>1) PetscCall(MatAXPY(st->A[0],st->sigma,st->A[1],st->str));
    else PetscCall(MatShift(st->A[0],st->sigma));
    st->Astate[0] = ((PetscObject)st->A[0])->state;
    st->state   = ST_STATE_INITIAL;
    st->opready = PETSC_FALSE;
  }
  PetscFunctionReturn(0);
}

/*
   Operator (sinvert):
               Op               P         M
   if nmat=1:  (A-sI)^-1        A-sI      NULL
   if nmat=2:  (A-sB)^-1 B      A-sB      B
*/
PetscErrorCode STComputeOperator_Sinvert(ST st)
{
  PetscFunctionBegin;
  /* if the user did not set the shift, use the target value */
  if (!st->sigma_set) st->sigma = st->defsigma;
  PetscCall(PetscObjectReference((PetscObject)st->A[1]));
  PetscCall(MatDestroy(&st->T[0]));
  st->T[0] = st->A[1];
  PetscCall(STMatMAXPY_Private(st,-st->sigma,0.0,0,NULL,PetscNot(st->state==ST_STATE_UPDATED),PETSC_FALSE,&st->T[1]));
  PetscCall(PetscObjectReference((PetscObject)st->T[1]));
  PetscCall(MatDestroy(&st->P));
  st->P = st->T[1];
  st->M = (st->nmat>1)? st->T[0]: NULL;
  if (st->Psplit) {  /* build custom preconditioner from the split matrices */
    PetscCall(STMatMAXPY_Private(st,-st->sigma,0.0,0,NULL,PETSC_TRUE,PETSC_TRUE,&st->Pmat));
  }
  PetscFunctionReturn(0);
}

PetscErrorCode STSetUp_Sinvert(ST st)
{
  PetscInt       k,nc,nmat=st->nmat;
  PetscScalar    *coeffs=NULL;

  PetscFunctionBegin;
  if (nmat>1) PetscCall(STSetWorkVecs(st,1));
  /* if the user did not set the shift, use the target value */
  if (!st->sigma_set) st->sigma = st->defsigma;
  if (nmat>2) {  /* set-up matrices for polynomial eigenproblems */
    if (st->transform) {
      nc = (nmat*(nmat+1))/2;
      PetscCall(PetscMalloc1(nc,&coeffs));
      /* Compute coeffs */
      PetscCall(STCoeffs_Monomial(st,coeffs));
      /* T[0] = A_n */
      k = nmat-1;
      PetscCall(PetscObjectReference((PetscObject)st->A[k]));
      PetscCall(MatDestroy(&st->T[0]));
      st->T[0] = st->A[k];
      for (k=1;k<nmat;k++) PetscCall(STMatMAXPY_Private(st,nmat>2?st->sigma:-st->sigma,0.0,nmat-k-1,coeffs?coeffs+(k*(k+1))/2:NULL,PetscNot(st->state==ST_STATE_UPDATED),PETSC_FALSE,&st->T[k]));
      PetscCall(PetscFree(coeffs));
      PetscCall(PetscObjectReference((PetscObject)st->T[nmat-1]));
      PetscCall(MatDestroy(&st->P));
      st->P = st->T[nmat-1];
      if (st->Psplit) {  /* build custom preconditioner from the split matrices */
        PetscCall(STMatMAXPY_Private(st,st->sigma,0.0,0,coeffs?coeffs+((nmat-1)*nmat)/2:NULL,PETSC_TRUE,PETSC_TRUE,&st->Pmat));
      }
      PetscCall(ST_KSPSetOperators(st,st->P,st->Pmat?st->Pmat:st->P));
    } else {
      for (k=0;k<nmat;k++) {
        PetscCall(PetscObjectReference((PetscObject)st->A[k]));
        PetscCall(MatDestroy(&st->T[k]));
        st->T[k] = st->A[k];
      }
    }
  }
  if (st->P) PetscCall(KSPSetUp(st->ksp));
  PetscFunctionReturn(0);
}

PetscErrorCode STSetShift_Sinvert(ST st,PetscScalar newshift)
{
  PetscInt       nmat=PetscMax(st->nmat,2),k,nc;
  PetscScalar    *coeffs=NULL;

  PetscFunctionBegin;
  if (st->transform) {
    if (st->matmode == ST_MATMODE_COPY && nmat>2) {
      nc = (nmat*(nmat+1))/2;
      PetscCall(PetscMalloc1(nc,&coeffs));
      /* Compute coeffs */
      PetscCall(STCoeffs_Monomial(st,coeffs));
    }
    for (k=1;k<nmat;k++) PetscCall(STMatMAXPY_Private(st,nmat>2?newshift:-newshift,nmat>2?st->sigma:-st->sigma,nmat-k-1,coeffs?coeffs+(k*(k+1))/2:NULL,PETSC_FALSE,PETSC_FALSE,&st->T[k]));
    if (st->matmode == ST_MATMODE_COPY && nmat>2) PetscCall(PetscFree(coeffs));
    if (st->P!=st->T[nmat-1]) {
      PetscCall(PetscObjectReference((PetscObject)st->T[nmat-1]));
      PetscCall(MatDestroy(&st->P));
      st->P = st->T[nmat-1];
    }
    if (st->Psplit) {  /* build custom preconditioner from the split matrices */
      PetscCall(STMatMAXPY_Private(st,nmat>2?newshift:-newshift,nmat>2?st->sigma:-st->sigma,0,coeffs?coeffs+((nmat-1)*nmat)/2:NULL,PETSC_FALSE,PETSC_TRUE,&st->Pmat));
    }
  }
  if (st->P) {
    PetscCall(ST_KSPSetOperators(st,st->P,st->Pmat?st->Pmat:st->P));
    PetscCall(KSPSetUp(st->ksp));
  }
  PetscFunctionReturn(0);
}

SLEPC_EXTERN PetscErrorCode STCreate_Sinvert(ST st)
{
  PetscFunctionBegin;
  st->usesksp = PETSC_TRUE;

  st->ops->apply           = STApply_Generic;
  st->ops->applytrans      = STApplyTranspose_Generic;
  st->ops->backtransform   = STBackTransform_Sinvert;
  st->ops->setshift        = STSetShift_Sinvert;
  st->ops->getbilinearform = STGetBilinearForm_Default;
  st->ops->setup           = STSetUp_Sinvert;
  st->ops->computeoperator = STComputeOperator_Sinvert;
  st->ops->postsolve       = STPostSolve_Sinvert;
  st->ops->checknullspace  = STCheckNullSpace_Default;
  st->ops->setdefaultksp   = STSetDefaultKSP_Default;
  PetscFunctionReturn(0);
}
