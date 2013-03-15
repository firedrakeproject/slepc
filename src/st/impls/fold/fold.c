/*
    Folding spectral transformation, applies (A + sigma I)^2 as operator, or 
    inv(B)(A + sigma I)^2 for generalized problems

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

#include <slepc-private/stimpl.h>          /*I "slepcst.h" I*/

typedef struct {
  Vec         w2;
} ST_FOLD;

#undef __FUNCT__
#define __FUNCT__ "STApply_Fold"
PetscErrorCode STApply_Fold(ST st,Vec x,Vec y)
{
  PetscErrorCode     ierr;
  ST_FOLD            *ctx = (ST_FOLD*)st->data;
  PetscInt           its;
  KSPConvergedReason reason;

  PetscFunctionBegin;
  if (st->nmat>1) {
    /* generalized eigenproblem: y = (B^-1 A + sI)^2 x */
    ierr = MatMult(st->A[0],x,st->w);CHKERRQ(ierr);
    ierr = KSPSolve(st->ksp,st->w,ctx->w2);CHKERRQ(ierr);
    ierr = KSPGetConvergedReason(st->ksp,&reason);CHKERRQ(ierr);
    if (reason<0) SETERRQ1(PetscObjectComm((PetscObject)st),PETSC_ERR_NOT_CONVERGED,"KSP did not converge (reason=%s)",KSPConvergedReasons[reason]);
    ierr = KSPGetIterationNumber(st->ksp,&its);CHKERRQ(ierr);  
    st->lineariterations += its;
    if (st->sigma != 0.0) {
      ierr = VecAXPY(ctx->w2,-st->sigma,x);CHKERRQ(ierr);
    }
    ierr = MatMult(st->A[0],ctx->w2,st->w);CHKERRQ(ierr);
    ierr = KSPSolve(st->ksp,st->w,y);CHKERRQ(ierr);
    ierr = KSPGetConvergedReason(st->ksp,&reason);CHKERRQ(ierr);
    if (reason<0) SETERRQ1(PetscObjectComm((PetscObject)st),PETSC_ERR_NOT_CONVERGED,"KSP did not converge (reason=%s)",KSPConvergedReasons[reason]);
    ierr = KSPGetIterationNumber(st->ksp,&its);CHKERRQ(ierr);  
    st->lineariterations += its;
    if (st->sigma != 0.0) {
      ierr = VecAXPY(y,-st->sigma,ctx->w2);CHKERRQ(ierr);
    }
  } else {
    /* standard eigenproblem: y = (A + sI)^2 x */
    ierr = MatMult(st->A[0],x,st->w);CHKERRQ(ierr);
    if (st->sigma != 0.0) {
      ierr = VecAXPY(st->w,-st->sigma,x);CHKERRQ(ierr);
    }
    ierr = MatMult(st->A[0],st->w,y);CHKERRQ(ierr);
    if (st->sigma != 0.0) {
      ierr = VecAXPY(y,-st->sigma,st->w);CHKERRQ(ierr);
    }
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "STApplyTranspose_Fold"
PetscErrorCode STApplyTranspose_Fold(ST st,Vec x,Vec y)
{
  PetscErrorCode     ierr;
  ST_FOLD            *ctx = (ST_FOLD*)st->data;
  PetscInt           its;
  KSPConvergedReason reason;

  PetscFunctionBegin;
  if (st->nmat>1) {
    /* generalized eigenproblem: y = (A^T B^-T + sI)^2 x */
    ierr = KSPSolveTranspose(st->ksp,x,st->w);CHKERRQ(ierr);
    ierr = KSPGetConvergedReason(st->ksp,&reason);CHKERRQ(ierr);
    if (reason<0) SETERRQ1(PetscObjectComm((PetscObject)st),PETSC_ERR_NOT_CONVERGED,"KSP did not converge (reason=%s)",KSPConvergedReasons[reason]);
    ierr = KSPGetIterationNumber(st->ksp,&its);CHKERRQ(ierr);  
    st->lineariterations += its;
    ierr = MatMult(st->A[0],st->w,ctx->w2);CHKERRQ(ierr);
    if (st->sigma != 0.0) {
      ierr = VecAXPY(ctx->w2,-st->sigma,x);CHKERRQ(ierr);
    }
    ierr = KSPSolveTranspose(st->ksp,ctx->w2,st->w);CHKERRQ(ierr);
    ierr = KSPGetConvergedReason(st->ksp,&reason);CHKERRQ(ierr);
    if (reason<0) SETERRQ1(PetscObjectComm((PetscObject)st),PETSC_ERR_NOT_CONVERGED,"KSP did not converge (reason=%s)",KSPConvergedReasons[reason]);
    ierr = KSPGetIterationNumber(st->ksp,&its);CHKERRQ(ierr);  
    st->lineariterations += its;
    ierr = MatMult(st->A[0],st->w,y);CHKERRQ(ierr);
    if (st->sigma != 0.0) {
      ierr = VecAXPY(y,-st->sigma,ctx->w2);CHKERRQ(ierr);
    }
  } else {
    /* standard eigenproblem: y = (A^T + sI)^2 x */
    ierr = MatMultTranspose(st->A[0],x,st->w);CHKERRQ(ierr);
    if (st->sigma != 0.0) {
      ierr = VecAXPY(st->w,-st->sigma,x);CHKERRQ(ierr);
    }
    ierr = MatMultTranspose(st->A[0],st->w,y);CHKERRQ(ierr);
    if (st->sigma != 0.0) {
      ierr = VecAXPY(y,-st->sigma,st->w);CHKERRQ(ierr);
    }
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "STBackTransform_Fold"
PetscErrorCode STBackTransform_Fold(ST st,PetscInt n,PetscScalar *eigr,PetscScalar *eigi)
{
  PetscInt    j;
#if !defined(PETSC_USE_COMPLEX)
  PetscScalar r,x,y;
#endif

  PetscFunctionBegin;
  for (j=0;j<n;j++) {
#if !defined(PETSC_USE_COMPLEX)
    if (eigi[j] == 0.0) {
#endif
      eigr[j] = st->sigma + PetscSqrtScalar(eigr[j]);
#if !defined(PETSC_USE_COMPLEX)
    } else {
      r = SlepcAbsEigenvalue(eigr[j],eigi[j]);
      x = PetscSqrtScalar((r + eigr[j]) / 2);
      y = PetscSqrtScalar((r - eigr[j]) / 2);
      if (eigi[j] < 0.0) y = - y;
      eigr[j] = st->sigma + x;
      eigi[j] = y;
    }
#endif
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "STSetUp_Fold"
PetscErrorCode STSetUp_Fold(ST st)
{
  PetscErrorCode ierr;
  ST_FOLD        *ctx = (ST_FOLD*)st->data;

  PetscFunctionBegin;
  /* if the user did not set the shift, use the target value */
  if (!st->sigma_set) st->sigma = st->defsigma;

  if (st->nmat>1) {
    if (!st->ksp) { ierr = STGetKSP(st,&st->ksp);CHKERRQ(ierr); }
    ierr = KSPSetOperators(st->ksp,st->A[1],st->A[1],DIFFERENT_NONZERO_PATTERN);CHKERRQ(ierr);
    ierr = KSPSetUp(st->ksp);CHKERRQ(ierr);
    ierr = VecDestroy(&ctx->w2);CHKERRQ(ierr);
    ierr = MatGetVecs(st->A[1],&ctx->w2,NULL);CHKERRQ(ierr); 
  } 
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "STSetFromOptions_Fold"
PetscErrorCode STSetFromOptions_Fold(ST st) 
{
  PetscErrorCode ierr;
  PC             pc;
  PCType         pctype;
  KSPType        ksptype;

  PetscFunctionBegin;
  if (!st->ksp) { ierr = STGetKSP(st,&st->ksp);CHKERRQ(ierr); }
  ierr = KSPGetPC(st->ksp,&pc);CHKERRQ(ierr);
  ierr = KSPGetType(st->ksp,&ksptype);CHKERRQ(ierr);
  ierr = PCGetType(pc,&pctype);CHKERRQ(ierr);
  if (!pctype && !ksptype) {
    if (st->shift_matrix == ST_MATMODE_SHELL) {
      /* in shell mode use GMRES with Jacobi as the default */
      ierr = KSPSetType(st->ksp,KSPGMRES);CHKERRQ(ierr);
      ierr = PCSetType(pc,PCJACOBI);CHKERRQ(ierr);
    } else {
      /* use direct solver as default */
      ierr = KSPSetType(st->ksp,KSPPREONLY);CHKERRQ(ierr);
      ierr = PCSetType(pc,PCREDUNDANT);CHKERRQ(ierr);
    }
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "STReset_Fold"
PetscErrorCode STReset_Fold(ST st)
{
  PetscErrorCode ierr;
  ST_FOLD        *ctx = (ST_FOLD*)st->data;

  PetscFunctionBegin;
  ierr = VecDestroy(&ctx->w2);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "STDestroy_Fold"
PetscErrorCode STDestroy_Fold(ST st)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscFree(st->data);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "STCreate_Fold"
PETSC_EXTERN PetscErrorCode STCreate_Fold(ST st)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscNewLog(st,ST_FOLD,&st->data);CHKERRQ(ierr);
  st->ops->apply           = STApply_Fold;
  st->ops->getbilinearform = STGetBilinearForm_Default;
  st->ops->applytrans      = STApplyTranspose_Fold;
  st->ops->backtransform   = STBackTransform_Fold;
  st->ops->setup           = STSetUp_Fold;
  st->ops->setfromoptions  = STSetFromOptions_Fold;
  st->ops->destroy         = STDestroy_Fold;
  st->ops->reset           = STReset_Fold;
  PetscFunctionReturn(0);
}

