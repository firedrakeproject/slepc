/*
    Shift spectral transformation, applies (A + sigma I) as operator, or 
    inv(B)(A + sigma B) for generalized problems

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

#undef __FUNCT__  
#define __FUNCT__ "STApply_Shift"
PetscErrorCode STApply_Shift(ST st,Vec x,Vec y)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (st->nmat>1) {
    /* generalized eigenproblem: y = (B^-1 A + sI) x */
    ierr = MatMult(st->A[0],x,st->w);CHKERRQ(ierr);
    ierr = STAssociatedKSPSolve(st,st->w,y);CHKERRQ(ierr);
  }
  else {
    /* standard eigenproblem: y = (A + sI) x */
    ierr = MatMult(st->A[0],x,y);CHKERRQ(ierr);
  }
  if (st->sigma != 0.0) {
    ierr = VecAXPY(y,st->sigma,x);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "STApplyTranspose_Shift"
PetscErrorCode STApplyTranspose_Shift(ST st,Vec x,Vec y)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (st->nmat>1) {
    /* generalized eigenproblem: y = (A^T B^-T + sI) x */
    ierr = STAssociatedKSPSolveTranspose(st,x,st->w);CHKERRQ(ierr);
    ierr = MatMultTranspose(st->A[0],st->w,y);CHKERRQ(ierr);
  }
  else {
    /* standard eigenproblem: y = (A^T + sI) x */
    ierr = MatMultTranspose(st->A[0],x,y);CHKERRQ(ierr);
  }
  if (st->sigma != 0.0) {
    ierr = VecAXPY(y,st->sigma,x);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "STBackTransform_Shift"
PetscErrorCode STBackTransform_Shift(ST st,PetscInt n,PetscScalar *eigr,PetscScalar *eigi)
{
  PetscInt j;

  PetscFunctionBegin;
  for (j=0;j<n;j++) {
    eigr[j] -= st->sigma;
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "STSetUp_Shift"
PetscErrorCode STSetUp_Shift(ST st)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (st->nmat>1) {
    if (!st->ksp) { ierr = STGetKSP(st,&st->ksp);CHKERRQ(ierr); }
    ierr = KSPSetOperators(st->ksp,st->A[1],st->A[1],DIFFERENT_NONZERO_PATTERN);CHKERRQ(ierr);
    ierr = KSPSetUp(st->ksp);CHKERRQ(ierr);
  } 
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "STSetFromOptions_Shift"
PetscErrorCode STSetFromOptions_Shift(ST st) 
{
  PetscErrorCode ierr;
  PC             pc;
  const PCType   pctype;
  const KSPType  ksptype;

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

EXTERN_C_BEGIN
#undef __FUNCT__  
#define __FUNCT__ "STCreate_Shift"
PetscErrorCode STCreate_Shift(ST st)
{
  PetscFunctionBegin;
  st->ops->apply           = STApply_Shift;
  st->ops->getbilinearform = STGetBilinearForm_Default;
  st->ops->applytrans      = STApplyTranspose_Shift;
  st->ops->backtr          = STBackTransform_Shift;
  st->ops->setfromoptions  = STSetFromOptions_Shift;
  st->ops->setup           = STSetUp_Shift;
  PetscFunctionReturn(0);
}
EXTERN_C_END

