/*
    Shift spectral transformation, applies (A + sigma I) as operator, or 
    inv(B)(A + sigma B) for generalized problems

   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
   SLEPc - Scalable Library for Eigenvalue Problem Computations
   Copyright (c) 2002-2009, Universidad Politecnica de Valencia, Spain

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

#include "private/stimpl.h"          /*I "slepcst.h" I*/

#undef __FUNCT__  
#define __FUNCT__ "STApply_Shift"
PetscErrorCode STApply_Shift(ST st,Vec x,Vec y)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (st->B) {
    /* generalized eigenproblem: y = (B^-1 A + sI) x */
    ierr = MatMult(st->A,x,st->w);CHKERRQ(ierr);
    ierr = STAssociatedKSPSolve(st,st->w,y);CHKERRQ(ierr);
  }
  else {
    /* standard eigenproblem: y = (A + sI) x */
    ierr = MatMult(st->A,x,y);CHKERRQ(ierr);
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
  if (st->B) {
    /* generalized eigenproblem: y = (A^T B^-T + sI) x */
    ierr = STAssociatedKSPSolveTranspose(st,x,st->w);CHKERRQ(ierr);
    ierr = MatMultTranspose(st->A,st->w,y);CHKERRQ(ierr);
  }
  else {
    /* standard eigenproblem: y = (A^T + sI) x */
    ierr = MatMultTranspose(st->A,x,y);CHKERRQ(ierr);
  }
  if (st->sigma != 0.0) {
    ierr = VecAXPY(y,st->sigma,x);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "STBackTransform_Shift"
PetscErrorCode STBackTransform_Shift(ST st,int n,PetscScalar *eigr,PetscScalar *eigi)
{
  PetscInt j;
  PetscFunctionBegin;
  PetscValidPointer(eigr,3);
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
  if (st->B) {
    ierr = KSPSetOperators(st->ksp,st->B,st->B,DIFFERENT_NONZERO_PATTERN);CHKERRQ(ierr);
    ierr = KSPSetUp(st->ksp);CHKERRQ(ierr);
  } 
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "STView_Shift"
PetscErrorCode STView_Shift(ST st,PetscViewer viewer)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (st->B) {
    ierr = STView_Default(st,viewer);CHKERRQ(ierr);
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
  st->ops->setup           = STSetUp_Shift;
  st->ops->view            = STView_Shift;
  st->checknullspace       = 0;
  PetscFunctionReturn(0);
}
EXTERN_C_END

