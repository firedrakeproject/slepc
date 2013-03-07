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
    /* generalized eigenproblem: y = B^-1 (A + sB) x */
    ierr = MatMult(st->T[0],x,st->w);CHKERRQ(ierr);
    ierr = STMatSolve(st,1,st->w,y);CHKERRQ(ierr);
  } else {
    /* standard eigenproblem: y = (A + sI) x */
    ierr = MatMult(st->T[0],x,y);CHKERRQ(ierr);
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
    /* generalized eigenproblem: y = (A + sB)^T B^-T  x */
    ierr = STMatSolveTranspose(st,1,x,st->w);CHKERRQ(ierr);
    ierr = MatMultTranspose(st->T[0],st->w,y);CHKERRQ(ierr);
  } else {
    /* standard eigenproblem: y = (A^T + sI) x */
    ierr = MatMultTranspose(st->T[0],x,y);CHKERRQ(ierr);
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
#define __FUNCT__ "STPostSolve_Shift"
PetscErrorCode STPostSolve_Shift(ST st)
{
  PetscErrorCode ierr;
  PetscScalar    s;

  PetscFunctionBegin;
  if (st->shift_matrix == ST_MATMODE_INPLACE) {
    if (st->nmat>1) {
      if (st->nmat==3) {
        ierr = MatAXPY(st->A[0],-st->sigma*st->sigma,st->A[2],st->str);CHKERRQ(ierr);
        ierr = MatAXPY(st->A[1],2.0*st->sigma,st->A[2],st->str);CHKERRQ(ierr);
        s = st->sigma;
      } else s = -st->sigma;
      ierr = MatAXPY(st->A[0],s,st->A[1],st->str);CHKERRQ(ierr);
    } else {
      ierr = MatShift(st->A[0],st->sigma);CHKERRQ(ierr);
    }
    st->Astate[0] = ((PetscObject)st->A[0])->state;
    st->setupcalled = 0;
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "STSetUp_Shift"
PetscErrorCode STSetUp_Shift(ST st)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (st->nmat<3) {
    /* T[1] = B */
    if (st->nmat>1) { ierr = PetscObjectReference((PetscObject)st->A[1]);CHKERRQ(ierr); }
    st->T[1] = st->A[1];
    /* T[0] = A+sigma*B  */
    ierr = STMatGAXPY_Private(st,st->sigma,0.0,1,0,PETSC_TRUE);CHKERRQ(ierr); 
  } else {
    /* T[2] = C */
    ierr = PetscObjectReference((PetscObject)st->A[2]);CHKERRQ(ierr);
    st->T[2] = st->A[2];
    /* T[0] = A-sigma*B+sigma*sigma*C */
    ierr = STMatGAXPY_Private(st,-st->sigma,0.0,2,0,PETSC_TRUE);CHKERRQ(ierr);
    /* T[1] = B-2*sigma*C  */
    ierr = STMatGAXPY_Private(st,-2.0*st->sigma,0.0,1,1,PETSC_TRUE);CHKERRQ(ierr);  
  }
  if (st->nmat==2) {
    if (!st->ksp) { ierr = STGetKSP(st,&st->ksp);CHKERRQ(ierr); }
    ierr = KSPSetOperators(st->ksp,st->T[1],st->T[1],DIFFERENT_NONZERO_PATTERN);CHKERRQ(ierr);
    ierr = KSPSetUp(st->ksp);CHKERRQ(ierr);
    st->kspidx = 1;
  }  
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "STSetShift_Shift"
PetscErrorCode STSetShift_Shift(ST st,PetscScalar newshift)
{
  PetscErrorCode ierr;
  MatStructure   flg;

  PetscFunctionBegin;
  /* Nothing to be done if STSetUp has not been called yet */
  if (!st->setupcalled) PetscFunctionReturn(0);

  if (st->nmat<3) {
    ierr = STMatGAXPY_Private(st,newshift,st->sigma,1,0,PETSC_FALSE);CHKERRQ(ierr);
  } else {
    ierr = STMatGAXPY_Private(st,-newshift,-st->sigma,2,2,PETSC_FALSE);CHKERRQ(ierr);
    ierr = STMatGAXPY_Private(st,-2.0*newshift,-2.0*st->sigma,1,1,PETSC_FALSE);CHKERRQ(ierr);
  }

  if (st->kspidx==0 || (st->nmat==3 && st->kspidx==1)) {  /* Update KSP operator */
    /* Check if the new KSP matrix has the same zero structure */
    if (st->nmat>1 && st->str == DIFFERENT_NONZERO_PATTERN && (st->sigma == 0.0 || newshift == 0.0)) flg = DIFFERENT_NONZERO_PATTERN;
    else flg = SAME_NONZERO_PATTERN;
    ierr = KSPSetOperators(st->ksp,st->T[st->kspidx],st->T[st->kspidx],flg);CHKERRQ(ierr);    
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
#define __FUNCT__ "STCreate_Shift"
PETSC_EXTERN PetscErrorCode STCreate_Shift(ST st)
{
  PetscFunctionBegin;
  st->ops->apply           = STApply_Shift;
  st->ops->getbilinearform = STGetBilinearForm_Default;
  st->ops->applytrans      = STApplyTranspose_Shift;
  st->ops->postsolve       = STPostSolve_Shift;
  st->ops->backtransform   = STBackTransform_Shift;
  st->ops->setfromoptions  = STSetFromOptions_Shift;
  st->ops->setup           = STSetUp_Shift;
  st->ops->setshift        = STSetShift_Shift;
  PetscFunctionReturn(0);
}

