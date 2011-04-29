/*
      Implements the shift-and-invert technique for eigenvalue problems.

   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
   SLEPc - Scalable Library for Eigenvalue Problem Computations
   Copyright (c) 2002-2010, Universidad Politecnica de Valencia, Spain

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

#include <private/stimpl.h>          /*I "slepcst.h" I*/

#undef __FUNCT__  
#define __FUNCT__ "STApply_Sinvert"
PetscErrorCode STApply_Sinvert(ST st,Vec x,Vec y)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (st->B) {
    /* generalized eigenproblem: y = (A - sB)^-1 B x */
    ierr = MatMult(st->B,x,st->w);CHKERRQ(ierr);
    ierr = STAssociatedKSPSolve(st,st->w,y);CHKERRQ(ierr);
  }
  else {
    /* standard eigenproblem: y = (A - sI)^-1 x */
    ierr = STAssociatedKSPSolve(st,x,y);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "STApplyTranspose_Sinvert"
PetscErrorCode STApplyTranspose_Sinvert(ST st,Vec x,Vec y)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (st->B) {
    /* generalized eigenproblem: y = B^T (A - sB)^-T x */
    ierr = STAssociatedKSPSolveTranspose(st,x,st->w);CHKERRQ(ierr);
    ierr = MatMultTranspose(st->B,st->w,y);CHKERRQ(ierr);
  }
  else {
    /* standard eigenproblem: y = (A - sI)^-T x */
    ierr = STAssociatedKSPSolveTranspose(st,x,y);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "STBackTransform_Sinvert"
PetscErrorCode STBackTransform_Sinvert(ST st,PetscInt n,PetscScalar *eigr,PetscScalar *eigi)
{
  PetscInt    j;
#if !defined(PETSC_USE_COMPLEX)
  PetscScalar t;
#endif

  PetscFunctionBegin;
  PetscValidPointer(eigr,3);
#if !defined(PETSC_USE_COMPLEX)
  PetscValidPointer(eigi,4);
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

#undef __FUNCT__  
#define __FUNCT__ "STPostSolve_Sinvert"
PetscErrorCode STPostSolve_Sinvert(ST st)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (st->shift_matrix == ST_MATMODE_INPLACE) {
    if( st->B ) {
      ierr = MatAXPY(st->A,st->sigma,st->B,st->str);CHKERRQ(ierr);
    } else {
      ierr = MatShift(st->A,st->sigma); CHKERRQ(ierr);
    }
    st->setupcalled = 0;
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "STSetUp_Sinvert"
PetscErrorCode STSetUp_Sinvert(ST st)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = MatDestroy(&st->mat);CHKERRQ(ierr);

  /* if the user did not set the shift, use the target value */
  if (!st->sigma_set) st->sigma = st->defsigma;

  switch (st->shift_matrix) {
  case ST_MATMODE_INPLACE:
    st->mat = PETSC_NULL;
    if (st->sigma != 0.0) {
      if (st->B) { 
        ierr = MatAXPY(st->A,-st->sigma,st->B,st->str);CHKERRQ(ierr); 
      } else { 
        ierr = MatShift(st->A,-st->sigma);CHKERRQ(ierr); 
      }
    }
    ierr = KSPSetOperators(st->ksp,st->A,st->A,DIFFERENT_NONZERO_PATTERN);CHKERRQ(ierr);
    break;
  case ST_MATMODE_SHELL:
    ierr = STMatShellCreate(st,&st->mat);CHKERRQ(ierr);
    ierr = KSPSetOperators(st->ksp,st->mat,st->mat,DIFFERENT_NONZERO_PATTERN);CHKERRQ(ierr);
    break;
  default:
    if (st->sigma != 0.0) {
      ierr = MatDuplicate(st->A,MAT_COPY_VALUES,&st->mat);CHKERRQ(ierr);
      if (st->B) { 
        ierr = MatAXPY(st->mat,-st->sigma,st->B,st->str);CHKERRQ(ierr); 
      } else { 
        ierr = MatShift(st->mat,-st->sigma);CHKERRQ(ierr); 
      }
      ierr = KSPSetOperators(st->ksp,st->mat,st->mat,DIFFERENT_NONZERO_PATTERN);CHKERRQ(ierr);
    } else {
      st->mat = PETSC_NULL;
      ierr = KSPSetOperators(st->ksp,st->A,st->A,DIFFERENT_NONZERO_PATTERN);CHKERRQ(ierr);
    }
  }

  ierr = KSPSetUp(st->ksp);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "STSetShift_Sinvert"
PetscErrorCode STSetShift_Sinvert(ST st,PetscScalar newshift)
{
  PetscErrorCode ierr;
  MatStructure   flg;

  PetscFunctionBegin;
  /* Nothing to be done if STSetUp has not been called yet */
  if (!st->setupcalled) PetscFunctionReturn(0);
  
  /* Check if the new KSP matrix has the same zero structure */
  if (st->B && st->str == DIFFERENT_NONZERO_PATTERN && (st->sigma == 0.0 || newshift == 0.0)) {
    flg = DIFFERENT_NONZERO_PATTERN;
  } else {
    flg = SAME_NONZERO_PATTERN;
  }

  switch (st->shift_matrix) {
  case ST_MATMODE_INPLACE:
    /* Undo previous operations */
    if (st->sigma != 0.0) {
      if (st->B) {
        ierr = MatAXPY(st->A,st->sigma,st->B,st->str);CHKERRQ(ierr);
      } else {
        ierr = MatShift(st->A,st->sigma);CHKERRQ(ierr);
      }
    }
    /* Apply new shift */
    if (newshift != 0.0) {
      if (st->B) {
        ierr = MatAXPY(st->A,-newshift,st->B,st->str);CHKERRQ(ierr);
      } else {
        ierr = MatShift(st->A,-newshift);CHKERRQ(ierr);
      }
    }
    ierr = KSPSetOperators(st->ksp,st->A,st->A,flg);CHKERRQ(ierr);
    break;
  case ST_MATMODE_SHELL:
    ierr = KSPSetOperators(st->ksp,st->mat,st->mat,DIFFERENT_NONZERO_PATTERN);CHKERRQ(ierr);    
    break;
  default:
    if (st->mat) {
      ierr = MatCopy(st->A,st->mat,SUBSET_NONZERO_PATTERN); CHKERRQ(ierr);
    } else {
      ierr = MatDuplicate(st->A,MAT_COPY_VALUES,&st->mat);CHKERRQ(ierr);
    }
    if (newshift != 0.0) {   
      if (st->B) {
        ierr = MatAXPY(st->mat,-newshift,st->B,st->str);CHKERRQ(ierr);
      } else {
        ierr = MatShift(st->mat,-newshift);CHKERRQ(ierr);
      }
    }
    ierr = KSPSetOperators(st->ksp,st->mat,st->mat,flg);CHKERRQ(ierr);    
  }
  st->sigma = newshift;
  ierr = KSPSetUp(st->ksp);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "STSetFromOptions_Sinvert"
PetscErrorCode STSetFromOptions_Sinvert(ST st) 
{
  PetscErrorCode ierr;
  PC             pc;
  const PCType   pctype;
  const KSPType  ksptype;

  PetscFunctionBegin;
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
#define __FUNCT__ "STCreate_Sinvert"
PetscErrorCode STCreate_Sinvert(ST st)
{
  PetscFunctionBegin;
  st->data                 = 0;
  st->ops->apply           = STApply_Sinvert;
  st->ops->getbilinearform = STGetBilinearForm_Default;
  st->ops->applytrans      = STApplyTranspose_Sinvert;
  st->ops->postsolve       = STPostSolve_Sinvert;
  st->ops->backtr          = STBackTransform_Sinvert;
  st->ops->setup           = STSetUp_Sinvert;
  st->ops->setshift        = STSetShift_Sinvert;
  st->ops->view            = STView_Default;
  st->ops->setfromoptions = STSetFromOptions_Sinvert;
  st->checknullspace      = STCheckNullSpace_Default;
  PetscFunctionReturn(0);
}
EXTERN_C_END

