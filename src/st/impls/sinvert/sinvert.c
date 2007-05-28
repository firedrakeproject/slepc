/*
      Implements the shift-and-invert technique for eigenvalue problems.
*/
#include "src/st/stimpl.h"          /*I "slepcst.h" I*/

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
PetscErrorCode STBackTransform_Sinvert(ST st,PetscScalar *eigr,PetscScalar *eigi)
{
#ifndef PETSC_USE_COMPLEX
  PetscScalar t;
  PetscFunctionBegin;
  PetscValidPointer(eigr,2);
  PetscValidPointer(eigi,3);
  if (*eigi == 0) *eigr = 1.0 / *eigr + st->sigma;
  else {
    t = *eigr * *eigr + *eigi * *eigi;
    *eigr = *eigr / t + st->sigma;
    *eigi = - *eigi / t;
  }
#else
  PetscFunctionBegin;
  PetscValidPointer(eigr,2);
  *eigr = 1.0 / *eigr + st->sigma;
#endif
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "STPostSolve_Sinvert"
PetscErrorCode STPostSolve_Sinvert(ST st)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (st->shift_matrix == STMATMODE_INPLACE) {
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
  if (st->mat) { ierr = MatDestroy(st->mat);CHKERRQ(ierr); }

  switch (st->shift_matrix) {
  case STMATMODE_INPLACE:
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
  case STMATMODE_SHELL:
    ierr = STMatShellCreate(st,&st->mat);CHKERRQ(ierr);
    ierr = KSPSetOperators(st->ksp,st->mat,st->mat,DIFFERENT_NONZERO_PATTERN);CHKERRQ(ierr);
    break;
  default:
    ierr = MatDuplicate(st->A,MAT_COPY_VALUES,&st->mat);CHKERRQ(ierr);
    if (st->sigma != 0.0) {
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
  case STMATMODE_INPLACE:
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
  case STMATMODE_SHELL:
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
  
  st->checknullspace      = STCheckNullSpace_Default;

  PetscFunctionReturn(0);
}
EXTERN_C_END

