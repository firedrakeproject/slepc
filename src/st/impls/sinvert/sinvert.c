/*
      Implements the shift-and-invert technique for eigenvalue problems.
*/
#include "src/st/stimpl.h"          /*I "slepcst.h" I*/

#undef __FUNCT__  
#define __FUNCT__ "STApply_Sinvert"
static int STApply_Sinvert(ST st,Vec x,Vec y)
{
  int       ierr;

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
#define __FUNCT__ "STApplyNoB_Sinvert"
static int STApplyNoB_Sinvert(ST st,Vec x,Vec y)
{
  int       ierr;

  PetscFunctionBegin;
  ierr = STAssociatedKSPSolve(st,x,y);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "STBackTransform_Sinvert"
int STBackTransform_Sinvert(ST st,PetscScalar *eigr,PetscScalar *eigi)
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
#define __FUNCT__ "STPost_Sinvert"
int STPost_Sinvert(ST st)
{
  PetscScalar  alpha;
  int          ierr;

  PetscFunctionBegin;
  if (st->shift_matrix == STMATMODE_INPLACE) {
    alpha = st->sigma;
    if( st->B ) { ierr = MatAXPY(&alpha,st->B,st->A,st->str);CHKERRQ(ierr); }
    else { ierr = MatShift( &alpha, st->A ); CHKERRQ(ierr); }
    st->setupcalled = 0;
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "STSetUp_Sinvert"
static int STSetUp_Sinvert(ST st)
{
  int          ierr;
  PetscScalar  alpha;

  PetscFunctionBegin;

  switch (st->shift_matrix) {
  case STMATMODE_INPLACE:
    if (st->sigma != 0.0) {
      alpha = -st->sigma;
      if (st->B) { 
        ierr = MatAXPY(&alpha,st->B,st->A,st->str);CHKERRQ(ierr); 
      } else { 
        ierr = MatShift(&alpha,st->A);CHKERRQ(ierr); 
      }
    }
    /* In the following line, the SAME_NONZERO_PATTERN flag has been used to
     * improve performance when solving a number of related eigenproblems */
    ierr = KSPSetOperators(st->ksp,st->A,st->A,SAME_NONZERO_PATTERN);CHKERRQ(ierr);
    break;
  case STMATMODE_SHELL:
    ierr = STMatShellCreate(st,&st->mat);CHKERRQ(ierr);
    ierr = KSPSetOperators(st->ksp,st->mat,st->mat,DIFFERENT_NONZERO_PATTERN);CHKERRQ(ierr);
    break;
  default:
    ierr = MatDuplicate(st->A,MAT_COPY_VALUES,&st->mat);CHKERRQ(ierr);
    if (st->sigma != 0.0) {
      alpha = -st->sigma;
      if (st->B) { 
        ierr = MatAXPY(&alpha,st->B,st->mat,st->str);CHKERRQ(ierr); 
      } else { 
        ierr = MatShift(&alpha,st->mat);CHKERRQ(ierr); 
      }
    }
    /* In the following line, the SAME_NONZERO_PATTERN flag has been used to
     * improve performance when solving a number of related eigenproblems */
    ierr = KSPSetOperators(st->ksp,st->mat,st->mat,SAME_NONZERO_PATTERN);CHKERRQ(ierr);
  }
  if (st->B) { 
    if (st->w) { ierr = VecDestroy(st->w);CHKERRQ(ierr); }
    ierr = MatGetVecs(st->B,&st->w,PETSC_NULL);CHKERRQ(ierr); 
  } 
  ierr = KSPSetUp(st->ksp);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "STSetShift_Sinvert"
static int STSetShift_Sinvert(ST st,PetscScalar newshift)
{
  int          ierr;
  PetscScalar  alpha;

  PetscFunctionBegin;

  /* Nothing to be done if STSetUp has not been called yet */
  if (!st->setupcalled) PetscFunctionReturn(0);

  switch (st->shift_matrix) {
  case STMATMODE_INPLACE:
    /* Undo previous operations */
    if (st->sigma != 0.0) {
      alpha = st->sigma;
      if (st->B) { ierr = MatAXPY(&alpha,st->B,st->A,st->str);CHKERRQ(ierr); }
      else { ierr = MatShift(&alpha,st->A);CHKERRQ(ierr); }
    }
    /* Apply new shift */
    if (newshift != 0.0) {
      alpha = -newshift;
      if (st->B) { ierr = MatAXPY(&alpha,st->B,st->A,st->str);CHKERRQ(ierr); }
      else { ierr = MatShift(&alpha,st->A);CHKERRQ(ierr); }
    }
    ierr = KSPSetOperators(st->ksp,st->A,st->A,SAME_NONZERO_PATTERN);CHKERRQ(ierr);
    break;
  case STMATMODE_SHELL:
    break;
  default:
    ierr = MatCopy(st->A, st->mat, DIFFERENT_NONZERO_PATTERN); CHKERRQ(ierr);
    if (newshift != 0.0) {   
      alpha = -newshift;
      if (st->B) { ierr = MatAXPY(&alpha,st->B,st->mat,st->str);CHKERRQ(ierr); }
      else { ierr = MatShift(&alpha,st->mat);CHKERRQ(ierr); }
    }
    /* In the following line, the SAME_NONZERO_PATTERN flag has been used to
     * improve performance when solving a number of related eigenproblems */
    ierr = KSPSetOperators(st->ksp,st->mat,st->mat,SAME_NONZERO_PATTERN);CHKERRQ(ierr);    
  }
  st->sigma = newshift;
  ierr = KSPSetUp(st->ksp);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

EXTERN_C_BEGIN
#undef __FUNCT__  
#define __FUNCT__ "STCreate_Sinvert"
int STCreate_Sinvert(ST st)
{
  PetscFunctionBegin;
  st->numberofshifts      = 1;
  st->data                = 0;

  st->ops->apply          = STApply_Sinvert;
  st->ops->applyB         = STApplyB_Default;
  st->ops->applynoB       = STApplyNoB_Sinvert;
  st->ops->postsolve      = STPost_Sinvert;
  st->ops->backtr         = STBackTransform_Sinvert;
  st->ops->setup          = STSetUp_Sinvert;
  st->ops->setshift       = STSetShift_Sinvert;
  
  st->checknullspace      = STCheckNullSpace_Default;

  PetscFunctionReturn(0);
}
EXTERN_C_END

