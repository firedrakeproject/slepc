/*
    Shift spectral transformation, applies (A + sigma I) as operator, or 
    inv(B)(A + sigma B) for generalized problems
*/
#include "src/st/stimpl.h"          /*I "slepcst.h" I*/

#undef __FUNCT__  
#define __FUNCT__ "STApply_Shift"
int STApply_Shift(ST st,Vec x,Vec y)
{
  int    ierr;

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
    ierr = VecAXPY(&st->sigma,x,y);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "STApplyB_Shift"
int STApplyB_Shift(ST st,Vec x,Vec y)
{
  int        ierr;

  PetscFunctionBegin;
  ierr = VecCopy( x, y ); CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "STBackTransform_Shift"
int STBackTransform_Shift(ST st,PetscScalar *eigr,PetscScalar *eigi)
{
  PetscFunctionBegin;
  if (eigr) *eigr -= st->sigma;
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "STSetUp_Shift"
static int STSetUp_Shift(ST st)
{
  int     ierr;

  PetscFunctionBegin;
  if (st->B) {
    ierr = KSPSetOperators(st->ksp,st->B,st->B,DIFFERENT_NONZERO_PATTERN);CHKERRQ(ierr);
    ierr = KSPSetUp(st->ksp);CHKERRQ(ierr);
  } 
  PetscFunctionReturn(0);
}

EXTERN_C_BEGIN
#undef __FUNCT__  
#define __FUNCT__ "STCreate_Shift"
int STCreate_Shift(ST st)
{
  PetscFunctionBegin;
  st->ops->apply       = STApply_Shift;
  st->ops->applyB      = STApplyB_Shift;
  st->ops->applynoB    = STApply_Shift;
  st->ops->backtr      = STBackTransform_Shift;
  st->ops->setup       = STSetUp_Shift;
  st->checknullspace   = 0;
  PetscFunctionReturn(0);
}
EXTERN_C_END

