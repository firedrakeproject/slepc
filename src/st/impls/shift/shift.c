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
  Vec    w;

  PetscFunctionBegin;
  if (st->B) {
    /* generalized eigenproblem: y = (B^-1 A + sI) x */
    w = (Vec) st->data;
    ierr = MatMult(st->A,x,w);CHKERRQ(ierr);
    ierr = STAssociatedKSPSolve(st,w,y);CHKERRQ(ierr);
  }
  else {
    /* standard eigenproblem: y = (A + sI) x */
    ierr = MatMult(st->A,x,y);CHKERRQ(ierr);
  }
  if (st->sigma != 0) {
    ierr = VecAXPY(&st->sigma,x,y);CHKERRQ(ierr);
  }
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
  Vec     w;

  PetscFunctionBegin;
  if (st->ksp) {
    ierr = VecDuplicate(st->vec,&w);CHKERRQ(ierr);
    st->data = (void *) w;
    ierr = KSPSetRhs(st->ksp,st->vec);CHKERRQ(ierr);
    ierr = KSPSetUp(st->ksp);CHKERRQ(ierr);
  } 
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "STDestroy_Shift"
static int STDestroy_Shift(ST st)
{
  int      ierr;
  Vec      w;

  PetscFunctionBegin;
  if (st->data) {
    w = (Vec) st->data;
    ierr = VecDestroy(w);CHKERRQ(ierr);
  } 
  PetscFunctionReturn(0);
}

EXTERN_C_BEGIN
#undef __FUNCT__  
#define __FUNCT__ "STCreate_Shift"
int STCreate_Shift(ST st)
{
  int     ierr;
  char    *prefix;

  PetscFunctionBegin;
  st->numberofshifts   = 1;
  st->ops->apply       = STApply_Shift;
  st->ops->backtr      = STBackTransform_Shift;
  st->ops->destroy     = STDestroy_Shift;
  st->ops->setup       = STSetUp_Shift;

  if (st->B) {
    ierr = KSPCreate(st->comm,&st->ksp);CHKERRQ(ierr);
    ierr = STGetOptionsPrefix(st,&prefix);CHKERRQ(ierr);
    ierr = KSPSetOptionsPrefix(st->ksp,prefix);CHKERRQ(ierr);
    ierr = KSPAppendOptionsPrefix(st->ksp,"st_");CHKERRQ(ierr);
    ierr = KSPSetOperators(st->ksp,st->B,st->B,DIFFERENT_NONZERO_PATTERN);CHKERRQ(ierr);
  }

  PetscFunctionReturn(0);
}
EXTERN_C_END

