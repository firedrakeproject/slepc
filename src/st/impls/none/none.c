/*
    Identity transformation, simply applies the matrix A as operator
    in the case of standard eigenproblems, or B^-1A in the case of
    generalized eigenproblems
*/
#include "src/st/stimpl.h"          /*I "slepcst.h" I*/

#undef __FUNCT__  
#define __FUNCT__ "STApply_None"
int STApply_None(ST st,Vec x,Vec y)
{
  int    ierr;
  Vec    w;

  PetscFunctionBegin;
  if (st->B) {
    /* generalized eigenproblem: y = B^-1 A x */
    w = (Vec) st->data;
    ierr = MatMult(st->A,x,w);CHKERRQ(ierr);
    ierr = STAssociatedSLESSolve(st,w,y);CHKERRQ(ierr);
  }
  else {
    /* standard eigenproblem: y = A x */
    ierr = MatMult(st->A,x,y);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "STSetUp_None"
static int STSetUp_None(ST st)
{
  int     ierr;
  Vec     w;

  PetscFunctionBegin;
  if (st->sles) {
    ierr = VecDuplicate(st->vec,&w);CHKERRQ(ierr);
    st->data = (void *) w;
    ierr = SLESSetUp(st->sles,st->vec,st->vec);CHKERRQ(ierr);
  } 
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "STDestroy_None"
static int STDestroy_None(ST st)
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
#define __FUNCT__ "STCreate_None"
int STCreate_None(ST st)
{
  int     ierr;
  char    *prefix;

  PetscFunctionBegin;
  st->numberofshifts   = 0;
  st->ops->apply       = STApply_None;
  st->ops->destroy     = STDestroy_None;
  st->ops->setup       = STSetUp_None;

  if (st->B) {
    ierr = SLESCreate(st->comm,&st->sles);CHKERRQ(ierr);
    ierr = STGetOptionsPrefix(st,&prefix);CHKERRQ(ierr);
    ierr = SLESSetOptionsPrefix(st->sles,prefix);CHKERRQ(ierr);
    ierr = SLESAppendOptionsPrefix(st->sles,"st_");CHKERRQ(ierr);
    ierr = SLESSetOperators(st->sles,st->B,st->B,DIFFERENT_NONZERO_PATTERN);CHKERRQ(ierr);
  }

  PetscFunctionReturn(0);
}
EXTERN_C_END
