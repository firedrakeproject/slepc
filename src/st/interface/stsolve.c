
/*
    The ST (spectral transformation) interface routines, callable by users.
*/

#include "src/st/stimpl.h"            /*I "slepcst.h" I*/

#undef __FUNCT__  
#define __FUNCT__ "STApply"
/*@
   STApply - Applies the spectral transformation operator to a vector, for
   instance (A - sB)^-1 B in the case of the shift-and-invert tranformation
   and generalized eigenproblem.

   Collective on ST and Vec

   Input Parameters:
+  st - the spectral transformation context
-  x  - input vector

   Output Parameter:
.  y - output vector

   Level: developer

.seealso: STApplyB(), STApplyNoB()
@*/
int STApply(ST st,Vec x,Vec y)
{
  int        ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(st,ST_COOKIE,1);
  PetscValidHeaderSpecific(x,VEC_COOKIE,2);
  PetscValidHeaderSpecific(y,VEC_COOKIE,3);
  if (x == y) SETERRQ(PETSC_ERR_ARG_IDN,"x and y must be different vectors");

  if (!st->setupcalled) { ierr = STSetUp(st); CHKERRQ(ierr); }

  ierr = PetscLogEventBegin(ST_Apply,st,x,y,0);CHKERRQ(ierr);
  ierr = (*st->ops->apply)(st,x,y);CHKERRQ(ierr);
  ierr = PetscLogEventEnd(ST_Apply,st,x,y,0);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "STApplyB"
/*@
   STApplyB - Applies the B matrix to a vector.

   Collective on ST and Vec

   Input Parameters:
+  st - the spectral transformation context
-  x - input vector

   Output Parameter:
.  y - output vector

   Level: developer

.seealso: STApply(), STApplyNoB()
@*/
int STApplyB(ST st,Vec x,Vec y)
{
  int        ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(st,ST_COOKIE,1);
  PetscValidHeaderSpecific(x,VEC_COOKIE,2);
  PetscValidHeaderSpecific(y,VEC_COOKIE,3);
  if (x == y) SETERRQ(PETSC_ERR_ARG_IDN,"x and y must be different vectors");

  if (!st->setupcalled) { ierr = STSetUp(st); CHKERRQ(ierr); }

  ierr = PetscLogEventBegin(ST_ApplyB,st,x,y,0);CHKERRQ(ierr);
  ierr = (*st->ops->applyB)(st,x,y);CHKERRQ(ierr);
  ierr = PetscLogEventEnd(ST_ApplyB,st,x,y,0);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "STApplyNoB"
/*@
   STApplyNoB - Applies the spectral transformation operator to a vector 
   which has already been multiplied by matrix B. For instance, this routine
   would perform the operation y =(A - sB)^-1 x in the case of the 
   shift-and-invert tranformation and generalized eigenproblem.

   Collective on ST and Vec

   Input Parameters:
+  st - the spectral transformation context
-  x  - input vector, where it is assumed that x=Bw for some vector w

   Output Parameter:
.  y - output vector

   Level: developer

.seealso: STApply(), STApplyB()
@*/
int STApplyNoB(ST st,Vec x,Vec y)
{
  int        ierr;
  PetscTruth isSinv;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(st,ST_COOKIE,1);
  PetscValidHeaderSpecific(x,VEC_COOKIE,2);
  PetscValidHeaderSpecific(y,VEC_COOKIE,3);
  if (x == y) SETERRQ(PETSC_ERR_ARG_IDN,"x and y must be different vectors");

  if (!st->setupcalled) { ierr = STSetUp(st); CHKERRQ(ierr); }

  ierr = PetscTypeCompare((PetscObject)st,STSINV,&isSinv);CHKERRQ(ierr);
  if (!isSinv) { SETERRQ(1,"Function only available in Shift-and-Invert"); }

  ierr = PetscLogEventBegin(ST_ApplyNoB,st,x,y,0);CHKERRQ(ierr);
  ierr = (*st->ops->applynoB)(st,x,y);CHKERRQ(ierr);
  ierr = PetscLogEventEnd(ST_ApplyNoB,st,x,y,0);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "STInnerProduct"
/*@
   STInnerProduct - Computes de inner product of two vectors.

   Collective on ST and Vec

   Input Parameters:
+  st - the spectral transformation context
.  x  - input vector
-  y  - input vector

   Output Parameter:
+  w - intermediate vector (see Notes below)
-  p - result of the inner product

   Notes:
   This function will usually compute the standard dot product of vectors
   x and y, (x,y)=y^H x. However this behaviour may be different if changed 
   via STSetBilinearForm(). This allows use of other inner products such as
   the indefinite product y^T x for complex symmetric problems or the
   B-inner product for positive definite B, (x,y)_B=y^H Bx.

   At the end of the execution, the intermediate vector w will hold x or Bx,
   depending on the type of inner product.

   Level: developer

.seealso: STSetBilinearForm(), STApplyB(), VecDot()
@*/
int STInnerProduct(ST st,Vec x,Vec y,Vec w,PetscScalar *p)
{
  int        ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(st,ST_COOKIE,1);
  PetscValidHeaderSpecific(x,VEC_COOKIE,2);
  PetscValidHeaderSpecific(y,VEC_COOKIE,3);
  PetscValidHeaderSpecific(w,VEC_COOKIE,4);
  PetscValidScalarPointer(p,5);
  
  ierr = PetscLogEventBegin(ST_InnerProduct,st,x,w,0);CHKERRQ(ierr);
  switch (st->bilinear_form) {
  case STINNER_HERMITIAN:
  case STINNER_SYMMETRIC:
    ierr = VecCopy(x,w);CHKERRQ(ierr);
    break;
  case STINNER_B_HERMITIAN:
  case STINNER_B_SYMMETRIC:
    ierr = STApplyB(st,x,w);CHKERRQ(ierr);
    break;
  }
  switch (st->bilinear_form) {
  case STINNER_HERMITIAN:
  case STINNER_B_HERMITIAN:
    ierr = VecDot(w,y,p);CHKERRQ(ierr);
    break;
  case STINNER_SYMMETRIC:
  case STINNER_B_SYMMETRIC:
    ierr = VecTDot(w,y,p);CHKERRQ(ierr);
    break;
  }  
  ierr = PetscLogEventEnd(ST_InnerProduct,st,x,w,0);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "STSetUp"
/*@
   STSetUp - Prepares for the use of a spectral transformation.

   Collective on ST

   Input Parameter:
.  st - the spectral transformation context

   Level: advanced

.seealso: STCreate(), STApply(), STDestroy()
@*/
int STSetUp(ST st)
{
  int ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(st,ST_COOKIE,1);

  PetscLogInfo(st,"STSetUp:Setting up new ST\n");
  if (st->setupcalled) PetscFunctionReturn(0);
  ierr = PetscLogEventBegin(ST_SetUp,st,0,0,0);CHKERRQ(ierr);
  if (!st->A) {SETERRQ(PETSC_ERR_ARG_WRONGSTATE,"Matrix must be set first");}
  if (!st->type_name) {
    ierr = STSetType(st,STSHIFT);CHKERRQ(ierr);
  }
  if (st->ops->setup) {
    ierr = (*st->ops->setup)(st); CHKERRQ(ierr);
  }
  st->setupcalled = 1;
  ierr = PetscLogEventEnd(ST_SetUp,st,0,0,0);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "STPreSolve"
/*
   STPreSolve - Optional pre-solve phase, intended for any actions that 
   must be performed on the ST object before the eigensolver starts iterating.

   Collective on ST

   Input Parameters:
   st  - the spectral transformation context
   eps - the eigenproblem solver context

   Level: developer

   Sample of Usage:

    STPreSolve(st,eps);
    EPSSolve(eps,its);
    STPostSolve(st,eps);
*/
int STPreSolve(ST st,EPS eps)
{
  int         ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(st,ST_COOKIE,1);

  if (st->ops->presolve) {
    ierr = (*st->ops->presolve)(st);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "STPostSolve"
/*
   STPostSolve - Optional post-solve phase, intended for any actions that must 
   be performed on the ST object after the eigensolver has finished.

   Collective on ST

   Input Parameters:
   st  - the spectral transformation context
   eps - the eigenproblem solver context

   Sample of Usage:

    STPreSolve(st,eps);
    EPSSolve(eps,its);
    STPostSolve(st,eps);
*/
int STPostSolve(ST st,EPS eps)
{
  int         ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(st,ST_COOKIE,1);
  if (st->ops->postsolve) {
    ierr = (*st->ops->postsolve)(st);CHKERRQ(ierr);
  }

  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "STBackTransform"
/*
   STBackTransform - Optional back-transformation phase, intended for 
   spectral transformation which require to transform the computed 
   eigenvalues back to the original eigenvalue problem.

   Collective on ST

   Input Parameters:
   st   - the spectral transformation context
   eigr - real part of a computed eigenvalue
   eigi - imaginary part of a computed eigenvalue

   Level: developer

.seealso: EPSBackTransform()
*/
int STBackTransform(ST st,PetscScalar* eigr,PetscScalar* eigi)
{
  int         ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(st,ST_COOKIE,1);
  if (st->ops->backtr) {
    ierr = (*st->ops->backtr)(st,eigr,eigi);CHKERRQ(ierr);
  }

  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "STApplyB_Default"
int STApplyB_Default(ST st,Vec x,Vec y)
{
  int        ierr;

  PetscFunctionBegin;
  if( st->B ) {
    ierr = MatMult( st->B, x, y ); CHKERRQ(ierr);
  }
  else {
    ierr = VecCopy( x, y ); CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}
