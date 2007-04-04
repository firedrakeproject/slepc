
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

.seealso: STApplyB()
@*/
PetscErrorCode STApply(ST st,Vec x,Vec y)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(st,ST_COOKIE,1);
  PetscValidHeaderSpecific(x,VEC_COOKIE,2);
  PetscValidHeaderSpecific(y,VEC_COOKIE,3);
  if (x == y) SETERRQ(PETSC_ERR_ARG_IDN,"x and y must be different vectors");

  if (!st->setupcalled) { ierr = STSetUp(st); CHKERRQ(ierr); }

  ierr = PetscLogEventBegin(ST_Apply,st,x,y,0);CHKERRQ(ierr);
  st->applys++;
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

.seealso: STApply()
@*/
PetscErrorCode STApplyB(ST st,Vec x,Vec y)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(st,ST_COOKIE,1);
  PetscValidHeaderSpecific(x,VEC_COOKIE,2);
  PetscValidHeaderSpecific(y,VEC_COOKIE,3);
  if (x == y) SETERRQ(PETSC_ERR_ARG_IDN,"x and y must be different vectors");

  if (!st->setupcalled) { ierr = STSetUp(st); CHKERRQ(ierr); }

  if (x->id == st->xid && x->state == st->xstate) {
    ierr = VecCopy(st->Bx, y);CHKERRQ(ierr);
    PetscFunctionReturn(0);
  }

  ierr = PetscLogEventBegin(ST_ApplyB,st,x,y,0);CHKERRQ(ierr);
  ierr = (*st->ops->applyB)(st,x,y);CHKERRQ(ierr);
  ierr = PetscLogEventEnd(ST_ApplyB,st,x,y,0);CHKERRQ(ierr);
  
  st->xid = x->id;
  st->xstate = x->state;
  ierr = VecCopy(y,st->Bx);CHKERRQ(ierr);  
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "STApplyB_Default"
PetscErrorCode STApplyB_Default(ST st,Vec x,Vec y)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if( st->B ) {
    ierr = MatMult( st->B, x, y ); CHKERRQ(ierr);
  }
  else {
    ierr = VecCopy( x, y ); CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "STApplyTranspose"
/*@
   STApplyTranspose - Applies the transpose of the operator to a vector, for
   instance B^T(A - sB)^-T in the case of the shift-and-invert tranformation
   and generalized eigenproblem.

   Collective on ST and Vec

   Input Parameters:
+  st - the spectral transformation context
-  x  - input vector

   Output Parameter:
.  y - output vector

   Level: developer

.seealso: STApply()
@*/
PetscErrorCode STApplyTranspose(ST st,Vec x,Vec y)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(st,ST_COOKIE,1);
  PetscValidHeaderSpecific(x,VEC_COOKIE,2);
  PetscValidHeaderSpecific(y,VEC_COOKIE,3);
  if (x == y) SETERRQ(PETSC_ERR_ARG_IDN,"x and y must be different vectors");

  if (!st->setupcalled) { ierr = STSetUp(st); CHKERRQ(ierr); }

  ierr = PetscLogEventBegin(ST_ApplyTranspose,st,x,y,0);CHKERRQ(ierr);
  ierr = (*st->ops->applytrans)(st,x,y);CHKERRQ(ierr);
  ierr = PetscLogEventEnd(ST_ApplyTranspose,st,x,y,0);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "STComputeExplicitOperator"
/*@
    STComputeExplicitOperator - Computes the explicit operator associated
    to the eigenvalue problem with the specified spectral transformation.  

    Collective on ST

    Input Parameter:
.   st - the spectral transform context

    Output Parameter:
.   mat - the explicit operator

    Notes:
    This routine builds a matrix containing the explicit operator. For 
    example, in generalized problems with shift-and-invert spectral
    transformation the result would be matrix (A - s B)^-1 B.
    
    This computation is done by applying the operator to columns of the 
    identity matrix. Note that the result is a dense matrix.

    Level: advanced

.seealso: STApply()   
@*/
PetscErrorCode STComputeExplicitOperator(ST st,Mat *mat)
{
  PetscErrorCode ierr;
  Vec            in,out;
  PetscInt       i,M,m,*rows,start,end;
  PetscScalar    *array,one = 1.0;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(st,ST_COOKIE,1);
  PetscValidPointer(mat,2);

  ierr = MatGetVecs(st->A,&in,&out);CHKERRQ(ierr);
  ierr = VecGetSize(out,&M);CHKERRQ(ierr);
  ierr = VecGetLocalSize(out,&m);CHKERRQ(ierr);
  ierr = VecGetOwnershipRange(out,&start,&end);CHKERRQ(ierr);
  ierr = PetscMalloc(m*sizeof(int),&rows);CHKERRQ(ierr);
  for (i=0; i<m; i++) rows[i] = start + i;

  ierr = MatCreateMPIDense(st->comm,m,m,M,M,PETSC_NULL,mat);CHKERRQ(ierr);

  for (i=0; i<M; i++) {
    ierr = VecSet(in,0.0);CHKERRQ(ierr);
    ierr = VecSetValues(in,1,&i,&one,INSERT_VALUES);CHKERRQ(ierr);
    ierr = VecAssemblyBegin(in);CHKERRQ(ierr);
    ierr = VecAssemblyEnd(in);CHKERRQ(ierr);

    ierr = STApply(st,in,out); CHKERRQ(ierr);
    
    ierr = VecGetArray(out,&array);CHKERRQ(ierr);
    ierr = MatSetValues(*mat,m,rows,1,&i,array,INSERT_VALUES);CHKERRQ(ierr); 
    ierr = VecRestoreArray(out,&array);CHKERRQ(ierr);
  }
  ierr = PetscFree(rows);CHKERRQ(ierr);
  ierr = VecDestroy(in);CHKERRQ(ierr);
  ierr = VecDestroy(out);CHKERRQ(ierr);
  ierr = MatAssemblyBegin(*mat,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(*mat,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
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
PetscErrorCode STSetUp(ST st)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(st,ST_COOKIE,1);

  PetscInfo(st,"Setting up new ST\n");
  if (st->setupcalled) PetscFunctionReturn(0);
  ierr = PetscLogEventBegin(ST_SetUp,st,0,0,0);CHKERRQ(ierr);
  if (!st->A) {SETERRQ(PETSC_ERR_ARG_WRONGSTATE,"Matrix must be set first");}
  if (!st->type_name) {
    ierr = STSetType(st,STSHIFT);CHKERRQ(ierr);
  }
  if (st->w) { ierr = VecDestroy(st->w);CHKERRQ(ierr); }
  if (st->Bx) { ierr = VecDestroy(st->Bx);CHKERRQ(ierr); }
  ierr = MatGetVecs(st->A,&st->w,&st->Bx);CHKERRQ(ierr);
  st->xid = 0; 
  if (st->ops->setup) {
    ierr = (*st->ops->setup)(st); CHKERRQ(ierr);
  }
  st->setupcalled = 1;
  ierr = PetscLogEventEnd(ST_SetUp,st,0,0,0);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "STPostSolve"
/*@
   STPostSolve - Optional post-solve phase, intended for any actions that must 
   be performed on the ST object after the eigensolver has finished.

   Collective on ST

   Input Parameters:
.  st  - the spectral transformation context

   Level: developer

.seealso: EPSSolve()
@*/
PetscErrorCode STPostSolve(ST st)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(st,ST_COOKIE,1);
  if (st->ops->postsolve) {
    ierr = (*st->ops->postsolve)(st);CHKERRQ(ierr);
  }

  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "STBackTransform"
/*@
   STBackTransform - Back-transformation phase, intended for 
   spectral transformations which require to transform the computed 
   eigenvalues back to the original eigenvalue problem.

   Collective on ST

   Input Parameters:
   st   - the spectral transformation context
   eigr - real part of a computed eigenvalue
   eigi - imaginary part of a computed eigenvalue

   Level: developer

.seealso: EPSBackTransform()
@*/
PetscErrorCode STBackTransform(ST st,PetscScalar* eigr,PetscScalar* eigi)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(st,ST_COOKIE,1);
  if (st->ops->backtr) {
    ierr = (*st->ops->backtr)(st,eigr,eigi);CHKERRQ(ierr);
  }

  PetscFunctionReturn(0);
}
