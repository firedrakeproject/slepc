/*
    The ST (spectral transformation) interface routines, callable by users.

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

#include <private/stimpl.h>            /*I "slepcst.h" I*/

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

.seealso: STApplyTranspose()
@*/
PetscErrorCode STApply(ST st,Vec x,Vec y)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(st,ST_CLASSID,1);
  PetscValidHeaderSpecific(x,VEC_CLASSID,2);
  PetscValidHeaderSpecific(y,VEC_CLASSID,3);
  if (x == y) SETERRQ(((PetscObject)st)->comm,PETSC_ERR_ARG_IDN,"x and y must be different vectors");

  if (!st->setupcalled) { ierr = STSetUp(st); CHKERRQ(ierr); }

  ierr = PetscLogEventBegin(ST_Apply,st,x,y,0);CHKERRQ(ierr);
  st->applys++;
  if (st->D) { /* with balancing */
    ierr = VecPointwiseDivide(st->wb,x,st->D);CHKERRQ(ierr);
    ierr = (*st->ops->apply)(st,st->wb,y);CHKERRQ(ierr);
    ierr = VecPointwiseMult(y,y,st->D);CHKERRQ(ierr);
  }
  else {
    ierr = (*st->ops->apply)(st,x,y);CHKERRQ(ierr);
  }
  ierr = PetscLogEventEnd(ST_Apply,st,x,y,0);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "STGetBilinearForm"
/*@
   STGetBilinearForm - Returns the matrix used in the bilinear form with a 
   generalized problem with semi-definite B.

   Collective on ST and Mat

   Input Parameters:
.  st - the spectral transformation context

   Output Parameter:
.  B - output matrix

   Note:
   The output matrix B must be destroyed after use.
   
   Level: developer
@*/
PetscErrorCode STGetBilinearForm(ST st,Mat *B)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(st,ST_CLASSID,1);
  PetscValidPointer(B,2);
  ierr = (*st->ops->getbilinearform)(st,B);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "STGetBilinearForm_Default"
PetscErrorCode STGetBilinearForm_Default(ST st,Mat *B)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  *B = st->B;
  if (*B) {
    ierr =  PetscObjectReference((PetscObject)*B);CHKERRQ(ierr);
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
  PetscValidHeaderSpecific(st,ST_CLASSID,1);
  PetscValidHeaderSpecific(x,VEC_CLASSID,2);
  PetscValidHeaderSpecific(y,VEC_CLASSID,3);
  if (x == y) SETERRQ(((PetscObject)st)->comm,PETSC_ERR_ARG_IDN,"x and y must be different vectors");

  if (!st->setupcalled) { ierr = STSetUp(st); CHKERRQ(ierr); }

  ierr = PetscLogEventBegin(ST_ApplyTranspose,st,x,y,0);CHKERRQ(ierr);
  st->applys++;
  if (st->D) { /* with balancing */
    ierr = VecPointwiseMult(st->wb,x,st->D);CHKERRQ(ierr);
    ierr = (*st->ops->applytrans)(st,st->wb,y);CHKERRQ(ierr);
    ierr = VecPointwiseDivide(y,y,st->D);CHKERRQ(ierr);
  }
  else {
    ierr = (*st->ops->applytrans)(st,x,y);CHKERRQ(ierr);
  }
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
.  st - the spectral transform context

   Output Parameter:
.  mat - the explicit operator

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
  PetscValidHeaderSpecific(st,ST_CLASSID,1);
  PetscValidPointer(mat,2);

  ierr = MatGetVecs(st->A,&in,&out);CHKERRQ(ierr);
  ierr = VecGetSize(out,&M);CHKERRQ(ierr);
  ierr = VecGetLocalSize(out,&m);CHKERRQ(ierr);
  ierr = VecGetOwnershipRange(out,&start,&end);CHKERRQ(ierr);
  ierr = PetscMalloc(m*sizeof(PetscInt),&rows);CHKERRQ(ierr);
  for (i=0; i<m; i++) rows[i] = start + i;

  ierr = MatCreateMPIDense(((PetscObject)st)->comm,m,m,M,M,PETSC_NULL,mat);CHKERRQ(ierr);

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
  ierr = VecDestroy(&in);CHKERRQ(ierr);
  ierr = VecDestroy(&out);CHKERRQ(ierr);
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
  PetscValidHeaderSpecific(st,ST_CLASSID,1);
  PetscInfo(st,"Setting up new ST\n");
  if (st->setupcalled) PetscFunctionReturn(0);
  ierr = PetscLogEventBegin(ST_SetUp,st,0,0,0);CHKERRQ(ierr);
  if (!st->A) {SETERRQ(((PetscObject)st)->comm,PETSC_ERR_ARG_WRONGSTATE,"Matrix must be set first");}
  if (!((PetscObject)st)->type_name) {
    ierr = STSetType(st,STSHIFT);CHKERRQ(ierr);
  }
  ierr = VecDestroy(&st->w);CHKERRQ(ierr);
  ierr = MatGetVecs(st->A,&st->w,PETSC_NULL);CHKERRQ(ierr);
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
  PetscValidHeaderSpecific(st,ST_CLASSID,1);
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
PetscErrorCode STBackTransform(ST st,PetscInt n,PetscScalar* eigr,PetscScalar* eigi)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(st,ST_CLASSID,1);
  if (st->ops->backtr) {
    ierr = (*st->ops->backtr)(st,n,eigr,eigi);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}
