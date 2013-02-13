/*
    The ST (spectral transformation) interface routines, callable by users.

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

#include <slepc-private/stimpl.h>            /*I "slepcst.h" I*/

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
  if (x == y) SETERRQ(PetscObjectComm((PetscObject)st),PETSC_ERR_ARG_IDN,"x and y must be different vectors");

  if (!st->setupcalled) { ierr = STSetUp(st);CHKERRQ(ierr); }

  if (!st->ops->apply) SETERRQ(PetscObjectComm((PetscObject)st),PETSC_ERR_SUP,"ST does not have apply");
  ierr = PetscLogEventBegin(ST_Apply,st,x,y,0);CHKERRQ(ierr);
  st->applys++;
  if (st->D) { /* with balancing */
    ierr = VecPointwiseDivide(st->wb,x,st->D);CHKERRQ(ierr);
    ierr = (*st->ops->apply)(st,st->wb,y);CHKERRQ(ierr);
    ierr = VecPointwiseMult(y,y,st->D);CHKERRQ(ierr);
  } else {
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

   Not collective, though a parallel Mat may be returned

   Input Parameters:
.  st - the spectral transformation context

   Output Parameter:
.  B - output matrix

   Notes:
   The output matrix B must be destroyed after use. It will be NULL in
   case of standard eigenproblems.
   
   Level: developer
@*/
PetscErrorCode STGetBilinearForm(ST st,Mat *B)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(st,ST_CLASSID,1);
  PetscValidPointer(B,2);
  if (!st->A) SETERRQ(PetscObjectComm((PetscObject)st),PETSC_ERR_ARG_WRONGSTATE,"Matrices must be set first");
  ierr = (*st->ops->getbilinearform)(st,B);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "STGetBilinearForm_Default"
PetscErrorCode STGetBilinearForm_Default(ST st,Mat *B)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (st->nmat==1) *B = NULL;
  else {
    *B = st->A[1];
    ierr = PetscObjectReference((PetscObject)*B);CHKERRQ(ierr);
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
  if (x == y) SETERRQ(PetscObjectComm((PetscObject)st),PETSC_ERR_ARG_IDN,"x and y must be different vectors");

  if (!st->setupcalled) { ierr = STSetUp(st);CHKERRQ(ierr); }

  if (!st->ops->applytrans) SETERRQ(PetscObjectComm((PetscObject)st),PETSC_ERR_SUP,"ST does not have applytrans");
  ierr = PetscLogEventBegin(ST_ApplyTranspose,st,x,y,0);CHKERRQ(ierr);
  st->applys++;
  if (st->D) { /* with balancing */
    ierr = VecPointwiseMult(st->wb,x,st->D);CHKERRQ(ierr);
    ierr = (*st->ops->applytrans)(st,st->wb,y);CHKERRQ(ierr);
    ierr = VecPointwiseDivide(y,y,st->D);CHKERRQ(ierr);
  } else {
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
   identity matrix. This is analogous to MatComputeExplicitOperator().

   Level: advanced

.seealso: STApply()   
@*/
PetscErrorCode STComputeExplicitOperator(ST st,Mat *mat)
{
  PetscErrorCode    ierr;
  Vec               in,out;
  PetscInt          i,M,m,*rows,start,end;
  const PetscScalar *array;
  PetscScalar       one = 1.0;
  PetscMPIInt       size;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(st,ST_CLASSID,1);
  PetscValidPointer(mat,2);
  if (!st->A) SETERRQ(PetscObjectComm((PetscObject)st),PETSC_ERR_ARG_WRONGSTATE,"Matrices must be set first");
  ierr = MPI_Comm_size(PetscObjectComm((PetscObject)st),&size);CHKERRQ(ierr);

  ierr = MatGetVecs(st->A[0],&in,&out);CHKERRQ(ierr);
  ierr = VecGetSize(out,&M);CHKERRQ(ierr);
  ierr = VecGetLocalSize(out,&m);CHKERRQ(ierr);
  ierr = VecSetOption(in,VEC_IGNORE_OFF_PROC_ENTRIES,PETSC_TRUE);CHKERRQ(ierr);
  ierr = VecGetOwnershipRange(out,&start,&end);CHKERRQ(ierr);
  ierr = PetscMalloc(m*sizeof(PetscInt),&rows);CHKERRQ(ierr);
  for (i=0;i<m;i++) rows[i] = start + i;

  ierr = MatCreate(PetscObjectComm((PetscObject)st),mat);CHKERRQ(ierr);
  ierr = MatSetSizes(*mat,m,m,M,M);CHKERRQ(ierr);
  if (size == 1) {
    ierr = MatSetType(*mat,MATSEQDENSE);CHKERRQ(ierr);
    ierr = MatSeqDenseSetPreallocation(*mat,NULL);CHKERRQ(ierr);
  } else {
    ierr = MatSetType(*mat,MATMPIAIJ);CHKERRQ(ierr);
    ierr = MatMPIAIJSetPreallocation(*mat,m,NULL,M-m,NULL);CHKERRQ(ierr);
  }

  for (i=0;i<M;i++) {
    ierr = VecSet(in,0.0);CHKERRQ(ierr);
    ierr = VecSetValues(in,1,&i,&one,INSERT_VALUES);CHKERRQ(ierr);
    ierr = VecAssemblyBegin(in);CHKERRQ(ierr);
    ierr = VecAssemblyEnd(in);CHKERRQ(ierr);

    ierr = STApply(st,in,out);CHKERRQ(ierr);
    
    ierr = VecGetArrayRead(out,&array);CHKERRQ(ierr);
    ierr = MatSetValues(*mat,m,rows,1,&i,array,INSERT_VALUES);CHKERRQ(ierr); 
    ierr = VecRestoreArrayRead(out,&array);CHKERRQ(ierr);
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
  PetscInt       i,n,k;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(st,ST_CLASSID,1);
  if (!st->A) SETERRQ(PetscObjectComm((PetscObject)st),PETSC_ERR_ARG_WRONGSTATE,"Matrices must be set first");
  if (st->setupcalled) PetscFunctionReturn(0);
  ierr = PetscInfo(st,"Setting up new ST\n");CHKERRQ(ierr);
  ierr = PetscLogEventBegin(ST_SetUp,st,0,0,0);CHKERRQ(ierr);
  if (!((PetscObject)st)->type_name) {
    ierr = STSetType(st,STSHIFT);CHKERRQ(ierr);
  }
  ierr = STReset(st);CHKERRQ(ierr);
  ierr = PetscMalloc(PetscMax(2,st->nmat)*sizeof(Mat),&st->T);CHKERRQ(ierr);
  for (i=0;i<PetscMax(2,st->nmat);i++) st->T[i] = NULL;
  ierr = MatGetVecs(st->A[0],&st->w,NULL);CHKERRQ(ierr);
  if (st->D) {
    ierr = MatGetLocalSize(st->A[0],NULL,&n);CHKERRQ(ierr);
    ierr = VecGetLocalSize(st->D,&k);CHKERRQ(ierr);
    if (n != k) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_ARG_SIZ,"Balance matrix has wrong dimension %D (should be %D)",k,n);
    ierr = VecDuplicate(st->D,&st->wb);CHKERRQ(ierr);
  }
  if (st->ops->setup) { ierr = (*st->ops->setup)(st);CHKERRQ(ierr); }
  st->setupcalled = 1;
  ierr = PetscLogEventEnd(ST_SetUp,st,0,0,0);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "STMatGAXPY_Private"
/*
   Computes a generalized AXPY operation on the A[:] matrices provided by the user,
   and stores the result in one of the T[:] matrices. The computation is different
   depending on whether the ST has two or three matrices. Also, in quadratic problems
   this function is used to compute two matrices, those corresponding to degree 1 and 2.

   Builds matrix in T[k] as follows:
   deg=1: T[k] = A+alpha*B (if st->nmat==2) or B+2*alpha*C (if st->nmat==3)
   deg=2: T[k] = A+alpha*B+alpha*alpha*C
*/
PetscErrorCode STMatGAXPY_Private(ST st,PetscScalar alpha,PetscScalar beta,PetscInt deg,PetscInt k,PetscBool initial)
{
  PetscErrorCode ierr;
  PetscScalar    gamma;
  PetscInt       matIdx[3],t,i;

  PetscFunctionBegin;
  if (st->nmat==3 && deg==1) t = 1;
  else t = 0;
  switch (st->shift_matrix) {
  case ST_MATMODE_INPLACE:
    if (initial) {
      ierr = PetscObjectReference((PetscObject)st->A[t]);CHKERRQ(ierr);
      st->T[k] = st->A[t];
      gamma = alpha;
    } else gamma = alpha-beta;
    if (gamma != 0.0) {
      if (st->nmat>1) {
        ierr = MatAXPY(st->T[k],gamma,st->A[t+1],st->str);CHKERRQ(ierr);
        if (st->nmat==3 && deg==2) {
          ierr = MatAXPY(st->T[k],gamma*gamma,st->A[2],st->str);CHKERRQ(ierr);
        }
      } else {
        ierr = MatShift(st->T[k],gamma);CHKERRQ(ierr);
      }
    }
    break;
  case ST_MATMODE_SHELL:
    if (initial) {
      if (st->nmat>1) {
        for (i=0;i<=deg;i++) {
          matIdx[i] = t+i;
        }
        ierr = STMatShellCreate(st,alpha,deg+1,matIdx,&st->T[k]);CHKERRQ(ierr);
      } else {
        ierr = STMatShellCreate(st,alpha,deg,NULL,&st->T[k]);CHKERRQ(ierr);
      }
    } else {
      ierr = STMatShellShift(st->T[k],alpha);CHKERRQ(ierr);
    }
    break;
  default:
    if (alpha == 0.0) {
      if (!initial) {
        ierr = MatDestroy(&st->T[k]);CHKERRQ(ierr);
      }
      ierr = PetscObjectReference((PetscObject)st->A[t]);CHKERRQ(ierr);
      st->T[k] = st->A[t];
    } else {
      if (initial) {
        ierr = MatDuplicate(st->A[t],MAT_COPY_VALUES,&st->T[k]);CHKERRQ(ierr);
      } else {
        if (beta==0.0) {
          ierr = MatDestroy(&st->T[k]);CHKERRQ(ierr);
          ierr = MatDuplicate(st->A[t],MAT_COPY_VALUES,&st->T[k]);CHKERRQ(ierr);
        } else {
          ierr = MatCopy(st->A[t],st->T[k],SAME_NONZERO_PATTERN);CHKERRQ(ierr);
        }
      }
      if (st->nmat>1) {
        ierr = MatAXPY(st->T[k],alpha,st->A[t+1],st->str);CHKERRQ(ierr);
        if (st->nmat==3 && deg==2) {
          ierr = MatAXPY(st->T[k],alpha*alpha,st->A[2],st->str);CHKERRQ(ierr);
        }
      } else {
        ierr = MatShift(st->T[k],alpha);CHKERRQ(ierr);
      }
    }
  }
  ierr = STMatSetHermitian(st,st->T[k]);CHKERRQ(ierr);
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

   Not Collective

   Input Parameters:
   st   - the spectral transformation context
   eigr - real part of a computed eigenvalue
   eigi - imaginary part of a computed eigenvalue

   Level: developer
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
