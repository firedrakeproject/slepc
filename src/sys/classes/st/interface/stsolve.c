/*
    The ST (spectral transformation) interface routines, callable by users.

   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
   SLEPc - Scalable Library for Eigenvalue Problem Computations
   Copyright (c) 2002-2016, Universitat Politecnica de Valencia, Spain

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

#include <slepc/private/stimpl.h>            /*I "slepcst.h" I*/

#undef __FUNCT__
#define __FUNCT__ "STApply"
/*@
   STApply - Applies the spectral transformation operator to a vector, for
   instance (A - sB)^-1 B in the case of the shift-and-invert transformation
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
  PetscValidType(st,1);
  STCheckMatrices(st,1);
  if (x == y) SETERRQ(PetscObjectComm((PetscObject)st),PETSC_ERR_ARG_IDN,"x and y must be different vectors");
  VecLocked(y,3);

  if (st->state!=ST_STATE_SETUP) { ierr = STSetUp(st);CHKERRQ(ierr); }

  if (!st->ops->apply) SETERRQ(PetscObjectComm((PetscObject)st),PETSC_ERR_SUP,"ST does not have apply");
  ierr = VecLockPush(x);CHKERRQ(ierr);
  ierr = PetscLogEventBegin(ST_Apply,st,x,y,0);CHKERRQ(ierr);
  if (st->D) { /* with balancing */
    ierr = VecPointwiseDivide(st->wb,x,st->D);CHKERRQ(ierr);
    ierr = (*st->ops->apply)(st,st->wb,y);CHKERRQ(ierr);
    ierr = VecPointwiseMult(y,y,st->D);CHKERRQ(ierr);
  } else {
    ierr = (*st->ops->apply)(st,x,y);CHKERRQ(ierr);
  }
  ierr = PetscLogEventEnd(ST_Apply,st,x,y,0);CHKERRQ(ierr);
  ierr = VecLockPop(x);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "STApplyTranspose"
/*@
   STApplyTranspose - Applies the transpose of the operator to a vector, for
   instance B^T(A - sB)^-T in the case of the shift-and-invert transformation
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
  PetscValidType(st,1);
  STCheckMatrices(st,1);
  if (x == y) SETERRQ(PetscObjectComm((PetscObject)st),PETSC_ERR_ARG_IDN,"x and y must be different vectors");
  VecLocked(y,3);

  if (st->state!=ST_STATE_SETUP) { ierr = STSetUp(st);CHKERRQ(ierr); }

  if (!st->ops->applytrans) SETERRQ(PetscObjectComm((PetscObject)st),PETSC_ERR_SUP,"ST does not have applytrans");
  ierr = VecLockPush(x);CHKERRQ(ierr);
  ierr = PetscLogEventBegin(ST_ApplyTranspose,st,x,y,0);CHKERRQ(ierr);
  if (st->D) { /* with balancing */
    ierr = VecPointwiseMult(st->wb,x,st->D);CHKERRQ(ierr);
    ierr = (*st->ops->applytrans)(st,st->wb,y);CHKERRQ(ierr);
    ierr = VecPointwiseDivide(y,y,st->D);CHKERRQ(ierr);
  } else {
    ierr = (*st->ops->applytrans)(st,x,y);CHKERRQ(ierr);
  }
  ierr = PetscLogEventEnd(ST_ApplyTranspose,st,x,y,0);CHKERRQ(ierr);
  ierr = VecLockPop(x);CHKERRQ(ierr);
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
  PetscValidType(st,1);
  PetscValidPointer(B,2);
  STCheckMatrices(st,1);
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
  STCheckMatrices(st,1);
  if (st->nmat>2) SETERRQ(PetscObjectComm((PetscObject)st),PETSC_ERR_ARG_WRONGSTATE,"Can only be used with 1 or 2 matrices");
  ierr = MPI_Comm_size(PetscObjectComm((PetscObject)st),&size);CHKERRQ(ierr);

  ierr = MatCreateVecs(st->A[0],&in,&out);CHKERRQ(ierr);
  ierr = VecGetSize(out,&M);CHKERRQ(ierr);
  ierr = VecGetLocalSize(out,&m);CHKERRQ(ierr);
  ierr = VecSetOption(in,VEC_IGNORE_OFF_PROC_ENTRIES,PETSC_TRUE);CHKERRQ(ierr);
  ierr = VecGetOwnershipRange(out,&start,&end);CHKERRQ(ierr);
  ierr = PetscMalloc1(m,&rows);CHKERRQ(ierr);
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
  STCheckMatrices(st,1);
  if (st->state==ST_STATE_SETUP) PetscFunctionReturn(0);
  ierr = PetscInfo(st,"Setting up new ST\n");CHKERRQ(ierr);
  ierr = PetscLogEventBegin(ST_SetUp,st,0,0,0);CHKERRQ(ierr);
  if (!((PetscObject)st)->type_name) {
    ierr = STSetType(st,STSHIFT);CHKERRQ(ierr);
  }
  if (!st->T) {
    ierr = PetscMalloc1(PetscMax(2,st->nmat),&st->T);CHKERRQ(ierr);
    ierr = PetscLogObjectMemory((PetscObject)st,PetscMax(2,st->nmat)*sizeof(Mat));CHKERRQ(ierr);
    for (i=0;i<PetscMax(2,st->nmat);i++) st->T[i] = NULL;
  } else if (st->state!=ST_STATE_UPDATED) {
    for (i=0;i<PetscMax(2,st->nmat);i++) {
      ierr = MatDestroy(&st->T[i]);CHKERRQ(ierr);
    }
  }
  if (st->state!=ST_STATE_UPDATED) { ierr = MatDestroy(&st->P);CHKERRQ(ierr); }
  if (st->D) {
    ierr = MatGetLocalSize(st->A[0],NULL,&n);CHKERRQ(ierr);
    ierr = VecGetLocalSize(st->D,&k);CHKERRQ(ierr);
    if (n != k) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_ARG_SIZ,"Balance matrix has wrong dimension %D (should be %D)",k,n);
    if (!st->wb) {
      ierr = VecDuplicate(st->D,&st->wb);CHKERRQ(ierr);
      ierr = PetscLogObjectParent((PetscObject)st,(PetscObject)st->wb);CHKERRQ(ierr);
    }
  }
  if (st->ops->setup) { ierr = (*st->ops->setup)(st);CHKERRQ(ierr); }
  st->state = ST_STATE_SETUP;
  ierr = PetscLogEventEnd(ST_SetUp,st,0,0,0);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "STMatMAXPY_Private"
/*
   Computes coefficients for the transformed polynomial,
   and stores the result in argument S.

   alpha - value of the parameter of the transformed polynomial
   beta - value of the previous shift (only used in inplace mode)
   k - number of A matrices involved in the computation
   coeffs - coefficients of the expansion
   initial - true if this is the first time (only relevant for shell mode)
*/
PetscErrorCode STMatMAXPY_Private(ST st,PetscScalar alpha,PetscScalar beta,PetscInt k,PetscScalar *coeffs,PetscBool initial,Mat *S)
{
  PetscErrorCode ierr;
  PetscInt       *matIdx=NULL,nmat,i,ini=-1;
  PetscScalar    t=1.0,ta,gamma;
  PetscBool      nz=PETSC_FALSE;

  PetscFunctionBegin;
  nmat = st->nmat-k;
  switch (st->shift_matrix) {
  case ST_MATMODE_INPLACE:
    if (st->nmat>2) SETERRQ(PetscObjectComm((PetscObject)st),PETSC_ERR_SUP,"ST_MATMODE_INPLACE not supported for polynomial eigenproblems");
    if (initial) {
      ierr = PetscObjectReference((PetscObject)st->A[0]);CHKERRQ(ierr);
      *S = st->A[0];
      gamma = alpha;
    } else gamma = alpha-beta;
    if (gamma != 0.0) {
      if (st->nmat>1) {
        ierr = MatAXPY(*S,gamma,st->A[1],st->str);CHKERRQ(ierr);
      } else {
        ierr = MatShift(*S,gamma);CHKERRQ(ierr);
      }
    }
    break;
  case ST_MATMODE_SHELL:
    if (initial) {
      if (st->nmat>2) {
        ierr = PetscMalloc1(nmat,&matIdx);CHKERRQ(ierr);
        for (i=0;i<nmat;i++) matIdx[i] = k+i;
      }
      ierr = STMatShellCreate(st,alpha,nmat,matIdx,coeffs,S);CHKERRQ(ierr);
      ierr = PetscLogObjectParent((PetscObject)st,(PetscObject)*S);CHKERRQ(ierr);
      if (st->nmat>2) { ierr = PetscFree(matIdx);CHKERRQ(ierr); }
    } else {
      ierr = STMatShellShift(*S,alpha);CHKERRQ(ierr);
    }
    break;
  case ST_MATMODE_COPY:
    if (coeffs) {
      for (i=0;i<nmat && ini==-1;i++) {
        if (coeffs[i]!=0.0) ini = i;
        else t *= alpha;
      }
      if (coeffs[ini] != 1.0) nz = PETSC_TRUE;
      for (i=ini+1;i<nmat&&!nz;i++) if (coeffs[i]!=0.0) nz = PETSC_TRUE;
    } else { nz = PETSC_TRUE; ini = 0; }
    if ((alpha == 0.0 || !nz) && t==1.0) {
      ierr = MatDestroy(S);CHKERRQ(ierr);
      ierr = PetscObjectReference((PetscObject)st->A[k+ini]);CHKERRQ(ierr);
      *S = st->A[k+ini];
    } else {
      if (*S && *S!=st->A[k+ini]) {
        ierr = MatSetOption(*S,MAT_NEW_NONZERO_ALLOCATION_ERR,PETSC_FALSE);CHKERRQ(ierr);
        ierr = MatCopy(st->A[k+ini],*S,DIFFERENT_NONZERO_PATTERN);CHKERRQ(ierr);
      } else {
        ierr = MatDestroy(S);CHKERRQ(ierr);
        ierr = MatDuplicate(st->A[k+ini],MAT_COPY_VALUES,S);CHKERRQ(ierr);
        ierr = PetscLogObjectParent((PetscObject)st,(PetscObject)*S);CHKERRQ(ierr);
      }
      if (coeffs && coeffs[ini]!=1.0) {
        ierr = MatScale(*S,coeffs[ini]);CHKERRQ(ierr);
      }
      for (i=ini+k+1;i<PetscMax(2,st->nmat);i++) {
        t *= alpha;
        ta = t;
        if (coeffs) ta *= coeffs[i-k];
        if (ta!=0.0) {
          if (st->nmat>1) {
            ierr = MatAXPY(*S,ta,st->A[i],st->str);CHKERRQ(ierr);
          } else {
            ierr = MatShift(*S,ta);CHKERRQ(ierr);
          }
        }
      }
    }
  }
  ierr = STMatSetHermitian(st,*S);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "STCoeffs_Monomial"
/*
   Computes the values of the coefficients required by STMatMAXPY_Private
   for the case of monomial basis.
*/
PetscErrorCode STCoeffs_Monomial(ST st, PetscScalar *coeffs)
{
  PetscInt  k,i,ini,inip;  

  PetscFunctionBegin;
  /* Compute binomial coefficients */
  ini = (st->nmat*(st->nmat-1))/2;
  for (i=0;i<st->nmat;i++) coeffs[ini+i]=1.0;
  for (k=st->nmat-1;k>=1;k--) {
    inip = ini+1;
    ini = (k*(k-1))/2;
    coeffs[ini] = 1.0;
    for (i=1;i<k;i++) coeffs[ini+i] = coeffs[ini+i-1]+coeffs[inip+i-1];
  }
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
  PetscValidType(st,1);
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
  PetscValidType(st,1);
  if (st->ops->backtransform) {
    ierr = (*st->ops->backtransform)(st,n,eigr,eigi);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "STMatSetUp"
/*@
   STMatSetUp - Build the preconditioner matrix used in STMatSolve().

   Collective on ST

   Input Parameters:
+  st     - the spectral transformation context
.  sigma  - the shift
-  coeffs - the coefficients (may be NULL)

   Note:
   This function is not intended to be called by end users, but by SLEPc
   solvers that use ST. It builds matrix st->P as follows, then calls KSPSetUp().
.vb
    If (coeffs):  st->P = Sum_{i=0:nmat-1} coeffs[i]*sigma^i*A_i.
    else          st->P = Sum_{i=0:nmat-1} sigma^i*A_i
.ve

   Level: developer

.seealso: STMatSolve()
@*/
PetscErrorCode STMatSetUp(ST st,PetscScalar sigma,PetscScalar *coeffs)
{
  PetscErrorCode ierr;
  PetscBool      flg;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(st,ST_CLASSID,1);
  PetscValidLogicalCollectiveScalar(st,sigma,2);
  STCheckMatrices(st,1);

  ierr = PetscLogEventBegin(ST_MatSetUp,st,0,0,0);CHKERRQ(ierr);
  ierr = STMatMAXPY_Private(st,sigma,0.0,0,coeffs,PETSC_TRUE,&st->P);CHKERRQ(ierr);
  if (!st->ksp) { ierr = STGetKSP(st,&st->ksp);CHKERRQ(ierr); }
  ierr = STCheckFactorPackage(st);CHKERRQ(ierr);
  ierr = KSPSetOperators(st->ksp,st->P,st->P);CHKERRQ(ierr);
  ierr = PetscObjectTypeCompare((PetscObject)st,STPRECOND,&flg);CHKERRQ(ierr);
  if (!flg) {
    ierr = KSPSetErrorIfNotConverged(st->ksp,PETSC_TRUE);CHKERRQ(ierr);
  }
  ierr = KSPSetUp(st->ksp);CHKERRQ(ierr);
  ierr = PetscLogEventEnd(ST_MatSetUp,st,0,0,0);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

