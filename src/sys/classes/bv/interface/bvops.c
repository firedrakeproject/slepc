/*
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
   SLEPc - Scalable Library for Eigenvalue Problem Computations
   Copyright (c) 2002-2021, Universitat Politecnica de Valencia, Spain

   This file is part of SLEPc.
   SLEPc is distributed under a 2-clause BSD license (see LICENSE).
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
*/
/*
   BV operations, except those involving global communication
*/

#include <slepc/private/bvimpl.h>      /*I "slepcbv.h" I*/
#include <slepcds.h>

/*@
   BVMult - Computes Y = beta*Y + alpha*X*Q.

   Logically Collective on Y

   Input Parameters:
+  Y     - first basis vectors context (modified on output)
.  alpha - first scalar
.  beta  - second scalar
.  X     - second basis vectors context
-  Q     - (optional) sequential dense matrix

   Notes:
   X and Y must be different objects. The case X=Y can be addressed with
   BVMultInPlace().

   If matrix Q is NULL, then an AXPY operation Y = beta*Y + alpha*X is done
   (i.e. results as if Q = identity). If provided,
   the matrix Q must be a sequential dense Mat, with all entries equal on
   all processes (otherwise each process will compute a different update).
   The dimensions of Q must be at least m,n where m is the number of active
   columns of X and n is the number of active columns of Y.

   The leading columns of Y are not modified. Also, if X has leading
   columns specified, then these columns do not participate in the computation.
   Hence, only rows (resp. columns) of Q starting from lx (resp. ly) are used,
   where lx (resp. ly) is the number of leading columns of X (resp. Y).

   Level: intermediate

.seealso: BVMultVec(), BVMultColumn(), BVMultInPlace(), BVSetActiveColumns()
@*/
PetscErrorCode BVMult(BV Y,PetscScalar alpha,PetscScalar beta,BV X,Mat Q)
{
  PetscErrorCode ierr;
  PetscInt       m,n;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(Y,BV_CLASSID,1);
  PetscValidLogicalCollectiveScalar(Y,alpha,2);
  PetscValidLogicalCollectiveScalar(Y,beta,3);
  PetscValidHeaderSpecific(X,BV_CLASSID,4);
  if (Q) PetscValidHeaderSpecific(Q,MAT_CLASSID,5);
  PetscValidType(Y,1);
  BVCheckSizes(Y,1);
  BVCheckOp(Y,1,mult);
  PetscValidType(X,4);
  BVCheckSizes(X,4);
  if (Q) PetscValidType(Q,5);
  PetscCheckSameTypeAndComm(Y,1,X,4);
  PetscCheck(X!=Y,PetscObjectComm((PetscObject)Y),PETSC_ERR_ARG_WRONG,"X and Y arguments must be different");
  if (Q) {
    PetscCheckTypeNames(Q,MATSEQDENSE,MATSEQDENSECUDA);
    ierr = MatGetSize(Q,&m,&n);CHKERRQ(ierr);
    PetscCheck(m>=X->k,PetscObjectComm((PetscObject)Y),PETSC_ERR_ARG_SIZ,"Mat argument has %" PetscInt_FMT " rows, should have at least %" PetscInt_FMT,m,X->k);
    PetscCheck(n>=Y->k,PetscObjectComm((PetscObject)Y),PETSC_ERR_ARG_SIZ,"Mat argument has %" PetscInt_FMT " columns, should have at least %" PetscInt_FMT,n,Y->k);
  }
  PetscCheck(X->n==Y->n,PetscObjectComm((PetscObject)Y),PETSC_ERR_ARG_INCOMP,"Mismatching local dimension X %" PetscInt_FMT ", Y %" PetscInt_FMT,X->n,Y->n);

  ierr = PetscLogEventBegin(BV_Mult,X,Y,0,0);CHKERRQ(ierr);
  ierr = (*Y->ops->mult)(Y,alpha,beta,X,Q);CHKERRQ(ierr);
  ierr = PetscLogEventEnd(BV_Mult,X,Y,0,0);CHKERRQ(ierr);
  ierr = PetscObjectStateIncrease((PetscObject)Y);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@
   BVMultVec - Computes y = beta*y + alpha*X*q.

   Logically Collective on X

   Input Parameters:
+  X     - a basis vectors object
.  alpha - first scalar
.  beta  - second scalar
.  y     - a vector (modified on output)
-  q     - an array of scalars

   Notes:
   This operation is the analogue of BVMult() but with a BV and a Vec,
   instead of two BV. Note that arguments are listed in different order
   with respect to BVMult().

   If X has leading columns specified, then these columns do not participate
   in the computation.

   The length of array q must be equal to the number of active columns of X
   minus the number of leading columns, i.e. the first entry of q multiplies
   the first non-leading column.

   Level: intermediate

.seealso: BVMult(), BVMultColumn(), BVMultInPlace(), BVSetActiveColumns()
@*/
PetscErrorCode BVMultVec(BV X,PetscScalar alpha,PetscScalar beta,Vec y,PetscScalar q[])
{
  PetscErrorCode ierr;
  PetscInt       n,N;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(X,BV_CLASSID,1);
  PetscValidLogicalCollectiveScalar(X,alpha,2);
  PetscValidLogicalCollectiveScalar(X,beta,3);
  PetscValidHeaderSpecific(y,VEC_CLASSID,4);
  PetscValidScalarPointer(q,5);
  PetscValidType(X,1);
  BVCheckSizes(X,1);
  BVCheckOp(X,1,multvec);
  PetscValidType(y,4);
  PetscCheckSameComm(X,1,y,4);

  ierr = VecGetSize(y,&N);CHKERRQ(ierr);
  ierr = VecGetLocalSize(y,&n);CHKERRQ(ierr);
  PetscCheck(N==X->N && n==X->n,PetscObjectComm((PetscObject)X),PETSC_ERR_ARG_INCOMP,"Vec sizes (global %" PetscInt_FMT ", local %" PetscInt_FMT ") do not match BV sizes (global %" PetscInt_FMT ", local %" PetscInt_FMT ")",N,n,X->N,X->n);

  ierr = PetscLogEventBegin(BV_MultVec,X,y,0,0);CHKERRQ(ierr);
  ierr = (*X->ops->multvec)(X,alpha,beta,y,q);CHKERRQ(ierr);
  ierr = PetscLogEventEnd(BV_MultVec,X,y,0,0);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@
   BVMultColumn - Computes y = beta*y + alpha*X*q, where y is the j-th column
   of X.

   Logically Collective on X

   Input Parameters:
+  X     - a basis vectors object
.  alpha - first scalar
.  beta  - second scalar
.  j     - the column index
-  q     - an array of scalars

   Notes:
   This operation is equivalent to BVMultVec() but it uses column j of X
   rather than taking a Vec as an argument. The number of active columns of
   X is set to j before the computation, and restored afterwards.
   If X has leading columns specified, then these columns do not participate
   in the computation. Therefore, the length of array q must be equal to j
   minus the number of leading columns.

   Developer Notes:
   If q is NULL, then the coefficients are taken from position nc+l of the
   internal buffer vector, see BVGetBufferVec().

   Level: advanced

.seealso: BVMult(), BVMultVec(), BVMultInPlace(), BVSetActiveColumns()
@*/
PetscErrorCode BVMultColumn(BV X,PetscScalar alpha,PetscScalar beta,PetscInt j,PetscScalar *q)
{
  PetscErrorCode ierr;
  PetscInt       ksave;
  Vec            y;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(X,BV_CLASSID,1);
  PetscValidLogicalCollectiveScalar(X,alpha,2);
  PetscValidLogicalCollectiveScalar(X,beta,3);
  PetscValidLogicalCollectiveInt(X,j,4);
  PetscValidType(X,1);
  BVCheckSizes(X,1);

  PetscCheck(j>=0,PetscObjectComm((PetscObject)X),PETSC_ERR_ARG_OUTOFRANGE,"Index j must be non-negative");
  PetscCheck(j<X->m,PetscObjectComm((PetscObject)X),PETSC_ERR_ARG_OUTOFRANGE,"Index j=%" PetscInt_FMT " but BV only has %" PetscInt_FMT " columns",j,X->m);

  ierr = PetscLogEventBegin(BV_MultVec,X,0,0,0);CHKERRQ(ierr);
  ksave = X->k;
  X->k = j;
  if (!q && !X->buffer) { ierr = BVGetBufferVec(X,&X->buffer);CHKERRQ(ierr); }
  ierr = BVGetColumn(X,j,&y);CHKERRQ(ierr);
  ierr = (*X->ops->multvec)(X,alpha,beta,y,q);CHKERRQ(ierr);
  ierr = BVRestoreColumn(X,j,&y);CHKERRQ(ierr);
  X->k = ksave;
  ierr = PetscLogEventEnd(BV_MultVec,X,0,0,0);CHKERRQ(ierr);
  ierr = PetscObjectStateIncrease((PetscObject)X);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@
   BVMultInPlace - Update a set of vectors as V(:,s:e-1) = V*Q(:,s:e-1).

   Logically Collective on V

   Input Parameters:
+  Q - a sequential dense matrix
.  s - first column of V to be overwritten
-  e - first column of V not to be overwritten

   Input/Output Parameter:
.  V - basis vectors

   Notes:
   The matrix Q must be a sequential dense Mat, with all entries equal on
   all processes (otherwise each process will compute a different update).

   This function computes V(:,s:e-1) = V*Q(:,s:e-1), that is, given a set of
   vectors V, columns from s to e-1 are overwritten with columns from s to
   e-1 of the matrix-matrix product V*Q. Only columns s to e-1 of Q are
   referenced.

   Level: intermediate

.seealso: BVMult(), BVMultVec(), BVMultInPlaceHermitianTranspose(), BVSetActiveColumns()
@*/
PetscErrorCode BVMultInPlace(BV V,Mat Q,PetscInt s,PetscInt e)
{
  PetscErrorCode ierr;
  PetscInt       m,n;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(V,BV_CLASSID,1);
  PetscValidHeaderSpecific(Q,MAT_CLASSID,2);
  PetscValidLogicalCollectiveInt(V,s,3);
  PetscValidLogicalCollectiveInt(V,e,4);
  PetscValidType(V,1);
  BVCheckSizes(V,1);
  PetscValidType(Q,2);
  PetscCheckTypeNames(Q,MATSEQDENSE,MATSEQDENSECUDA);

  PetscCheck(s>=V->l && s<=V->m,PetscObjectComm((PetscObject)V),PETSC_ERR_ARG_OUTOFRANGE,"Argument s has wrong value %" PetscInt_FMT ", should be between %" PetscInt_FMT " and %" PetscInt_FMT,s,V->l,V->m);
  PetscCheck(e>=V->l && e<=V->m,PetscObjectComm((PetscObject)V),PETSC_ERR_ARG_OUTOFRANGE,"Argument e has wrong value %" PetscInt_FMT ", should be between %" PetscInt_FMT " and %" PetscInt_FMT,e,V->l,V->m);
  ierr = MatGetSize(Q,&m,&n);CHKERRQ(ierr);
  PetscCheck(m>=V->k,PetscObjectComm((PetscObject)V),PETSC_ERR_ARG_SIZ,"Mat argument has %" PetscInt_FMT " rows, should have at least %" PetscInt_FMT,m,V->k);
  PetscCheck(e<=n,PetscObjectComm((PetscObject)V),PETSC_ERR_ARG_SIZ,"Mat argument only has %" PetscInt_FMT " columns, the requested value of e is larger: %" PetscInt_FMT,n,e);
  if (s>=e) PetscFunctionReturn(0);

  ierr = PetscLogEventBegin(BV_MultInPlace,V,Q,0,0);CHKERRQ(ierr);
  ierr = (*V->ops->multinplace)(V,Q,s,e);CHKERRQ(ierr);
  ierr = PetscLogEventEnd(BV_MultInPlace,V,Q,0,0);CHKERRQ(ierr);
  ierr = PetscObjectStateIncrease((PetscObject)V);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@
   BVMultInPlaceHermitianTranspose - Update a set of vectors as V(:,s:e-1) = V*Q'(:,s:e-1).

   Logically Collective on V

   Input Parameters:
+  Q - a sequential dense matrix
.  s - first column of V to be overwritten
-  e - first column of V not to be overwritten

   Input/Output Parameter:
.  V - basis vectors

   Notes:
   This is a variant of BVMultInPlace() where the conjugate transpose
   of Q is used.

   Level: intermediate

.seealso: BVMultInPlace()
@*/
PetscErrorCode BVMultInPlaceHermitianTranspose(BV V,Mat Q,PetscInt s,PetscInt e)
{
  PetscErrorCode ierr;
  PetscInt       m,n;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(V,BV_CLASSID,1);
  PetscValidHeaderSpecific(Q,MAT_CLASSID,2);
  PetscValidLogicalCollectiveInt(V,s,3);
  PetscValidLogicalCollectiveInt(V,e,4);
  PetscValidType(V,1);
  BVCheckSizes(V,1);
  PetscValidType(Q,2);
  PetscCheckTypeNames(Q,MATSEQDENSE,MATSEQDENSECUDA);

  PetscCheck(s>=V->l && s<=V->m,PetscObjectComm((PetscObject)V),PETSC_ERR_ARG_OUTOFRANGE,"Argument s has wrong value %" PetscInt_FMT ", should be between %" PetscInt_FMT " and %" PetscInt_FMT,s,V->l,V->m);
  PetscCheck(e>=V->l && e<=V->m,PetscObjectComm((PetscObject)V),PETSC_ERR_ARG_OUTOFRANGE,"Argument e has wrong value %" PetscInt_FMT ", should be between %" PetscInt_FMT " and %" PetscInt_FMT,e,V->l,V->m);
  ierr = MatGetSize(Q,&m,&n);CHKERRQ(ierr);
  PetscCheck(n>=V->k,PetscObjectComm((PetscObject)V),PETSC_ERR_ARG_SIZ,"Mat argument has %" PetscInt_FMT " columns, should have at least %" PetscInt_FMT,n,V->k);
  PetscCheck(e<=m,PetscObjectComm((PetscObject)V),PETSC_ERR_ARG_SIZ,"Mat argument only has %" PetscInt_FMT " rows, the requested value of e is larger: %" PetscInt_FMT,m,e);
  if (s>=e || !V->n) PetscFunctionReturn(0);

  ierr = PetscLogEventBegin(BV_MultInPlace,V,Q,0,0);CHKERRQ(ierr);
  ierr = (*V->ops->multinplacetrans)(V,Q,s,e);CHKERRQ(ierr);
  ierr = PetscLogEventEnd(BV_MultInPlace,V,Q,0,0);CHKERRQ(ierr);
  ierr = PetscObjectStateIncrease((PetscObject)V);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@
   BVScale - Multiply the BV entries by a scalar value.

   Logically Collective on bv

   Input Parameters:
+  bv    - basis vectors
-  alpha - scaling factor

   Note:
   All active columns (except the leading ones) are scaled.

   Level: intermediate

.seealso: BVScaleColumn(), BVSetActiveColumns()
@*/
PetscErrorCode BVScale(BV bv,PetscScalar alpha)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(bv,BV_CLASSID,1);
  PetscValidLogicalCollectiveScalar(bv,alpha,2);
  PetscValidType(bv,1);
  BVCheckSizes(bv,1);
  if (alpha == (PetscScalar)1.0) PetscFunctionReturn(0);

  ierr = PetscLogEventBegin(BV_Scale,bv,0,0,0);CHKERRQ(ierr);
  if (bv->n) {
    ierr = (*bv->ops->scale)(bv,-1,alpha);CHKERRQ(ierr);
  }
  ierr = PetscLogEventEnd(BV_Scale,bv,0,0,0);CHKERRQ(ierr);
  ierr = PetscObjectStateIncrease((PetscObject)bv);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@
   BVScaleColumn - Scale one column of a BV.

   Logically Collective on bv

   Input Parameters:
+  bv    - basis vectors
.  j     - column number to be scaled
-  alpha - scaling factor

   Level: intermediate

.seealso: BVScale(), BVSetActiveColumns()
@*/
PetscErrorCode BVScaleColumn(BV bv,PetscInt j,PetscScalar alpha)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(bv,BV_CLASSID,1);
  PetscValidLogicalCollectiveInt(bv,j,2);
  PetscValidLogicalCollectiveScalar(bv,alpha,3);
  PetscValidType(bv,1);
  BVCheckSizes(bv,1);

  PetscCheck(j>=0 && j<bv->m,PetscObjectComm((PetscObject)bv),PETSC_ERR_ARG_OUTOFRANGE,"Argument j has wrong value %" PetscInt_FMT ", the number of columns is %" PetscInt_FMT,j,bv->m);
  if (alpha == (PetscScalar)1.0) PetscFunctionReturn(0);

  ierr = PetscLogEventBegin(BV_Scale,bv,0,0,0);CHKERRQ(ierr);
  if (bv->n) {
    ierr = (*bv->ops->scale)(bv,j,alpha);CHKERRQ(ierr);
  }
  ierr = PetscLogEventEnd(BV_Scale,bv,0,0,0);CHKERRQ(ierr);
  ierr = PetscObjectStateIncrease((PetscObject)bv);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static inline PetscErrorCode BVSetRandomColumn_Private(BV bv,PetscInt k)
{
  PetscErrorCode ierr;
  PetscInt       i,low,high;
  PetscScalar    *px,t;
  Vec            x;

  PetscFunctionBegin;
  ierr = BVGetColumn(bv,k,&x);CHKERRQ(ierr);
  if (bv->rrandom) {  /* generate the same vector irrespective of number of processes */
    ierr = VecGetOwnershipRange(x,&low,&high);CHKERRQ(ierr);
    ierr = VecGetArray(x,&px);CHKERRQ(ierr);
    for (i=0;i<bv->N;i++) {
      ierr = PetscRandomGetValue(bv->rand,&t);CHKERRQ(ierr);
      if (i>=low && i<high) px[i-low] = t;
    }
    ierr = VecRestoreArray(x,&px);CHKERRQ(ierr);
  } else {
    ierr = VecSetRandom(x,bv->rand);CHKERRQ(ierr);
  }
  ierr = BVRestoreColumn(bv,k,&x);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static inline PetscErrorCode BVSetRandomNormalColumn_Private(BV bv,PetscInt k,Vec w1,Vec w2)
{
  PetscErrorCode ierr;
  PetscInt       i,low,high;
  PetscScalar    *px,s,t;
  Vec            x;

  PetscFunctionBegin;
  ierr = BVGetColumn(bv,k,&x);CHKERRQ(ierr);
  if (bv->rrandom) {  /* generate the same vector irrespective of number of processes */
    ierr = VecGetOwnershipRange(x,&low,&high);CHKERRQ(ierr);
    ierr = VecGetArray(x,&px);CHKERRQ(ierr);
    for (i=0;i<bv->N;i++) {
      ierr = PetscRandomGetValue(bv->rand,&s);CHKERRQ(ierr);
      ierr = PetscRandomGetValue(bv->rand,&t);CHKERRQ(ierr);
      if (i>=low && i<high) {
#if defined(PETSC_USE_COMPLEX)
        px[i-low] = PetscCMPLX(PetscSqrtReal(-2.0*PetscLogReal(PetscRealPart(s)))*PetscCosReal(2.0*PETSC_PI*PetscRealPart(t)),PetscSqrtReal(-2.0*PetscLogReal(PetscImaginaryPart(s)))*PetscCosReal(2.0*PETSC_PI*PetscImaginaryPart(t)));
#else
        px[i-low] = PetscSqrtReal(-2.0*PetscLogReal(s))*PetscCosReal(2.0*PETSC_PI*t);
#endif
      }
    }
    ierr = VecRestoreArray(x,&px);CHKERRQ(ierr);
  } else {
    ierr = VecSetRandomNormal(x,bv->rand,w1,w2);CHKERRQ(ierr);
  }
  ierr = BVRestoreColumn(bv,k,&x);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static inline PetscErrorCode BVSetRandomSignColumn_Private(BV bv,PetscInt k)
{
  PetscErrorCode ierr;
  PetscInt       i,low,high;
  PetscScalar    *px,t;
  Vec            x;

  PetscFunctionBegin;
  ierr = BVGetColumn(bv,k,&x);CHKERRQ(ierr);
  ierr = VecGetOwnershipRange(x,&low,&high);CHKERRQ(ierr);
  if (bv->rrandom) {  /* generate the same vector irrespective of number of processes */
    ierr = VecGetArray(x,&px);CHKERRQ(ierr);
    for (i=0;i<bv->N;i++) {
      ierr = PetscRandomGetValue(bv->rand,&t);CHKERRQ(ierr);
      if (i>=low && i<high) px[i-low] = (PetscRealPart(t)<0.5)? -1.0: 1.0;
    }
    ierr = VecRestoreArray(x,&px);CHKERRQ(ierr);
  } else {
    ierr = VecSetRandom(x,bv->rand);CHKERRQ(ierr);
    ierr = VecGetArray(x,&px);CHKERRQ(ierr);
    for (i=low;i<high;i++) {
      px[i-low] = (PetscRealPart(px[i-low])<0.5)? -1.0: 1.0;
    }
    ierr = VecRestoreArray(x,&px);CHKERRQ(ierr);
  }
  ierr = BVRestoreColumn(bv,k,&x);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@
   BVSetRandom - Set the columns of a BV to random numbers.

   Logically Collective on bv

   Input Parameters:
.  bv - basis vectors

   Note:
   All active columns (except the leading ones) are modified.

   Level: advanced

.seealso: BVSetRandomContext(), BVSetRandomColumn(), BVSetRandomNormal(), BVSetRandomSign(), BVSetRandomCond(), BVSetActiveColumns()
@*/
PetscErrorCode BVSetRandom(BV bv)
{
  PetscErrorCode ierr;
  PetscInt       k;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(bv,BV_CLASSID,1);
  PetscValidType(bv,1);
  BVCheckSizes(bv,1);

  ierr = BVGetRandomContext(bv,&bv->rand);CHKERRQ(ierr);
  ierr = PetscLogEventBegin(BV_SetRandom,bv,0,0,0);CHKERRQ(ierr);
  for (k=bv->l;k<bv->k;k++) {
    ierr = BVSetRandomColumn_Private(bv,k);CHKERRQ(ierr);
  }
  ierr = PetscLogEventEnd(BV_SetRandom,bv,0,0,0);CHKERRQ(ierr);
  ierr = PetscObjectStateIncrease((PetscObject)bv);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@
   BVSetRandomColumn - Set one column of a BV to random numbers.

   Logically Collective on bv

   Input Parameters:
+  bv - basis vectors
-  j  - column number to be set

   Level: advanced

.seealso: BVSetRandomContext(), BVSetRandom(), BVSetRandomNormal(), BVSetRandomCond()
@*/
PetscErrorCode BVSetRandomColumn(BV bv,PetscInt j)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(bv,BV_CLASSID,1);
  PetscValidLogicalCollectiveInt(bv,j,2);
  PetscValidType(bv,1);
  BVCheckSizes(bv,1);
  PetscCheck(j>=0 && j<bv->m,PetscObjectComm((PetscObject)bv),PETSC_ERR_ARG_OUTOFRANGE,"Argument j has wrong value %" PetscInt_FMT ", the number of columns is %" PetscInt_FMT,j,bv->m);

  ierr = BVGetRandomContext(bv,&bv->rand);CHKERRQ(ierr);
  ierr = PetscLogEventBegin(BV_SetRandom,bv,0,0,0);CHKERRQ(ierr);
  ierr = BVSetRandomColumn_Private(bv,j);CHKERRQ(ierr);
  ierr = PetscLogEventEnd(BV_SetRandom,bv,0,0,0);CHKERRQ(ierr);
  ierr = PetscObjectStateIncrease((PetscObject)bv);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@
   BVSetRandomNormal - Set the columns of a BV to random numbers with a normal
   distribution.

   Logically Collective on bv

   Input Parameter:
.  bv - basis vectors

   Notes:
   All active columns (except the leading ones) are modified.

   Other functions such as BVSetRandom(), BVSetRandomColumn(), and BVSetRandomCond()
   produce random numbers with a uniform distribution. This function returns values
   that fit a normal distribution (Gaussian).

   Developer Notes:
   The current implementation obtains each of the columns by applying the Box-Muller
   transform on two random vectors with uniformly distributed entries.

   Level: advanced

.seealso: BVSetRandomContext(), BVSetRandom(), BVSetRandomColumn(), BVSetRandomCond(), BVSetActiveColumns()
@*/
PetscErrorCode BVSetRandomNormal(BV bv)
{
  PetscErrorCode ierr;
  PetscInt       k;
  Vec            w1=NULL,w2=NULL;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(bv,BV_CLASSID,1);
  PetscValidType(bv,1);
  BVCheckSizes(bv,1);

  ierr = BVGetRandomContext(bv,&bv->rand);CHKERRQ(ierr);
  if (!bv->rrandom) {
    ierr = BVCreateVec(bv,&w1);CHKERRQ(ierr);
    ierr = BVCreateVec(bv,&w2);CHKERRQ(ierr);
  }
  ierr = PetscLogEventBegin(BV_SetRandom,bv,0,0,0);CHKERRQ(ierr);
  for (k=bv->l;k<bv->k;k++) {
    ierr = BVSetRandomNormalColumn_Private(bv,k,w1,w2);CHKERRQ(ierr);
  }
  ierr = PetscLogEventEnd(BV_SetRandom,bv,0,0,0);CHKERRQ(ierr);
  if (!bv->rrandom) {
    ierr = VecDestroy(&w1);CHKERRQ(ierr);
    ierr = VecDestroy(&w2);CHKERRQ(ierr);
  }
  ierr = PetscObjectStateIncrease((PetscObject)bv);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@
   BVSetRandomSign - Set the entries of a BV to values 1 or -1 with equal
   probability.

   Logically Collective on bv

   Input Parameter:
.  bv - basis vectors

   Notes:
   All active columns (except the leading ones) are modified.

   This function is used, e.g., in contour integral methods when estimating
   the number of eigenvalues enclosed by the contour via an unbiased
   estimator of tr(f(A)) [Bai et al., JCAM 1996].

   Developer Notes:
   The current implementation obtains random numbers and then replaces them
   with -1 or 1 depending on the value being less than 0.5 or not.

   Level: advanced

.seealso: BVSetRandomContext(), BVSetRandom(), BVSetRandomColumn(), BVSetActiveColumns()
@*/
PetscErrorCode BVSetRandomSign(BV bv)
{
  PetscErrorCode ierr;
  PetscScalar    low,high;
  PetscInt       k;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(bv,BV_CLASSID,1);
  PetscValidType(bv,1);
  BVCheckSizes(bv,1);

  ierr = BVGetRandomContext(bv,&bv->rand);CHKERRQ(ierr);
  ierr = PetscRandomGetInterval(bv->rand,&low,&high);CHKERRQ(ierr);
  PetscCheck(PetscRealPart(low)==0.0 && PetscRealPart(high)==1.0,PetscObjectComm((PetscObject)bv),PETSC_ERR_ARG_WRONGSTATE,"The PetscRandom object in the BV must have interval [0,1]");
  ierr = PetscLogEventBegin(BV_SetRandom,bv,0,0,0);CHKERRQ(ierr);
  for (k=bv->l;k<bv->k;k++) {
    ierr = BVSetRandomSignColumn_Private(bv,k);CHKERRQ(ierr);
  }
  ierr = PetscLogEventEnd(BV_SetRandom,bv,0,0,0);CHKERRQ(ierr);
  ierr = PetscObjectStateIncrease((PetscObject)bv);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@
   BVSetRandomCond - Set the columns of a BV to random numbers, in a way that
   the generated matrix has a given condition number.

   Logically Collective on bv

   Input Parameters:
+  bv    - basis vectors
-  condn - condition number

   Note:
   All active columns (except the leading ones) are modified.

   Level: advanced

.seealso: BVSetRandomContext(), BVSetRandom(), BVSetRandomColumn(), BVSetRandomNormal(), BVSetActiveColumns()
@*/
PetscErrorCode BVSetRandomCond(BV bv,PetscReal condn)
{
  PetscErrorCode ierr;
  PetscInt       k,i;
  PetscScalar    *eig,*d;
  DS             ds;
  Mat            A,X,Xt,M,G;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(bv,BV_CLASSID,1);
  PetscValidType(bv,1);
  BVCheckSizes(bv,1);

  ierr = BVGetRandomContext(bv,&bv->rand);CHKERRQ(ierr);
  ierr = PetscLogEventBegin(BV_SetRandom,bv,0,0,0);CHKERRQ(ierr);
  /* B = rand(n,k) */
  for (k=bv->l;k<bv->k;k++) {
    ierr = BVSetRandomColumn_Private(bv,k);CHKERRQ(ierr);
  }
  ierr = DSCreate(PetscObjectComm((PetscObject)bv),&ds);CHKERRQ(ierr);
  ierr = DSSetType(ds,DSHEP);CHKERRQ(ierr);
  ierr = DSAllocate(ds,bv->m);CHKERRQ(ierr);
  ierr = DSSetDimensions(ds,bv->k,bv->l,bv->k);CHKERRQ(ierr);
  /* [V,S] = eig(B'*B) */
  ierr = DSGetMat(ds,DS_MAT_A,&A);CHKERRQ(ierr);
  ierr = BVDot(bv,bv,A);CHKERRQ(ierr);
  ierr = DSRestoreMat(ds,DS_MAT_A,&A);CHKERRQ(ierr);
  ierr = PetscMalloc1(bv->m,&eig);CHKERRQ(ierr);
  ierr = DSSolve(ds,eig,NULL);CHKERRQ(ierr);
  ierr = DSSynchronize(ds,eig,NULL);CHKERRQ(ierr);
  ierr = DSVectors(ds,DS_MAT_X,NULL,NULL);CHKERRQ(ierr);
  /* M = diag(linspace(1/condn,1,n)./sqrt(diag(S)))' */
  ierr = MatCreateSeqDense(PETSC_COMM_SELF,bv->k,bv->k,NULL,&M);CHKERRQ(ierr);
  ierr = MatZeroEntries(M);CHKERRQ(ierr);
  ierr = MatDenseGetArray(M,&d);CHKERRQ(ierr);
  for (i=0;i<bv->k;i++) d[i+i*bv->m] = (1.0/condn+(1.0-1.0/condn)/(bv->k-1)*i)/PetscSqrtScalar(eig[i]);
  ierr = MatDenseRestoreArray(M,&d);CHKERRQ(ierr);
  /* G = X*M*X' */
  ierr = MatCreateSeqDense(PETSC_COMM_SELF,bv->k,bv->k,NULL,&Xt);CHKERRQ(ierr);
  ierr = DSGetMat(ds,DS_MAT_X,&X);CHKERRQ(ierr);
  ierr = MatTranspose(X,MAT_REUSE_MATRIX,&Xt);CHKERRQ(ierr);
  ierr = MatProductCreate(Xt,M,NULL,&G);CHKERRQ(ierr);
  ierr = MatProductSetType(G,MATPRODUCT_PtAP);CHKERRQ(ierr);
  ierr = MatProductSetFromOptions(G);CHKERRQ(ierr);
  ierr = MatProductSymbolic(G);CHKERRQ(ierr);
  ierr = MatProductNumeric(G);CHKERRQ(ierr);
  ierr = MatProductClear(G);CHKERRQ(ierr);
  ierr = MatDestroy(&X);CHKERRQ(ierr);
  ierr = MatDestroy(&Xt);CHKERRQ(ierr);
  ierr = MatDestroy(&M);CHKERRQ(ierr);
  /* B = B*G */
  ierr = BVMultInPlace(bv,G,bv->l,bv->k);CHKERRQ(ierr);
  ierr = MatDestroy(&G);CHKERRQ(ierr);
  ierr = PetscFree(eig);CHKERRQ(ierr);
  ierr = DSDestroy(&ds);CHKERRQ(ierr);
  ierr = PetscLogEventEnd(BV_SetRandom,bv,0,0,0);CHKERRQ(ierr);
  ierr = PetscObjectStateIncrease((PetscObject)bv);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@
   BVMatMult - Computes the matrix-vector product for each column, Y=A*V.

   Neighbor-wise Collective on A

   Input Parameters:
+  V - basis vectors context
-  A - the matrix

   Output Parameter:
.  Y - the result

   Notes:
   Both V and Y must be distributed in the same manner. Only active columns
   (excluding the leading ones) are processed.
   In the result Y, columns are overwritten starting from the leading ones.
   The number of active columns in V and Y should match, although they need
   not be the same columns.

   It is possible to choose whether the computation is done column by column
   or as a Mat-Mat product, see BVSetMatMultMethod().

   Level: beginner

.seealso: BVCopy(), BVSetActiveColumns(), BVMatMultColumn(), BVMatMultTranspose(), BVMatMultHermitianTranspose(), BVSetMatMultMethod()
@*/
PetscErrorCode BVMatMult(BV V,Mat A,BV Y)
{
  PetscErrorCode ierr;
  PetscInt       M,N,m,n;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(V,BV_CLASSID,1);
  PetscValidType(V,1);
  BVCheckSizes(V,1);
  BVCheckOp(V,1,matmult);
  PetscValidHeaderSpecific(A,MAT_CLASSID,2);
  PetscValidType(A,2);
  PetscValidHeaderSpecific(Y,BV_CLASSID,3);
  PetscValidType(Y,3);
  BVCheckSizes(Y,3);
  PetscCheckSameComm(V,1,A,2);
  PetscCheckSameTypeAndComm(V,1,Y,3);

  ierr = MatGetSize(A,&M,&N);CHKERRQ(ierr);
  ierr = MatGetLocalSize(A,&m,&n);CHKERRQ(ierr);
  PetscCheck(M==Y->N,PetscObjectComm((PetscObject)Y),PETSC_ERR_ARG_INCOMP,"Mismatching row dimension A %" PetscInt_FMT ", Y %" PetscInt_FMT,M,Y->N);
  PetscCheck(m==Y->n,PetscObjectComm((PetscObject)Y),PETSC_ERR_ARG_INCOMP,"Mismatching local row dimension A %" PetscInt_FMT ", Y %" PetscInt_FMT,m,Y->n);
  PetscCheck(N==V->N,PetscObjectComm((PetscObject)V),PETSC_ERR_ARG_INCOMP,"Mismatching column dimension A %" PetscInt_FMT ", V %" PetscInt_FMT,N,V->N);
  PetscCheck(n==V->n,PetscObjectComm((PetscObject)V),PETSC_ERR_ARG_INCOMP,"Mismatching local column dimension A %" PetscInt_FMT ", V %" PetscInt_FMT,n,V->n);
  PetscCheck(V->k-V->l==Y->k-Y->l,PetscObjectComm((PetscObject)V),PETSC_ERR_ARG_SIZ,"Y has %" PetscInt_FMT " active columns, should match %" PetscInt_FMT " active columns in V",Y->k-Y->l,V->k-V->l);

  ierr = PetscLogEventBegin(BV_MatMult,V,A,Y,0);CHKERRQ(ierr);
  ierr = (*V->ops->matmult)(V,A,Y);CHKERRQ(ierr);
  ierr = PetscLogEventEnd(BV_MatMult,V,A,Y,0);CHKERRQ(ierr);
  ierr = PetscObjectStateIncrease((PetscObject)Y);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@
   BVMatMultTranspose - Computes the matrix-vector product with the transpose
   of a matrix for each column, Y=A^T*V.

   Neighbor-wise Collective on A

   Input Parameters:
+  V - basis vectors context
-  A - the matrix

   Output Parameter:
.  Y - the result

   Notes:
   Both V and Y must be distributed in the same manner. Only active columns
   (excluding the leading ones) are processed.
   In the result Y, columns are overwritten starting from the leading ones.
   The number of active columns in V and Y should match, although they need
   not be the same columns.

   Currently implemented via MatCreateTranspose().

   Level: beginner

.seealso: BVMatMult(), BVMatMultHermitianTranspose()
@*/
PetscErrorCode BVMatMultTranspose(BV V,Mat A,BV Y)
{
  PetscErrorCode ierr;
  PetscInt       M,N,m,n;
  Mat            AT;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(V,BV_CLASSID,1);
  PetscValidType(V,1);
  BVCheckSizes(V,1);
  PetscValidHeaderSpecific(A,MAT_CLASSID,2);
  PetscValidType(A,2);
  PetscValidHeaderSpecific(Y,BV_CLASSID,3);
  PetscValidType(Y,3);
  BVCheckSizes(Y,3);
  PetscCheckSameComm(V,1,A,2);
  PetscCheckSameTypeAndComm(V,1,Y,3);

  ierr = MatGetSize(A,&M,&N);CHKERRQ(ierr);
  ierr = MatGetLocalSize(A,&m,&n);CHKERRQ(ierr);
  PetscCheck(M==V->N,PetscObjectComm((PetscObject)V),PETSC_ERR_ARG_INCOMP,"Mismatching row dimension A %" PetscInt_FMT ", V %" PetscInt_FMT,M,V->N);
  PetscCheck(m==V->n,PetscObjectComm((PetscObject)V),PETSC_ERR_ARG_INCOMP,"Mismatching local row dimension A %" PetscInt_FMT ", V %" PetscInt_FMT,m,V->n);
  PetscCheck(N==Y->N,PetscObjectComm((PetscObject)Y),PETSC_ERR_ARG_INCOMP,"Mismatching column dimension A %" PetscInt_FMT ", Y %" PetscInt_FMT,N,Y->N);
  PetscCheck(n==Y->n,PetscObjectComm((PetscObject)Y),PETSC_ERR_ARG_INCOMP,"Mismatching local column dimension A %" PetscInt_FMT ", Y %" PetscInt_FMT,n,Y->n);
  PetscCheck(V->k-V->l==Y->k-Y->l,PetscObjectComm((PetscObject)V),PETSC_ERR_ARG_SIZ,"Y has %" PetscInt_FMT " active columns, should match %" PetscInt_FMT " active columns in V",Y->k-Y->l,V->k-V->l);

  ierr = MatCreateTranspose(A,&AT);CHKERRQ(ierr);
  ierr = BVMatMult(V,AT,Y);CHKERRQ(ierr);
  ierr = MatDestroy(&AT);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@
   BVMatMultHermitianTranspose - Computes the matrix-vector product with the
   conjugate transpose of a matrix for each column, Y=A^H*V.

   Neighbor-wise Collective on A

   Input Parameters:
+  V - basis vectors context
-  A - the matrix

   Output Parameter:
.  Y - the result

   Note:
   Both V and Y must be distributed in the same manner. Only active columns
   (excluding the leading ones) are processed.
   In the result Y, columns are overwritten starting from the leading ones.
   The number of active columns in V and Y should match, although they need
   not be the same columns.

   Currently implemented via MatCreateHermitianTranspose().

   Level: beginner

.seealso: BVMatMult(), BVMatMultTranspose()
@*/
PetscErrorCode BVMatMultHermitianTranspose(BV V,Mat A,BV Y)
{
  PetscErrorCode ierr;
  PetscInt       M,N,m,n;
  Mat            AH;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(V,BV_CLASSID,1);
  PetscValidType(V,1);
  BVCheckSizes(V,1);
  PetscValidHeaderSpecific(A,MAT_CLASSID,2);
  PetscValidType(A,2);
  PetscValidHeaderSpecific(Y,BV_CLASSID,3);
  PetscValidType(Y,3);
  BVCheckSizes(Y,3);
  PetscCheckSameComm(V,1,A,2);
  PetscCheckSameTypeAndComm(V,1,Y,3);

  ierr = MatGetSize(A,&M,&N);CHKERRQ(ierr);
  ierr = MatGetLocalSize(A,&m,&n);CHKERRQ(ierr);
  PetscCheck(M==V->N,PetscObjectComm((PetscObject)V),PETSC_ERR_ARG_INCOMP,"Mismatching row dimension A %" PetscInt_FMT ", V %" PetscInt_FMT,M,V->N);
  PetscCheck(m==V->n,PetscObjectComm((PetscObject)V),PETSC_ERR_ARG_INCOMP,"Mismatching local row dimension A %" PetscInt_FMT ", V %" PetscInt_FMT,m,V->n);
  PetscCheck(N==Y->N,PetscObjectComm((PetscObject)Y),PETSC_ERR_ARG_INCOMP,"Mismatching column dimension A %" PetscInt_FMT ", Y %" PetscInt_FMT,N,Y->N);
  PetscCheck(n==Y->n,PetscObjectComm((PetscObject)Y),PETSC_ERR_ARG_INCOMP,"Mismatching local column dimension A %" PetscInt_FMT ", Y %" PetscInt_FMT,n,Y->n);
  PetscCheck(V->k-V->l==Y->k-Y->l,PetscObjectComm((PetscObject)V),PETSC_ERR_ARG_SIZ,"Y has %" PetscInt_FMT " active columns, should match %" PetscInt_FMT " active columns in V",Y->k-Y->l,V->k-V->l);

  ierr = MatCreateHermitianTranspose(A,&AH);CHKERRQ(ierr);
  ierr = BVMatMult(V,AH,Y);CHKERRQ(ierr);
  ierr = MatDestroy(&AH);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@
   BVMatMultColumn - Computes the matrix-vector product for a specified
   column, storing the result in the next column v_{j+1}=A*v_j.

   Neighbor-wise Collective on A

   Input Parameters:
+  V - basis vectors context
.  A - the matrix
-  j - the column

   Level: beginner

.seealso: BVMatMult(), BVMatMultTransposeColumn(), BVMatMultHermitianTransposeColumn()
@*/
PetscErrorCode BVMatMultColumn(BV V,Mat A,PetscInt j)
{
  PetscErrorCode ierr;
  Vec            vj,vj1;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(V,BV_CLASSID,1);
  PetscValidType(V,1);
  BVCheckSizes(V,1);
  PetscValidHeaderSpecific(A,MAT_CLASSID,2);
  PetscCheckSameComm(V,1,A,2);
  PetscValidLogicalCollectiveInt(V,j,3);
  PetscCheck(j>=0,PetscObjectComm((PetscObject)V),PETSC_ERR_ARG_OUTOFRANGE,"Index j must be non-negative");
  PetscCheck(j+1<V->m,PetscObjectComm((PetscObject)V),PETSC_ERR_ARG_OUTOFRANGE,"Result should go in index j+1=%" PetscInt_FMT " but BV only has %" PetscInt_FMT " columns",j+1,V->m);

  ierr = PetscLogEventBegin(BV_MatMultVec,V,A,0,0);CHKERRQ(ierr);
  ierr = BVGetColumn(V,j,&vj);CHKERRQ(ierr);
  ierr = BVGetColumn(V,j+1,&vj1);CHKERRQ(ierr);
  ierr = MatMult(A,vj,vj1);CHKERRQ(ierr);
  ierr = BVRestoreColumn(V,j,&vj);CHKERRQ(ierr);
  ierr = BVRestoreColumn(V,j+1,&vj1);CHKERRQ(ierr);
  ierr = PetscLogEventEnd(BV_MatMultVec,V,A,0,0);CHKERRQ(ierr);
  ierr = PetscObjectStateIncrease((PetscObject)V);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@
   BVMatMultTransposeColumn - Computes the transpose matrix-vector product for a
   specified column, storing the result in the next column v_{j+1}=A^T*v_j.

   Neighbor-wise Collective on A

   Input Parameters:
+  V - basis vectors context
.  A - the matrix
-  j - the column

   Level: beginner

.seealso: BVMatMult(), BVMatMultColumn(), BVMatMultHermitianTransposeColumn()
@*/
PetscErrorCode BVMatMultTransposeColumn(BV V,Mat A,PetscInt j)
{
  PetscErrorCode ierr;
  Vec            vj,vj1;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(V,BV_CLASSID,1);
  PetscValidType(V,1);
  BVCheckSizes(V,1);
  PetscValidHeaderSpecific(A,MAT_CLASSID,2);
  PetscCheckSameComm(V,1,A,2);
  PetscValidLogicalCollectiveInt(V,j,3);
  PetscCheck(j>=0,PetscObjectComm((PetscObject)V),PETSC_ERR_ARG_OUTOFRANGE,"Index j must be non-negative");
  PetscCheck(j+1<V->m,PetscObjectComm((PetscObject)V),PETSC_ERR_ARG_OUTOFRANGE,"Result should go in index j+1=%" PetscInt_FMT " but BV only has %" PetscInt_FMT " columns",j+1,V->m);

  ierr = PetscLogEventBegin(BV_MatMultVec,V,A,0,0);CHKERRQ(ierr);
  ierr = BVGetColumn(V,j,&vj);CHKERRQ(ierr);
  ierr = BVGetColumn(V,j+1,&vj1);CHKERRQ(ierr);
  ierr = MatMultTranspose(A,vj,vj1);CHKERRQ(ierr);
  ierr = BVRestoreColumn(V,j,&vj);CHKERRQ(ierr);
  ierr = BVRestoreColumn(V,j+1,&vj1);CHKERRQ(ierr);
  ierr = PetscLogEventEnd(BV_MatMultVec,V,A,0,0);CHKERRQ(ierr);
  ierr = PetscObjectStateIncrease((PetscObject)V);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@
   BVMatMultHermitianTransposeColumn - Computes the conjugate-transpose matrix-vector
   product for a specified column, storing the result in the next column v_{j+1}=A^H*v_j.

   Neighbor-wise Collective on A

   Input Parameters:
+  V - basis vectors context
.  A - the matrix
-  j - the column

   Level: beginner

.seealso: BVMatMult(), BVMatMultColumn(), BVMatMultTransposeColumn()
@*/
PetscErrorCode BVMatMultHermitianTransposeColumn(BV V,Mat A,PetscInt j)
{
  PetscErrorCode ierr;
  Vec            vj,vj1;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(V,BV_CLASSID,1);
  PetscValidType(V,1);
  BVCheckSizes(V,1);
  PetscValidHeaderSpecific(A,MAT_CLASSID,2);
  PetscCheckSameComm(V,1,A,2);
  PetscValidLogicalCollectiveInt(V,j,3);
  PetscCheck(j>=0,PetscObjectComm((PetscObject)V),PETSC_ERR_ARG_OUTOFRANGE,"Index j must be non-negative");
  PetscCheck(j+1<V->m,PetscObjectComm((PetscObject)V),PETSC_ERR_ARG_OUTOFRANGE,"Result should go in index j+1=%" PetscInt_FMT " but BV only has %" PetscInt_FMT " columns",j+1,V->m);

  ierr = PetscLogEventBegin(BV_MatMultVec,V,A,0,0);CHKERRQ(ierr);
  ierr = BVGetColumn(V,j,&vj);CHKERRQ(ierr);
  ierr = BVGetColumn(V,j+1,&vj1);CHKERRQ(ierr);
  ierr = MatMultHermitianTranspose(A,vj,vj1);CHKERRQ(ierr);
  ierr = BVRestoreColumn(V,j,&vj);CHKERRQ(ierr);
  ierr = BVRestoreColumn(V,j+1,&vj1);CHKERRQ(ierr);
  ierr = PetscLogEventEnd(BV_MatMultVec,V,A,0,0);CHKERRQ(ierr);
  ierr = PetscObjectStateIncrease((PetscObject)V);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

