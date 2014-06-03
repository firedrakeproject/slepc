/*
   BV operations.

   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
   SLEPc - Scalable Library for Eigenvalue Problem Computations
   Copyright (c) 2002-2013, Universitat Politecnica de Valencia, Spain

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

#include <slepc-private/bvimpl.h>      /*I "slepcbv.h" I*/

#undef __FUNCT__
#define __FUNCT__ "BVMult"
/*@
   BVMult - Computes Y = beta*Y + alpha*X*Q.

   Logically Collective on BV

   Input Parameters:
+  Y,X        - basis vectors
.  alpha,beta - scalars
-  Q          - a sequential dense matrix

   Output Parameter:
.  Y          - the modified basis vectors

   Notes:
   X and Y must be different objects. The case X=Y can be addressed with
   BVMultInPlace().

   The matrix Q must be a sequential dense Mat, with all entries equal on
   all processes (otherwise each process will compute a different update).
   The dimensions of Q must be m,n where m is the number of active columns
   of X and n is the number of active columns of Y.

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
  PetscBool      match;
  PetscInt       m,n;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(Y,BV_CLASSID,1);
  PetscValidLogicalCollectiveScalar(Y,alpha,2);
  PetscValidLogicalCollectiveScalar(Y,beta,3);
  PetscValidHeaderSpecific(X,BV_CLASSID,4);
  PetscValidHeaderSpecific(Q,MAT_CLASSID,5);
  PetscValidType(Y,1);
  BVCheckSizes(Y,1);
  PetscValidType(X,4);
  BVCheckSizes(X,4);
  PetscValidType(Q,5);
  PetscCheckSameTypeAndComm(Y,1,X,4);
  if (X==Y) SETERRQ(PetscObjectComm((PetscObject)Y),PETSC_ERR_ARG_WRONG,"X and Y arguments must be different");
  ierr = PetscObjectTypeCompare((PetscObject)Q,MATSEQDENSE,&match);CHKERRQ(ierr);
  if (!match) SETERRQ(PetscObjectComm((PetscObject)Y),PETSC_ERR_SUP,"Mat argument must be of type seqdense");

  ierr = MatGetSize(Q,&m,&n);CHKERRQ(ierr);
  if (m!=X->k) SETERRQ2(PetscObjectComm((PetscObject)Y),PETSC_ERR_ARG_SIZ,"Mat argument has %D rows, cannot multiply a BV with %D active columns",m,X->k);
  if (n!=Y->k) SETERRQ2(PetscObjectComm((PetscObject)Y),PETSC_ERR_ARG_SIZ,"Mat argument has %D columns, result cannot be added to a BV with %D active columns",n,Y->k);
  if (X->n!=Y->n) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_ARG_INCOMP,"Mismatching local dimension X %D, Y %D",X->n,Y->n);
  if (!X->n) PetscFunctionReturn(0);

  ierr = PetscLogEventBegin(BV_Mult,X,Y,0,0);CHKERRQ(ierr);
  ierr = (*Y->ops->mult)(Y,alpha,beta,X,Q);CHKERRQ(ierr);
  ierr = PetscLogEventEnd(BV_Mult,X,Y,0,0);CHKERRQ(ierr);
  ierr = PetscObjectStateIncrease((PetscObject)Y);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "BVMultVec"
/*@
   BVMultVec - Computes y = beta*y + alpha*X*q.

   Logically Collective on BV and Vec

   Input Parameters:
+  X          - a basis vectors object
.  alpha,beta - scalars
.  y          - a vector
-  q          - an array of scalars

   Output Parameter:
.  y          - the modified vector

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
PetscErrorCode BVMultVec(BV X,PetscScalar alpha,PetscScalar beta,Vec y,PetscScalar *q)
{
  PetscErrorCode ierr;
  PetscInt       n,N;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(X,BV_CLASSID,1);
  PetscValidLogicalCollectiveScalar(X,alpha,2);
  PetscValidLogicalCollectiveScalar(X,beta,3);
  PetscValidHeaderSpecific(y,VEC_CLASSID,4);
  PetscValidPointer(q,5);
  PetscValidType(X,1);
  BVCheckSizes(X,1);
  PetscValidType(y,4);
  PetscCheckSameComm(X,1,y,4);

  ierr = VecGetSize(y,&N);CHKERRQ(ierr);
  ierr = VecGetLocalSize(y,&n);CHKERRQ(ierr);
  if (N!=X->N || n!=X->n) SETERRQ4(PetscObjectComm((PetscObject)X),PETSC_ERR_ARG_INCOMP,"Vec sizes (global %D, local %D) do not match BV sizes (global %D, local %D)",N,n,X->N,X->n);
  if (!X->n) PetscFunctionReturn(0);

  ierr = PetscLogEventBegin(BV_Mult,X,y,0,0);CHKERRQ(ierr);
  ierr = (*X->ops->multvec)(X,alpha,beta,y,q);CHKERRQ(ierr);
  ierr = PetscLogEventEnd(BV_Mult,X,y,0,0);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "BVMultColumn"
/*@
   BVMultColumn - Computes y = beta*y + alpha*X*q, where y is the j-th column
   of X.

   Logically Collective on BV

   Input Parameters:
+  X          - a basis vectors object
.  alpha,beta - scalars
.  j          - the column index
-  q          - an array of scalars

   Notes:
   This operation is equivalent to BVMultVec() but it uses column j of X
   rather than taking a Vec as an argument. The number of active columns of
   X is set to j before the computation, and restored afterwards.
   If X has leading columns specified, then these columns do not participate
   in the computation. Therefore, the length of array q must be equal to j
   minus the number of leading columns.

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
  PetscValidPointer(q,5);
  PetscValidType(X,1);
  BVCheckSizes(X,1);

  if (j<0) SETERRQ(PetscObjectComm((PetscObject)X),PETSC_ERR_ARG_OUTOFRANGE,"Index j must be non-negative");
  if (j>=X->m) SETERRQ2(PetscObjectComm((PetscObject)X),PETSC_ERR_ARG_OUTOFRANGE,"Index j=%D but BV only has %D columns",j,X->m);
  if (!X->n) PetscFunctionReturn(0);

  ierr = PetscLogEventBegin(BV_Mult,X,y,0,0);CHKERRQ(ierr);
  ksave = X->k;
  X->k = j;
  ierr = BVGetColumn(X,j,&y);CHKERRQ(ierr);
  ierr = (*X->ops->multvec)(X,alpha,beta,y,q);CHKERRQ(ierr);
  ierr = BVRestoreColumn(X,j,&y);CHKERRQ(ierr);
  X->k = ksave;
  ierr = PetscLogEventEnd(BV_Mult,X,y,0,0);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "BVMultInPlace"
/*@
   BVMultInPlace - Update a set of vectors as V(:,s:e-1) = V*Q(:,s:e-1).

   Logically Collective on BV

   Input Parameters:
+  Q - a sequential dense matrix
.  s - first column of V to be overwritten
-  e - first column of V not to be overwritten

   Input/Output Parameter:
+  V - basis vectors

   Notes:
   The matrix Q must be a sequential dense Mat, with all entries equal on
   all processes (otherwise each process will compute a different update).

   This function computes V(:,s:e-1) = V*Q(:,s:e-1), that is, given a set of
   vectors V, columns from s to e-1 are overwritten with columns from s to
   e-1 of the matrix-matrix product V*Q. Only columns s to e-1 of Q are
   referenced.

   Level: intermediate

.seealso: BVMult(), BVMultVec(), BVMultInPlaceTranspose(), BVSetActiveColumns()
@*/
PetscErrorCode BVMultInPlace(BV V,Mat Q,PetscInt s,PetscInt e)
{
  PetscErrorCode ierr;
  PetscBool      match;
  PetscInt       m,n;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(V,BV_CLASSID,1);
  PetscValidHeaderSpecific(Q,MAT_CLASSID,2);
  PetscValidLogicalCollectiveInt(V,s,3);
  PetscValidLogicalCollectiveInt(V,e,4);
  PetscValidType(V,1);
  BVCheckSizes(V,1);
  PetscValidType(Q,2);
  ierr = PetscObjectTypeCompare((PetscObject)Q,MATSEQDENSE,&match);CHKERRQ(ierr);
  if (!match) SETERRQ(PetscObjectComm((PetscObject)V),PETSC_ERR_SUP,"Mat argument must be of type seqdense");

  if (s<V->l || s>=V->k) SETERRQ3(PetscObjectComm((PetscObject)V),PETSC_ERR_ARG_OUTOFRANGE,"Argument s has wrong value %D, should be between %D and %D",s,V->l,V->k-1);
  if (e<V->l || e>V->k) SETERRQ3(PetscObjectComm((PetscObject)V),PETSC_ERR_ARG_OUTOFRANGE,"Argument e has wrong value %D, should be between %D and %D",e,V->l,V->k);
  ierr = MatGetSize(Q,&m,&n);CHKERRQ(ierr);
  if (m!=V->k) SETERRQ2(PetscObjectComm((PetscObject)V),PETSC_ERR_ARG_SIZ,"Mat argument has %D rows, cannot multiply a BV with %D active columns",m,V->k);
  if (e>n) SETERRQ2(PetscObjectComm((PetscObject)V),PETSC_ERR_ARG_SIZ,"Mat argument only has %D columns, the requested value of e is larger: %D",n,e);
  if (s>=e || !V->n) PetscFunctionReturn(0);

  ierr = PetscLogEventBegin(BV_Mult,V,Q,0,0);CHKERRQ(ierr);
  ierr = (*V->ops->multinplace)(V,Q,s,e);CHKERRQ(ierr);
  ierr = PetscLogEventEnd(BV_Mult,V,Q,0,0);CHKERRQ(ierr);
  ierr = PetscObjectStateIncrease((PetscObject)V);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "BVMultInPlaceTranspose"
/*@
   BVMultInPlaceTranspose - Update a set of vectors as V(:,s:e-1) = V*Q'(:,s:e-1).

   Logically Collective on BV

   Input Parameters:
+  Q - a sequential dense matrix
.  s - first column of V to be overwritten
-  e - first column of V not to be overwritten

   Input/Output Parameter:
+  V - basis vectors

   Notes:
   This is a variant of BVMultInPlace() where the conjugate transpose
   of Q is used.

   Level: intermediate

.seealso: BVMultInPlace()
@*/
PetscErrorCode BVMultInPlaceTranspose(BV V,Mat Q,PetscInt s,PetscInt e)
{
  PetscErrorCode ierr;
  PetscBool      match;
  PetscInt       m,n;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(V,BV_CLASSID,1);
  PetscValidHeaderSpecific(Q,MAT_CLASSID,2);
  PetscValidLogicalCollectiveInt(V,s,3);
  PetscValidLogicalCollectiveInt(V,e,4);
  PetscValidType(V,1);
  BVCheckSizes(V,1);
  PetscValidType(Q,2);
  ierr = PetscObjectTypeCompare((PetscObject)Q,MATSEQDENSE,&match);CHKERRQ(ierr);
  if (!match) SETERRQ(PetscObjectComm((PetscObject)V),PETSC_ERR_SUP,"Mat argument must be of type seqdense");

  if (s<V->l || s>=V->k) SETERRQ3(PetscObjectComm((PetscObject)V),PETSC_ERR_ARG_OUTOFRANGE,"Argument s has wrong value %D, should be between %D and %D",s,V->l,V->k-1);
  if (e<V->l || e>V->k) SETERRQ3(PetscObjectComm((PetscObject)V),PETSC_ERR_ARG_OUTOFRANGE,"Argument e has wrong value %D, should be between %D and %D",e,V->l,V->k);
  ierr = MatGetSize(Q,&m,&n);CHKERRQ(ierr);
  if (n!=V->k) SETERRQ2(PetscObjectComm((PetscObject)V),PETSC_ERR_ARG_SIZ,"Mat argument has %D rows, cannot multiply a BV with %D active columns",n,V->k);
  if (e>m) SETERRQ2(PetscObjectComm((PetscObject)V),PETSC_ERR_ARG_SIZ,"Mat argument only has %D columns, the requested value of e is larger: %D",m,e);
  if (s>=e || !V->n) PetscFunctionReturn(0);

  ierr = PetscLogEventBegin(BV_Mult,V,Q,0,0);CHKERRQ(ierr);
  ierr = (*V->ops->multinplacetrans)(V,Q,s,e);CHKERRQ(ierr);
  ierr = PetscLogEventEnd(BV_Mult,V,Q,0,0);CHKERRQ(ierr);
  ierr = PetscObjectStateIncrease((PetscObject)V);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "BVDot_Private"
/*
  BVDot for the particular case of non-standard inner product with
  matrix B, which is assumed to be symmetric (or complex Hermitian)
*/
PETSC_STATIC_INLINE PetscErrorCode BVDot_Private(BV X,BV Y,Mat M)
{
  PetscErrorCode ierr;
  PetscObjectId  idx,idy;
  PetscInt       i,j,jend,m;
  PetscScalar    *marray;
  PetscBool      symm=PETSC_FALSE;
  Vec            z;

  PetscFunctionBegin;
  ierr = MatGetSize(M,&m,NULL);CHKERRQ(ierr);
  ierr = MatDenseGetArray(M,&marray);CHKERRQ(ierr);
  ierr = PetscObjectGetId((PetscObject)X,&idx);CHKERRQ(ierr);
  ierr = PetscObjectGetId((PetscObject)Y,&idy);CHKERRQ(ierr);
  if (idx==idy) symm=PETSC_TRUE;  /* M=X'BX is symmetric */
  jend = X->k;
  for (j=X->l;j<jend;j++) {
    if (symm) Y->k = j+1;
    ierr = BVGetColumn(X,j,&z);CHKERRQ(ierr);
    ierr = (*Y->ops->dotvec)(Y,z,marray+j*m+Y->l);CHKERRQ(ierr);
    ierr = BVRestoreColumn(X,j,&z);CHKERRQ(ierr);
    if (symm) {
      for (i=X->l;i<j;i++)
        marray[j+i*m] = PetscConj(marray[i+j*m]);
    }
  }
  ierr = MatDenseRestoreArray(M,&marray);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "BVDot"
/*@
   BVDot - Computes the 'block-dot' product of two basis vectors objects.

   Collective on BV

   Input Parameters:
+  X, Y - basis vectors
-  M    - Mat object where the result must be placed

   Output Parameter:
.  M    - the resulting matrix

   Notes:
   This is the generalization of VecDot() for a collection of vectors, M = Y^H*X.
   The result is a matrix M whose entry m_ij is equal to y_i^H x_j (where y_i^H
   denotes the conjugate transpose of y_i).

   If a non-standard inner product has been specified with BVSetMatrix(),
   then the result is M = Y^H*B*X. In this case, both X and Y must have
   the same associated matrix.

   On entry, M must be a sequential dense Mat with dimensions m,n where
   m is the number of active columns of Y and n is the number of active columns of X.
   Only rows (resp. columns) of M starting from ly (resp. lx) are computed,
   where ly (resp. lx) is the number of leading columns of Y (resp. X).

   X and Y need not be different objects.

   Level: intermediate

.seealso: BVDotVec(), BVDotColumn(), BVSetActiveColumns(), BVSetMatrix()
@*/
PetscErrorCode BVDot(BV X,BV Y,Mat M)
{
  PetscErrorCode ierr;
  PetscBool      match;
  PetscInt       m,n;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(X,BV_CLASSID,1);
  PetscValidHeaderSpecific(Y,BV_CLASSID,2);
  PetscValidHeaderSpecific(M,MAT_CLASSID,3);
  PetscValidType(X,1);
  BVCheckSizes(X,1);
  PetscValidType(Y,2);
  BVCheckSizes(Y,2);
  PetscValidType(M,3);
  PetscCheckSameTypeAndComm(X,1,Y,2);
  ierr = PetscObjectTypeCompare((PetscObject)M,MATSEQDENSE,&match);CHKERRQ(ierr);
  if (!match) SETERRQ(PetscObjectComm((PetscObject)X),PETSC_ERR_SUP,"Mat argument must be of type seqdense");

  ierr = MatGetSize(M,&m,&n);CHKERRQ(ierr);
  if (m!=Y->k) SETERRQ2(PetscObjectComm((PetscObject)X),PETSC_ERR_ARG_SIZ,"Mat argument has %D rows, should be %D",m,Y->k);
  if (n!=X->k) SETERRQ2(PetscObjectComm((PetscObject)X),PETSC_ERR_ARG_SIZ,"Mat argument has %D columns, should be %D",n,X->k);
  if (X->n!=Y->n) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_ARG_INCOMP,"Mismatching local dimension X %D, Y %D",X->n,Y->n);
  if (X->matrix!=Y->matrix) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONGSTATE,"X and Y must have the same inner product matrix");

  ierr = PetscLogEventBegin(BV_Dot,X,Y,0,0);CHKERRQ(ierr);
  if (X->matrix) { /* non-standard inner product: cast into dotvec ops */
    ierr = BVDot_Private(X,Y,M);CHKERRQ(ierr);
  } else {
    ierr = (*X->ops->dot)(X,Y,M);CHKERRQ(ierr);
  }
  ierr = PetscLogEventEnd(BV_Dot,X,Y,0,0);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "BVDotVec"
/*@
   BVDotVec - Computes multiple dot products of a vector against all the
   column vectors of a BV.

   Collective on BV and Vec

   Input Parameters:
+  X - basis vectors
-  y - a vector

   Output Parameter:
.  m - an array where the result must be placed

   Notes:
   This is analogue to VecMDot(), but using BV to represent a collection
   of vectors. The result is m = X^H*y, so m_i is equal to x_j^H y. Note
   that here X is transposed as opposed to BVDot().

   If a non-standard inner product has been specified with BVSetMatrix(),
   then the result is m = X^H*B*y.

   The length of array m must be equal to the number of active columns of X
   minus the number of leading columns, i.e. the first entry of m is the
   product of the first non-leading column with y.

   Level: intermediate

.seealso: BVDot(), BVDotColumn(), BVSetActiveColumns(), BVSetMatrix()
@*/
PetscErrorCode BVDotVec(BV X,Vec y,PetscScalar *m)
{
  PetscErrorCode ierr;
  PetscInt       n;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(X,BV_CLASSID,1);
  PetscValidHeaderSpecific(y,VEC_CLASSID,2);
  PetscValidType(X,1);
  BVCheckSizes(X,1);
  PetscValidType(y,2);
  PetscCheckSameTypeAndComm(X,1,y,2);

  ierr = VecGetLocalSize(y,&n);CHKERRQ(ierr);
  if (X->n!=n) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_ARG_INCOMP,"Mismatching local dimension X %D, y %D",X->n,n);

  ierr = PetscLogEventBegin(BV_Dot,X,y,0,0);CHKERRQ(ierr);
  ierr = (*X->ops->dotvec)(X,y,m);CHKERRQ(ierr);
  ierr = PetscLogEventEnd(BV_Dot,X,y,0,0);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "BVDotColumn"
/*@
   BVDotColumn - Computes multiple dot products of a column against all the
   previous columns of a BV.

   Collective on BV

   Input Parameters:
+  X - basis vectors
-  j - the column index

   Output Parameter:
.  m - an array where the result must be placed

   Notes:
   This operation is equivalent to BVDotVec() but it uses column j of X
   rather than taking a Vec as an argument. The number of active columns of
   X is set to j before the computation, and restored afterwards.
   If X has leading columns specified, then these columns do not participate
   in the computation. Therefore, the length of array m must be equal to j
   minus the number of leading columns.

   Level: advanced

.seealso: BVDot(), BVDotVec(), BVSetActiveColumns(), BVSetMatrix()
@*/
PetscErrorCode BVDotColumn(BV X,PetscInt j,PetscScalar *m)
{
  PetscErrorCode ierr;
  PetscInt       ksave;
  Vec            y;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(X,BV_CLASSID,1);
  PetscValidLogicalCollectiveInt(X,j,2);
  PetscValidType(X,1);
  BVCheckSizes(X,1);

  if (j<0) SETERRQ(PetscObjectComm((PetscObject)X),PETSC_ERR_ARG_OUTOFRANGE,"Index j must be non-negative");
  if (j>=X->m) SETERRQ2(PetscObjectComm((PetscObject)X),PETSC_ERR_ARG_OUTOFRANGE,"Index j=%D but BV only has %D columns",j,X->m);

  ierr = PetscLogEventBegin(BV_Dot,X,y,0,0);CHKERRQ(ierr);
  ksave = X->k;
  X->k = j;
  ierr = BVGetColumn(X,j,&y);CHKERRQ(ierr);
  ierr = (*X->ops->dotvec)(X,y,m);CHKERRQ(ierr);
  ierr = BVRestoreColumn(X,j,&y);CHKERRQ(ierr);
  X->k = ksave;
  ierr = PetscLogEventEnd(BV_Dot,X,y,0,0);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "BVScale"
/*@
   BVScale - Scale all columns of a BV.

   Logically Collective on BV

   Input Parameters:
+  bv    - basis vectors
-  alpha - scaling factor

   Note:
   All active columns are scaled.

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
  if (!bv->n || alpha == (PetscScalar)1.0) PetscFunctionReturn(0);

  ierr = PetscLogEventBegin(BV_Scale,bv,0,0,0);CHKERRQ(ierr);
  ierr = (*bv->ops->scale)(bv,-1,alpha);CHKERRQ(ierr);
  ierr = PetscLogEventEnd(BV_Scale,bv,0,0,0);CHKERRQ(ierr);
  ierr = PetscObjectStateIncrease((PetscObject)bv);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "BVScaleColumn"
/*@
   BVScaleColumn - Scale one column of a BV.

   Logically Collective on BV

   Input Parameters:
+  bv    - basis vectors
-  j     - column number to be scaled
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

  if (j<0 || j>=bv->m) SETERRQ2(PetscObjectComm((PetscObject)bv),PETSC_ERR_ARG_OUTOFRANGE,"Argument j has wrong value %D, the number of columns is %D",j,bv->m);
  if (!bv->n || alpha == (PetscScalar)1.0) PetscFunctionReturn(0);

  ierr = PetscLogEventBegin(BV_Scale,bv,0,0,0);CHKERRQ(ierr);
  ierr = (*bv->ops->scale)(bv,j,alpha);CHKERRQ(ierr);
  ierr = PetscLogEventEnd(BV_Scale,bv,0,0,0);CHKERRQ(ierr);
  ierr = PetscObjectStateIncrease((PetscObject)bv);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "BVNorm_Private"
PETSC_STATIC_INLINE PetscErrorCode BVNorm_Private(BV bv,Vec z,NormType type,PetscReal *val)
{
  PetscErrorCode ierr;
  PetscScalar    p;

  PetscFunctionBegin;
  if (type==NORM_1_AND_2) SETERRQ(PetscObjectComm((PetscObject)bv),PETSC_ERR_SUP,"Requested norm not available");
  ierr = BV_IPMatMult(bv,z);CHKERRQ(ierr);
  ierr = VecDot(bv->Bx,z,&p);CHKERRQ(ierr);
  if (PetscAbsScalar(p)<PETSC_MACHINE_EPSILON)
    ierr = PetscInfo(bv,"Zero norm, either the vector is zero or a semi-inner product is being used\n");CHKERRQ(ierr);
  if (bv->indef) {
    if (PetscAbsReal(PetscImaginaryPart(p))/PetscAbsScalar(p)>PETSC_MACHINE_EPSILON) SETERRQ(PetscObjectComm((PetscObject)bv),1,"BVNorm: The inner product is not well defined");
    if (PetscRealPart(p)<0.0) *val = -PetscSqrtScalar(-PetscRealPart(p));
    else *val = PetscSqrtScalar(PetscRealPart(p));
  } else { 
    if (PetscRealPart(p)<0.0 || PetscAbsReal(PetscImaginaryPart(p))/PetscAbsScalar(p)>PETSC_MACHINE_EPSILON) SETERRQ(PetscObjectComm((PetscObject)bv),1,"BVNorm: The inner product is not well defined");
    *val = PetscSqrtScalar(PetscRealPart(p));
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "BVNorm"
/*@
   BVNorm - Computes the matrix norm of all columns.

   Collective on BV

   Input Parameters:
+  bv   - basis vectors
-  type - the norm type

   Output Parameter:
.  val  - the norm

   Notes:
   All active columns are considered as a matrix. The allowed norms
   are NORM_1, NORM_FROBENIUS, and NORM_INFINITY.

   This operation fails if a non-standard inner product has been
   specified with BVSetMatrix().

   Level: intermediate

.seealso: BVNormVec(), BVNormColumn(), BVSetActiveColumns(), BVSetMatrix()
@*/
PetscErrorCode BVNorm(BV bv,NormType type,PetscReal *val)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(bv,BV_CLASSID,1);
  PetscValidLogicalCollectiveEnum(bv,type,2);
  PetscValidPointer(val,3);
  PetscValidType(bv,1);
  BVCheckSizes(bv,1);

  if (type==NORM_2) SETERRQ(PetscObjectComm((PetscObject)bv),PETSC_ERR_SUP,"Requested norm not available");
  if (bv->matrix) SETERRQ(PetscObjectComm((PetscObject)bv),PETSC_ERR_SUP,"Matrix norm not available for non-standard inner product");

  ierr = PetscLogEventBegin(BV_Norm,bv,0,0,0);CHKERRQ(ierr);
  ierr = (*bv->ops->norm)(bv,-1,type,val);CHKERRQ(ierr);
  ierr = PetscLogEventEnd(BV_Norm,bv,0,0,0);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "BVNormVec"
/*@
   BVNormVec - Computes the norm of a given vector.

   Collective on BV

   Input Parameters:
+  bv   - basis vectors
-  v    - the vector
-  type - the norm type

   Output Parameter:
.  val  - the norm

   Notes:
   This is the analogue of BVNormColumn() but for a vector that is not in the BV.
   If a non-standard inner product has been specified with BVSetMatrix(),
   then the returned value is sqrt(v'*B*v), where B is the inner product
   matrix (argument 'type' is ignored). Otherwise, VecNorm() is called.

   Level: developer

.seealso: BVNorm(), BVNormColumn(), BVSetMatrix()
@*/
PetscErrorCode BVNormVec(BV bv,Vec v,NormType type,PetscReal *val)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(bv,BV_CLASSID,1);
  PetscValidHeaderSpecific(v,VEC_CLASSID,2);
  PetscValidLogicalCollectiveEnum(bv,type,3);
  PetscValidPointer(val,4);
  PetscValidType(bv,1);
  BVCheckSizes(bv,1);
  PetscCheckSameComm(bv,1,v,2);

  ierr = PetscLogEventBegin(BV_Norm,bv,0,0,0);CHKERRQ(ierr);
  if (bv->matrix) { /* non-standard inner product */
    ierr = BVNorm_Private(bv,v,type,val);CHKERRQ(ierr);
  } else {
    ierr = VecNorm(v,type,val);CHKERRQ(ierr);
  }
  ierr = PetscLogEventEnd(BV_Norm,bv,0,0,0);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "BVNormColumn"
/*@
   BVNormColumn - Computes the vector norm of a selected column.

   Collective on BV

   Input Parameters:
+  bv   - basis vectors
-  j    - column number to be used
-  type - the norm type

   Output Parameter:
.  val  - the norm

   Notes:
   The norm of V[j] is computed (NORM_1, NORM_2, or NORM_INFINITY).
   If a non-standard inner product has been specified with BVSetMatrix(),
   then the returned value is sqrt(V[j]'*B*V[j]), 
   where B is the inner product matrix (argument 'type' is ignored).

   Level: intermediate

.seealso: BVNorm(), BVNormVec(), BVSetActiveColumns(), BVSetMatrix()
@*/
PetscErrorCode BVNormColumn(BV bv,PetscInt j,NormType type,PetscReal *val)
{
  PetscErrorCode ierr;
  Vec            z;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(bv,BV_CLASSID,1);
  PetscValidLogicalCollectiveInt(bv,j,2);
  PetscValidLogicalCollectiveEnum(bv,type,3);
  PetscValidPointer(val,4);
  PetscValidType(bv,1);
  BVCheckSizes(bv,1);
  if (j<0 || j>=bv->m) SETERRQ2(PetscObjectComm((PetscObject)bv),PETSC_ERR_ARG_OUTOFRANGE,"Argument j has wrong value %D, the number of columns is %D",j,bv->m);

  ierr = PetscLogEventBegin(BV_Norm,bv,0,0,0);CHKERRQ(ierr);
  if (bv->matrix) { /* non-standard inner product */
    ierr = BVGetColumn(bv,j,&z);CHKERRQ(ierr);
    ierr = BVNorm_Private(bv,z,type,val);CHKERRQ(ierr);
    ierr = BVRestoreColumn(bv,j,&z);CHKERRQ(ierr);
  } else {
    ierr = (*bv->ops->norm)(bv,j,type,val);CHKERRQ(ierr);
  }
  ierr = PetscLogEventEnd(BV_Norm,bv,0,0,0);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "BVSetRandom"
/*@
   BVSetRandom - Set all columns of a BV to random numbers.

   Logically Collective on BV

   Input Parameters:
+  bv    - basis vectors
-  rctx - the random number context, formed by PetscRandomCreate(), or NULL and
          it will create one internally.

   Note:
   All active columns are modified.

   Level: advanced

.seealso: BVSetRandomColumn(), BVSetActiveColumns()
@*/
PetscErrorCode BVSetRandom(BV bv,PetscRandom rctx)
{
  PetscErrorCode ierr;
  PetscRandom    rand=NULL;
  PetscInt       i,low,high,k;
  PetscScalar    *px,t;
  Vec            x;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(bv,BV_CLASSID,1);
  if (rctx) PetscValidHeaderSpecific(rctx,PETSC_RANDOM_CLASSID,2);
  else {
    ierr = PetscRandomCreate(PetscObjectComm((PetscObject)bv),&rand);CHKERRQ(ierr);
    ierr = PetscRandomSetSeed(rand,0x12345678);CHKERRQ(ierr);
    ierr = PetscRandomSetFromOptions(rand);CHKERRQ(ierr);
    rctx = rand;
  }
  PetscValidType(bv,1);
  BVCheckSizes(bv,1);

  ierr = PetscLogEventBegin(BV_SetRandom,bv,rctx,0,0);CHKERRQ(ierr);
  for (k=0;k<bv->k;k++) {
    ierr = BVGetColumn(bv,k,&x);CHKERRQ(ierr);
    ierr = VecGetOwnershipRange(x,&low,&high);CHKERRQ(ierr);
    ierr = VecGetArray(x,&px);CHKERRQ(ierr);
    for (i=0;i<bv->N;i++) {
      ierr = PetscRandomGetValue(rctx,&t);CHKERRQ(ierr);
      if (i>=low && i<high) px[i-low] = t;
    }
    ierr = VecRestoreArray(x,&px);CHKERRQ(ierr);
    ierr = BVRestoreColumn(bv,k,&x);CHKERRQ(ierr);
  }
  ierr = PetscLogEventEnd(BV_SetRandom,bv,rctx,0,0);CHKERRQ(ierr);
  ierr = PetscRandomDestroy(&rand);CHKERRQ(ierr);
  ierr = PetscObjectStateIncrease((PetscObject)bv);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "BVSetRandomColumn"
/*@
   BVSetRandomColumn - Set one column of a BV to random numbers.

   Logically Collective on BV

   Input Parameters:
+  bv    - basis vectors
-  j     - column number to be set
-  rctx - the random number context, formed by PetscRandomCreate(), or NULL and
          it will create one internally.

   Note:
   This operation is analogue to VecSetRandom - the difference is that the
   generated random vector is the same irrespective of the size of the
   communicator (if all processes pass a PetscRandom context initialized
   with the same seed).

   Level: advanced

.seealso: BVSetRandom(), BVSetActiveColumns()
@*/
PetscErrorCode BVSetRandomColumn(BV bv,PetscInt j,PetscRandom rctx)
{
  PetscErrorCode ierr;
  PetscRandom    rand=NULL;
  PetscInt       i,low,high;
  PetscScalar    *px,t;
  Vec            x;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(bv,BV_CLASSID,1);
  PetscValidLogicalCollectiveInt(bv,j,2);
  if (rctx) PetscValidHeaderSpecific(rctx,PETSC_RANDOM_CLASSID,3);
  else {
    ierr = PetscRandomCreate(PetscObjectComm((PetscObject)bv),&rand);CHKERRQ(ierr);
    ierr = PetscRandomSetSeed(rand,0x12345678);CHKERRQ(ierr);
    ierr = PetscRandomSetFromOptions(rand);CHKERRQ(ierr);
    rctx = rand;
  }
  PetscValidType(bv,1);
  BVCheckSizes(bv,1);
  if (j<0 || j>=bv->m) SETERRQ2(PetscObjectComm((PetscObject)bv),PETSC_ERR_ARG_OUTOFRANGE,"Argument j has wrong value %D, the number of columns is %D",j,bv->m);

  ierr = PetscLogEventBegin(BV_SetRandom,bv,rctx,0,0);CHKERRQ(ierr);
  ierr = BVGetColumn(bv,j,&x);CHKERRQ(ierr);
  ierr = VecGetOwnershipRange(x,&low,&high);CHKERRQ(ierr);
  ierr = VecGetArray(x,&px);CHKERRQ(ierr);
  for (i=0;i<bv->N;i++) {
    ierr = PetscRandomGetValue(rctx,&t);CHKERRQ(ierr);
    if (i>=low && i<high) px[i-low] = t;
  }
  ierr = VecRestoreArray(x,&px);CHKERRQ(ierr);
  ierr = BVRestoreColumn(bv,j,&x);CHKERRQ(ierr);
  ierr = PetscLogEventEnd(BV_SetRandom,bv,rctx,0,0);CHKERRQ(ierr);
  ierr = PetscRandomDestroy(&rand);CHKERRQ(ierr);
  ierr = PetscObjectStateIncrease((PetscObject)bv);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "BVMatMult"
/*@
   BVMatMult - Computes the matrix-vector product for each column, Y=A*X.

   Neighbor-wise Collective on Mat and BV

   Input Parameters:
+  V - basis vectors context
-  A - the matrix

   Output Parameter:
.  Y - the result

   Note:
   Both V and Y must be distributed in the same manner. Only active columns
   (excluding the leading ones) are processed.
   In the result Y, columns are overwritten starting from the leading ones.

   Level: beginner

.seealso: BVCopy(), BVSetActiveColumns(), BVMatMultColumn()
@*/
PetscErrorCode BVMatMult(BV V,Mat A,BV Y)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(V,BV_CLASSID,1);
  PetscValidType(V,1);
  BVCheckSizes(V,1);
  PetscValidHeaderSpecific(A,MAT_CLASSID,2);
  PetscValidHeaderSpecific(Y,BV_CLASSID,3);
  PetscValidType(Y,3);
  BVCheckSizes(Y,3);
  PetscCheckSameComm(V,1,A,2);
  PetscCheckSameTypeAndComm(V,1,Y,3);
  if (V->n!=Y->n) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_ARG_INCOMP,"Mismatching local dimension V %D, Y %D",V->n,Y->n);
  if (V->k-V->l>Y->m-Y->l) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_ARG_SIZ,"Y has %D non-leading columns, not enough to store %D columns",Y->m-Y->l,V->k-V->l);
  if (!V->n) PetscFunctionReturn(0);

  ierr = PetscLogEventBegin(BV_MatMult,V,A,Y,0);CHKERRQ(ierr);
  ierr = (*V->ops->matmult)(V,A,Y);CHKERRQ(ierr);
  ierr = PetscLogEventEnd(BV_MatMult,V,A,Y,0);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "BVMatMultColumn"
/*@
   BVMatMultColumn - Computes the matrix-vector product for a specified
   column, storing the result in the next column: v_{j+1}=A*v_j.

   Neighbor-wise Collective on Mat and BV

   Input Parameters:
+  V - basis vectors context
.  A - the matrix
-  j - the column

   Output Parameter:
.  Y - the result

   Level: beginner

.seealso: BVMatMult()
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
  if (j<0) SETERRQ(PetscObjectComm((PetscObject)V),PETSC_ERR_ARG_OUTOFRANGE,"Index j must be non-negative");
  if (j+1>=V->m) SETERRQ2(PetscObjectComm((PetscObject)V),PETSC_ERR_ARG_OUTOFRANGE,"Result should go in index j+1=%D but BV only has %D columns",j+1,V->m);

  ierr = PetscLogEventBegin(BV_MatMult,V,A,0,0);CHKERRQ(ierr);
  ierr = BVGetColumn(V,j,&vj);CHKERRQ(ierr);
  ierr = BVGetColumn(V,j+1,&vj1);CHKERRQ(ierr);
  ierr = MatMult(A,vj,vj1);CHKERRQ(ierr);
  ierr = BVRestoreColumn(V,j,&vj);CHKERRQ(ierr);
  ierr = BVRestoreColumn(V,j+1,&vj1);CHKERRQ(ierr);
  ierr = PetscLogEventEnd(BV_MatMult,V,A,0,0);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "BVMatProject"
/*@
   BVMatProject - Computes the projection of a matrix onto a subspace.

   Collective on BV

   Input Parameters:
+  X - basis vectors
.  A - (optional) matrix to be projected
.  Y - left basis vectors, can be equal to X
-  M - Mat object where the result must be placed

   Output Parameter:
.  M - the resulting matrix

   Notes:
   If A=NULL, then it is assumed that X already contains A*X.

   This operation is similar to BVDot(), with important differences.
   The goal is to compute the matrix resulting from the orthogonal projection
   of A onto the subspace spanned by the columns of X, M = X^H*A*X, or the
   oblique projection onto X along Y, M = Y^H*A*X.

   A difference with respect to BVDot() is that the standard inner product
   is always used, regardless of a non-standard inner product being specified
   with BVSetMatrix().

   On entry, M must be a sequential dense Mat with dimensions ky,kx where
   ky (resp. kx) is the number of active columns of Y (resp. X).
   Another difference with respect to BVDot() is that all entries of M are
   computed except the leading ly,lx part, where ly (resp. lx) is the
   number of leading columns of Y (resp. X). Hence, the leading columns of
   X and Y participate in the computation, as opposed to BVDot().
   The leading part of M is assumed to be already available from previous
   computations.

   In the orthogonal projection case, Y=X, some computation can be saved if
   A is real symmetric (or complex Hermitian). In order to exploit this
   property, the symmetry flag of A must be set with MatSetOption().

   Level: intermediate

.seealso: BVDot(), BVSetActiveColumns(), BVSetMatrix()
@*/
PetscErrorCode BVMatProject(BV X,Mat A,BV Y,Mat M)
{
  PetscErrorCode ierr;
  PetscBool      match,set,flg,symm=PETSC_FALSE;
  PetscInt       i,j,m,n,lx,ly,kx,ky,ulim;
  PetscScalar    *marray,*harray;
  Vec            z,f;
  Mat            matrix,H;
  PetscObjectId  idx,idy;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(X,BV_CLASSID,1);
  if (A) PetscValidHeaderSpecific(A,MAT_CLASSID,2);
  PetscValidHeaderSpecific(Y,BV_CLASSID,3);
  PetscValidHeaderSpecific(M,MAT_CLASSID,4);
  PetscValidType(X,1);
  BVCheckSizes(X,1);
  if (A) {
    PetscValidType(A,2);
    PetscCheckSameComm(X,1,A,2);
  }
  PetscValidType(Y,3);
  BVCheckSizes(Y,3);
  PetscValidType(M,4);
  PetscCheckSameTypeAndComm(X,1,Y,3);
  ierr = PetscObjectTypeCompare((PetscObject)M,MATSEQDENSE,&match);CHKERRQ(ierr);
  if (!match) SETERRQ(PetscObjectComm((PetscObject)X),PETSC_ERR_SUP,"Matrix M must be of type seqdense");

  ierr = MatGetSize(M,&m,&n);CHKERRQ(ierr);
  if (m!=Y->k) SETERRQ2(PetscObjectComm((PetscObject)X),PETSC_ERR_ARG_SIZ,"Matrix M has %D rows, should have %D",m,Y->k);
  if (n!=X->k) SETERRQ2(PetscObjectComm((PetscObject)X),PETSC_ERR_ARG_SIZ,"Matrix M has %D columns, should have %D",n,X->k);
  if (X->n!=Y->n) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_ARG_INCOMP,"Mismatching local dimension X %D, Y %D",X->n,Y->n);

  ierr = PetscLogEventBegin(BV_MatProject,X,A,Y,0);CHKERRQ(ierr);
  matrix = X->matrix;
  X->matrix = NULL;  /* temporarily set standard inner product */

  ierr = PetscObjectGetId((PetscObject)X,&idx);CHKERRQ(ierr);
  ierr = PetscObjectGetId((PetscObject)Y,&idy);CHKERRQ(ierr);
  if (!A && idx==idy) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONG,"Cannot set X=Y if A=NULL");

  ierr = MatDenseGetArray(M,&marray);CHKERRQ(ierr);
  lx = X->l; kx = X->k;
  ly = Y->l; ky = Y->k;

  if (A && idx==idy) { /* check symmetry of M=X'AX */
    ierr = MatIsHermitianKnown(A,&set,&flg);CHKERRQ(ierr);
    symm = set? flg: PETSC_FALSE;
  }

  if (A) {  /* perform computation column by column */

    ierr = BVGetVec(X,&f);CHKERRQ(ierr);
    for (j=lx;j<kx;j++) {
      ierr = BVGetColumn(X,j,&z);CHKERRQ(ierr);
      ierr = MatMult(A,z,f);CHKERRQ(ierr);
      ierr = BVRestoreColumn(X,j,&z);CHKERRQ(ierr);
      ulim = PetscMin(ly+(j-lx)+1,ky);
      Y->l = 0; Y->k = ulim;
      ierr = (*Y->ops->dotvec)(Y,f,marray+j*m);CHKERRQ(ierr);
      if (symm) {
        for (i=0;i<j;i++) marray[j+i*m] = PetscConj(marray[i+j*m]);
      }
    }
    if (!symm) {
      ierr = BV_AllocateCoeffs(Y);CHKERRQ(ierr);
      for (j=ly;j<ky;j++) {
        ierr = BVGetColumn(Y,j,&z);CHKERRQ(ierr);
        ierr = MatMultHermitianTranspose(A,z,f);CHKERRQ(ierr);
        ierr = BVRestoreColumn(Y,j,&z);CHKERRQ(ierr);
        ulim = PetscMin(lx+(j-ly),kx);
        X->l = 0; X->k = ulim;
        ierr = (*X->ops->dotvec)(X,f,Y->h);CHKERRQ(ierr);
        for (i=0;i<ulim;i++) marray[j+i*m] = PetscConj(Y->h[i]);
      }
    }
    ierr = VecDestroy(&f);CHKERRQ(ierr);

  } else {  /* use BVDot on subblocks   AX = [ AX0 AX1 ], Y = [ Y0 Y1 ]

                M = [    M0   | Y0'*AX1 ]
                    [ Y1'*AX0 | Y1'*AX1 ]
    */

    /* upper part, Y0'*AX1 */
    ierr = MatCreateSeqDense(PETSC_COMM_SELF,ly,kx,NULL,&H);CHKERRQ(ierr);
    X->l = lx; X->k = kx;
    Y->l = 0;  Y->k = ly;
    ierr = BVDot(X,Y,H);CHKERRQ(ierr);
    ierr = MatDenseGetArray(H,&harray);CHKERRQ(ierr);
    for (j=lx;j<kx;j++) {
      ierr = PetscMemcpy(marray+m*j,harray+j*ly,ly*sizeof(PetscScalar));CHKERRQ(ierr);
    }
    ierr = MatDenseRestoreArray(H,&harray);CHKERRQ(ierr);
    ierr = MatDestroy(&H);CHKERRQ(ierr);

    /* lower part, Y1'*AX */
    ierr = MatCreateSeqDense(PETSC_COMM_SELF,ky,kx,NULL,&H);CHKERRQ(ierr);
    X->l = 0;  X->k = kx;
    Y->l = ly; Y->k = ky;
    ierr = BVDot(X,Y,H);CHKERRQ(ierr);
    ierr = MatDenseGetArray(H,&harray);CHKERRQ(ierr);
    for (j=0;j<kx;j++) {
      ierr = PetscMemcpy(marray+m*j+ly,harray+j*ky+ly,(ky-ly)*sizeof(PetscScalar));CHKERRQ(ierr);
    }
    ierr = MatDenseRestoreArray(H,&harray);CHKERRQ(ierr);
    ierr = MatDestroy(&H);CHKERRQ(ierr);
  }

  X->l = lx; X->k = kx;
  Y->l = ly; Y->k = ky;
  ierr = MatDenseRestoreArray(M,&marray);CHKERRQ(ierr);
  ierr = PetscLogEventEnd(BV_MatProject,X,A,Y,0);CHKERRQ(ierr);
  X->matrix = matrix;  /* restore non-standard inner product */
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "BVAXPY"
/*@
   BVAXPY - Computes Y = Y + alpha*X.

   Logically Collective on BV

   Input Parameters:
+  Y,X   - basis vectors
-  alpha - scalar

   Output Parameter:
.  Y     - the modified basis vectors

   Notes:
   X and Y must be different objects, with compatible dimensions.
   The effect is the same as doing a VecAXPY for each of the active
   columns (excluding the leading ones).

   Level: intermediate

.seealso: BVMult(), BVSetActiveColumns()
@*/
PetscErrorCode BVAXPY(BV Y,PetscScalar alpha,BV X)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(Y,BV_CLASSID,1);
  PetscValidLogicalCollectiveScalar(Y,alpha,2);
  PetscValidHeaderSpecific(X,BV_CLASSID,3);
  PetscValidType(Y,1);
  BVCheckSizes(Y,1);
  PetscValidType(X,3);
  BVCheckSizes(X,3);
  PetscCheckSameTypeAndComm(Y,1,X,3);
  if (X==Y) SETERRQ(PetscObjectComm((PetscObject)Y),PETSC_ERR_ARG_WRONG,"X and Y arguments must be different");
  if (X->n!=Y->n) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_ARG_INCOMP,"Mismatching local dimension X %D, Y %D",X->n,Y->n);
  if (X->k-X->l!=Y->k-Y->l) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_ARG_SIZ,"Y has %D non-leading columns, while X has %D",Y->m-Y->l,X->k-X->l);
  if (!X->n) PetscFunctionReturn(0);

  ierr = PetscLogEventBegin(BV_AXPY,X,Y,0,0);CHKERRQ(ierr);
  ierr = (*Y->ops->axpy)(Y,alpha,X);CHKERRQ(ierr);
  ierr = PetscLogEventEnd(BV_AXPY,X,Y,0,0);CHKERRQ(ierr);
  ierr = PetscObjectStateIncrease((PetscObject)Y);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

