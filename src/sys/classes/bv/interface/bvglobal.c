/*
   BV operations involving global communication.

   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
   SLEPc - Scalable Library for Eigenvalue Problem Computations
   Copyright (c) 2002-2014, Universitat Politecnica de Valencia, Spain

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

   On entry, M must be a sequential dense Mat with dimensions m,n at least, where
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
  if (m<Y->k) SETERRQ2(PetscObjectComm((PetscObject)X),PETSC_ERR_ARG_SIZ,"Mat argument has %D rows, should have at least %D",m,Y->k);
  if (n<X->k) SETERRQ2(PetscObjectComm((PetscObject)X),PETSC_ERR_ARG_SIZ,"Mat argument has %D columns, should have at least %D",n,X->k);
  if (X->n!=Y->n) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_ARG_INCOMP,"Mismatching local dimension X %D, Y %D",X->n,Y->n);
  if (X->matrix!=Y->matrix) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONGSTATE,"X and Y must have the same inner product matrix");
  if (X->l==X->k || Y->l==Y->k) PetscFunctionReturn(0);

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

  ierr = PetscLogEventBegin(BV_Dot,X,0,0,0);CHKERRQ(ierr);
  ksave = X->k;
  X->k = j;
  ierr = BVGetColumn(X,j,&y);CHKERRQ(ierr);
  ierr = (*X->ops->dotvec)(X,y,m);CHKERRQ(ierr);
  ierr = BVRestoreColumn(X,j,&y);CHKERRQ(ierr);
  X->k = ksave;
  ierr = PetscLogEventEnd(BV_Dot,X,0,0,0);CHKERRQ(ierr);
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
   BVNorm - Computes the matrix norm of the BV.

   Collective on BV

   Input Parameters:
+  bv   - basis vectors
-  type - the norm type

   Output Parameter:
.  val  - the norm

   Notes:
   All active columns (except the leading ones) are considered as a matrix.
   The allowed norms are NORM_1, NORM_FROBENIUS, and NORM_INFINITY.

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
.  v    - the vector
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
.  j    - column number to be used
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

   On entry, M must be a sequential dense Mat with dimensions ky,kx at least,
   where ky (resp. kx) is the number of active columns of Y (resp. X).
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
  Mat            Xmatrix,Ymatrix,H;
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
  if (m<Y->k) SETERRQ2(PetscObjectComm((PetscObject)X),PETSC_ERR_ARG_SIZ,"Matrix M has %D rows, should have at least %D",m,Y->k);
  if (n<X->k) SETERRQ2(PetscObjectComm((PetscObject)X),PETSC_ERR_ARG_SIZ,"Matrix M has %D columns, should have at least %D",n,X->k);
  if (X->n!=Y->n) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_ARG_INCOMP,"Mismatching local dimension X %D, Y %D",X->n,Y->n);

  ierr = PetscLogEventBegin(BV_MatProject,X,A,Y,0);CHKERRQ(ierr);
  /* temporarily set standard inner product */
  Xmatrix = X->matrix;
  Ymatrix = Y->matrix;
  X->matrix = Y->matrix = NULL;

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
    if (ly>0 && lx<kx) {
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
    }

    /* lower part, Y1'*AX */
    if (kx>0 && ly<ky) {
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
  }

  X->l = lx; X->k = kx;
  Y->l = ly; Y->k = ky;
  ierr = MatDenseRestoreArray(M,&marray);CHKERRQ(ierr);
  ierr = PetscLogEventEnd(BV_MatProject,X,A,Y,0);CHKERRQ(ierr);
  /* restore non-standard inner product */
  X->matrix = Xmatrix;
  Y->matrix = Ymatrix;
  PetscFunctionReturn(0);
}

