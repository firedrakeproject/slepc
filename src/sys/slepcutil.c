/*
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

#include <slepc/private/slepcimpl.h>            /*I "slepcsys.h" I*/

#undef __FUNCT__
#define __FUNCT__ "SlepcVecNormalize"
/*@C
   SlepcVecNormalize - Normalizes a possibly complex vector by the 2-norm.

   Collective on Vec

   Input parameters:
+  xr - the real part of the vector (overwritten on output)
.  xi - the imaginary part of the vector (not referenced if iscomplex is false)
-  iscomplex - a flag indicating if the vector is complex

   Output parameter:
.  norm - the vector norm before normalization (can be set to NULL)

   Level: developer
@*/
PetscErrorCode SlepcVecNormalize(Vec xr,Vec xi,PetscBool iscomplex,PetscReal *norm)
{
  PetscErrorCode ierr;
#if !defined(PETSC_USE_COMPLEX)
  PetscReal      normr,normi,alpha;
#endif

  PetscFunctionBegin;
  PetscValidHeaderSpecific(xr,VEC_CLASSID,1);
#if !defined(PETSC_USE_COMPLEX)
  if (iscomplex) {
    PetscValidHeaderSpecific(xi,VEC_CLASSID,2);
    ierr = VecNormBegin(xr,NORM_2,&normr);CHKERRQ(ierr);
    ierr = VecNormBegin(xi,NORM_2,&normi);CHKERRQ(ierr);
    ierr = VecNormEnd(xr,NORM_2,&normr);CHKERRQ(ierr);
    ierr = VecNormEnd(xi,NORM_2,&normi);CHKERRQ(ierr);
    alpha = SlepcAbsEigenvalue(normr,normi);
    if (norm) *norm = alpha;
    alpha = 1.0 / alpha;
    ierr = VecScale(xr,alpha);CHKERRQ(ierr);
    ierr = VecScale(xi,alpha);CHKERRQ(ierr);
  } else
#endif
  {
    ierr = VecNormalize(xr,norm);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "SlepcMatConvertSeqDense"
/*@C
   SlepcMatConvertSeqDense - Converts a parallel matrix to another one in sequential
   dense format replicating the values in every processor.

   Collective on Mat

   Input parameters:
+  A  - the source matrix
-  B  - the target matrix

   Level: developer
@*/
PetscErrorCode SlepcMatConvertSeqDense(Mat mat,Mat *newmat)
{
  PetscErrorCode ierr;
  PetscInt       m,n;
  PetscMPIInt    size;
  PetscBool      flg;
  Mat            *M;
  IS             isrow,iscol;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(mat,MAT_CLASSID,1);
  PetscValidPointer(newmat,2);
  ierr = MPI_Comm_size(PetscObjectComm((PetscObject)mat),&size);CHKERRQ(ierr);
  if (size > 1) {
    ierr = MatHasOperation(mat,MATOP_GET_SUBMATRICES,&flg);CHKERRQ(ierr);
    if (!flg) SETERRQ1(PetscObjectComm((PetscObject)mat),PETSC_ERR_SUP,"Mat type %s",((PetscObject)mat)->type_name);

    /* assemble full matrix on every processor */
    ierr = MatGetSize(mat,&m,&n);CHKERRQ(ierr);
    ierr = ISCreateStride(PETSC_COMM_SELF,m,0,1,&isrow);CHKERRQ(ierr);
    ierr = ISCreateStride(PETSC_COMM_SELF,n,0,1,&iscol);CHKERRQ(ierr);
    ierr = MatGetSubMatrices(mat,1,&isrow,&iscol,MAT_INITIAL_MATRIX,&M);CHKERRQ(ierr);
    ierr = ISDestroy(&isrow);CHKERRQ(ierr);
    ierr = ISDestroy(&iscol);CHKERRQ(ierr);

    /* Fake support for "inplace" convert */
    if (*newmat == mat) {
      ierr = MatDestroy(&mat);CHKERRQ(ierr);
    }

    /* convert matrix to MatSeqDense */
    ierr = MatConvert(*M,MATSEQDENSE,MAT_INITIAL_MATRIX,newmat);CHKERRQ(ierr);
    ierr = MatDestroyMatrices(1,&M);CHKERRQ(ierr);
  } else {
    /* convert matrix to MatSeqDense */
    ierr = MatConvert(mat,MATSEQDENSE,MAT_INITIAL_MATRIX,newmat);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "SlepcMatTile_SeqAIJ"
static PetscErrorCode SlepcMatTile_SeqAIJ(PetscScalar a,Mat A,PetscScalar b,Mat B,PetscScalar c,Mat C,PetscScalar d,Mat D,Mat G)
{
  PetscErrorCode    ierr;
  PetscInt          i,j,M1,M2,N1,N2,*nnz,ncols,*scols;
  PetscScalar       *svals,*buf;
  const PetscInt    *cols;
  const PetscScalar *vals;

  PetscFunctionBegin;
  ierr = MatGetSize(A,&M1,&N1);CHKERRQ(ierr);
  ierr = MatGetSize(D,&M2,&N2);CHKERRQ(ierr);

  ierr = PetscCalloc1(M1+M2,&nnz);CHKERRQ(ierr);
  /* Preallocate for A */
  if (a!=0.0) {
    for (i=0;i<M1;i++) {
      ierr = MatGetRow(A,i,&ncols,NULL,NULL);CHKERRQ(ierr);
      nnz[i] += ncols;
      ierr = MatRestoreRow(A,i,&ncols,NULL,NULL);CHKERRQ(ierr);
    }
  }
  /* Preallocate for B */
  if (b!=0.0) {
    for (i=0;i<M1;i++) {
      ierr = MatGetRow(B,i,&ncols,NULL,NULL);CHKERRQ(ierr);
      nnz[i] += ncols;
      ierr = MatRestoreRow(B,i,&ncols,NULL,NULL);CHKERRQ(ierr);
    }
  }
  /* Preallocate for C */
  if (c!=0.0) {
    for (i=0;i<M2;i++) {
      ierr = MatGetRow(C,i,&ncols,NULL,NULL);CHKERRQ(ierr);
      nnz[i+M1] += ncols;
      ierr = MatRestoreRow(C,i,&ncols,NULL,NULL);CHKERRQ(ierr);
    }
  }
  /* Preallocate for D */
  if (d!=0.0) {
    for (i=0;i<M2;i++) {
      ierr = MatGetRow(D,i,&ncols,NULL,NULL);CHKERRQ(ierr);
      nnz[i+M1] += ncols;
      ierr = MatRestoreRow(D,i,&ncols,NULL,NULL);CHKERRQ(ierr);
    }
  }
  ierr = MatSeqAIJSetPreallocation(G,0,nnz);CHKERRQ(ierr);
  ierr = PetscFree(nnz);CHKERRQ(ierr);

  ierr = PetscMalloc2(PetscMax(N1,N2),&buf,PetscMax(N1,N2),&scols);CHKERRQ(ierr);
  /* Transfer A */
  if (a!=0.0) {
    for (i=0;i<M1;i++) {
      ierr = MatGetRow(A,i,&ncols,&cols,&vals);CHKERRQ(ierr);
      if (a!=1.0) {
        svals=buf;
        for (j=0;j<ncols;j++) svals[j] = vals[j]*a;
      } else svals=(PetscScalar*)vals;
      ierr = MatSetValues(G,1,&i,ncols,cols,svals,INSERT_VALUES);CHKERRQ(ierr);
      ierr = MatRestoreRow(A,i,&ncols,&cols,&vals);CHKERRQ(ierr);
    }
  }
  /* Transfer B */
  if (b!=0.0) {
    for (i=0;i<M1;i++) {
      ierr = MatGetRow(B,i,&ncols,&cols,&vals);CHKERRQ(ierr);
      if (b!=1.0) {
        svals=buf;
        for (j=0;j<ncols;j++) svals[j] = vals[j]*b;
      } else svals=(PetscScalar*)vals;
      for (j=0;j<ncols;j++) scols[j] = cols[j]+N1;
      ierr = MatSetValues(G,1,&i,ncols,scols,svals,INSERT_VALUES);CHKERRQ(ierr);
      ierr = MatRestoreRow(B,i,&ncols,&cols,&vals);CHKERRQ(ierr);
    }
  }
  /* Transfer C */
  if (c!=0.0) {
    for (i=0;i<M2;i++) {
      ierr = MatGetRow(C,i,&ncols,&cols,&vals);CHKERRQ(ierr);
      if (c!=1.0) {
        svals=buf;
        for (j=0;j<ncols;j++) svals[j] = vals[j]*c;
      } else svals=(PetscScalar*)vals;
      j = i+M1;
      ierr = MatSetValues(G,1,&j,ncols,cols,svals,INSERT_VALUES);CHKERRQ(ierr);
      ierr = MatRestoreRow(C,i,&ncols,&cols,&vals);CHKERRQ(ierr);
    }
  }
  /* Transfer D */
  if (d!=0.0) {
    for (i=0;i<M2;i++) {
      ierr = MatGetRow(D,i,&ncols,&cols,&vals);CHKERRQ(ierr);
      if (d!=1.0) {
        svals=buf;
        for (j=0;j<ncols;j++) svals[j] = vals[j]*d;
      } else svals=(PetscScalar*)vals;
      for (j=0;j<ncols;j++) scols[j] = cols[j]+N1;
      j = i+M1;
      ierr = MatSetValues(G,1,&j,ncols,scols,svals,INSERT_VALUES);CHKERRQ(ierr);
      ierr = MatRestoreRow(D,i,&ncols,&cols,&vals);CHKERRQ(ierr);
    }
  }
  ierr = PetscFree2(buf,scols);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "SlepcMatTile_MPIAIJ"
static PetscErrorCode SlepcMatTile_MPIAIJ(PetscScalar a,Mat A,PetscScalar b,Mat B,PetscScalar c,Mat C,PetscScalar d,Mat D,Mat G)
{
  PetscErrorCode ierr;
  PetscMPIInt       np;
  PetscInt          p,i,j,N1,N2,m1,m2,n1,n2,*map1,*map2;
  PetscInt          *dnz,*onz,ncols,*scols,start,gstart;
  PetscScalar       *svals,*buf;
  const PetscInt    *cols,*mapptr1,*mapptr2;
  const PetscScalar *vals;

  PetscFunctionBegin;
  ierr = MatGetSize(A,NULL,&N1);CHKERRQ(ierr);
  ierr = MatGetLocalSize(A,&m1,&n1);CHKERRQ(ierr);
  ierr = MatGetSize(D,NULL,&N2);CHKERRQ(ierr);
  ierr = MatGetLocalSize(D,&m2,&n2);CHKERRQ(ierr);

  /* Create mappings */
  ierr = MPI_Comm_size(PetscObjectComm((PetscObject)G),&np);CHKERRQ(ierr);
  ierr = MatGetOwnershipRangesColumn(A,&mapptr1);CHKERRQ(ierr);
  ierr = MatGetOwnershipRangesColumn(B,&mapptr2);CHKERRQ(ierr);
  ierr = PetscMalloc4(PetscMax(N1,N2),&buf,PetscMax(N1,N2),&scols,N1,&map1,N2,&map2);CHKERRQ(ierr);
  for (p=0;p<np;p++) {
    for (i=mapptr1[p];i<mapptr1[p+1];i++) map1[i] = i+mapptr2[p];
  }
  for (p=0;p<np;p++) {
    for (i=mapptr2[p];i<mapptr2[p+1];i++) map2[i] = i+mapptr1[p+1];
  }

  ierr = MatPreallocateInitialize(PetscObjectComm((PetscObject)G),m1+m2,n1+n2,dnz,onz);CHKERRQ(ierr);
  ierr = MatGetOwnershipRange(G,&gstart,NULL);CHKERRQ(ierr);
  /* Preallocate for A */
  if (a!=0.0) {
    ierr = MatGetOwnershipRange(A,&start,NULL);CHKERRQ(ierr);
    for (i=0;i<m1;i++) {
      ierr = MatGetRow(A,i+start,&ncols,&cols,NULL);CHKERRQ(ierr);
      for (j=0;j<ncols;j++) scols[j] = map1[cols[j]];
      ierr = MatPreallocateSet(gstart+i,ncols,scols,dnz,onz);CHKERRQ(ierr);
      ierr = MatRestoreRow(A,i+start,&ncols,&cols,NULL);CHKERRQ(ierr);
    }
  }
  /* Preallocate for B */
  if (b!=0.0) {
    ierr = MatGetOwnershipRange(B,&start,NULL);CHKERRQ(ierr);
    for (i=0;i<m1;i++) {
      ierr = MatGetRow(B,i+start,&ncols,&cols,NULL);CHKERRQ(ierr);
      for (j=0;j<ncols;j++) scols[j] = map2[cols[j]];
      ierr = MatPreallocateSet(gstart+i,ncols,scols,dnz,onz);CHKERRQ(ierr);
      ierr = MatRestoreRow(B,i+start,&ncols,&cols,NULL);CHKERRQ(ierr);
    }
  }
  /* Preallocate for C */
  if (c!=0.0) {
    ierr = MatGetOwnershipRange(C,&start,NULL);CHKERRQ(ierr);
    for (i=0;i<m2;i++) {
      ierr = MatGetRow(C,i+start,&ncols,&cols,NULL);CHKERRQ(ierr);
      for (j=0;j<ncols;j++) scols[j] = map1[cols[j]];
      ierr = MatPreallocateSet(gstart+m1+i,ncols,scols,dnz,onz);CHKERRQ(ierr);
      ierr = MatRestoreRow(C,i+start,&ncols,&cols,NULL);CHKERRQ(ierr);
    }
  }
  /* Preallocate for D */
  if (d!=0.0) {
    ierr = MatGetOwnershipRange(D,&start,NULL);CHKERRQ(ierr);
    for (i=0;i<m2;i++) {
      ierr = MatGetRow(D,i+start,&ncols,&cols,NULL);CHKERRQ(ierr);
      for (j=0;j<ncols;j++) scols[j] = map2[cols[j]];
      ierr = MatPreallocateSet(gstart+m1+i,ncols,scols,dnz,onz);CHKERRQ(ierr);
      ierr = MatRestoreRow(D,i+start,&ncols,&cols,NULL);CHKERRQ(ierr);
    }
  }
  ierr = MatMPIAIJSetPreallocation(G,0,dnz,0,onz);CHKERRQ(ierr);
  ierr = MatPreallocateFinalize(dnz,onz);CHKERRQ(ierr);

  /* Transfer A */
  if (a!=0.0) {
    ierr = MatGetOwnershipRange(A,&start,NULL);CHKERRQ(ierr);
    for (i=0;i<m1;i++) {
      ierr = MatGetRow(A,i+start,&ncols,&cols,&vals);CHKERRQ(ierr);
      if (a!=1.0) {
        svals=buf;
        for (j=0;j<ncols;j++) svals[j] = vals[j]*a;
      } else svals=(PetscScalar*)vals;
      for (j=0;j<ncols;j++) scols[j] = map1[cols[j]];
      j = gstart+i;
      ierr = MatSetValues(G,1,&j,ncols,scols,svals,INSERT_VALUES);CHKERRQ(ierr);
      ierr = MatRestoreRow(A,i+start,&ncols,&cols,&vals);CHKERRQ(ierr);
    }
  }
  /* Transfer B */
  if (b!=0.0) {
    ierr = MatGetOwnershipRange(B,&start,NULL);CHKERRQ(ierr);
    for (i=0;i<m1;i++) {
      ierr = MatGetRow(B,i+start,&ncols,&cols,&vals);CHKERRQ(ierr);
      if (b!=1.0) {
        svals=buf;
        for (j=0;j<ncols;j++) svals[j] = vals[j]*b;
      } else svals=(PetscScalar*)vals;
      for (j=0;j<ncols;j++) scols[j] = map2[cols[j]];
      j = gstart+i;
      ierr = MatSetValues(G,1,&j,ncols,scols,svals,INSERT_VALUES);CHKERRQ(ierr);
      ierr = MatRestoreRow(B,i+start,&ncols,&cols,&vals);CHKERRQ(ierr);
    }
  }
  /* Transfer C */
  if (c!=0.0) {
    ierr = MatGetOwnershipRange(C,&start,NULL);CHKERRQ(ierr);
    for (i=0;i<m2;i++) {
      ierr = MatGetRow(C,i+start,&ncols,&cols,&vals);CHKERRQ(ierr);
      if (c!=1.0) {
        svals=buf;
        for (j=0;j<ncols;j++) svals[j] = vals[j]*c;
      } else svals=(PetscScalar*)vals;
      for (j=0;j<ncols;j++) scols[j] = map1[cols[j]];
      j = gstart+m1+i;
      ierr = MatSetValues(G,1,&j,ncols,scols,svals,INSERT_VALUES);CHKERRQ(ierr);
      ierr = MatRestoreRow(C,i+start,&ncols,&cols,&vals);CHKERRQ(ierr);
    }
  }
  /* Transfer D */
  if (d!=0.0) {
    ierr = MatGetOwnershipRange(D,&start,NULL);CHKERRQ(ierr);
    for (i=0;i<m2;i++) {
      ierr = MatGetRow(D,i+start,&ncols,&cols,&vals);CHKERRQ(ierr);
      if (d!=1.0) {
        svals=buf;
        for (j=0;j<ncols;j++) svals[j] = vals[j]*d;
      } else svals=(PetscScalar*)vals;
      for (j=0;j<ncols;j++) scols[j] = map2[cols[j]];
      j = gstart+m1+i;
      ierr = MatSetValues(G,1,&j,ncols,scols,svals,INSERT_VALUES);CHKERRQ(ierr);
      ierr = MatRestoreRow(D,i+start,&ncols,&cols,&vals);CHKERRQ(ierr);
    }
  }
  ierr = PetscFree4(buf,scols,map1,map2);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "SlepcMatTile"
/*@
   SlepcMatTile - Explicitly build a matrix from four blocks, G = [ a*A b*B; c*C d*D ].

   Collective on Mat

   Input parameters:
+  A - matrix for top-left block
.  a - scaling factor for block A
.  B - matrix for top-right block
.  b - scaling factor for block B
.  C - matrix for bottom-left block
.  c - scaling factor for block C
.  D - matrix for bottom-right block
-  d - scaling factor for block D

   Output parameter:
.  G  - the resulting matrix

   Notes:
   In the case of a parallel matrix, a permuted version of G is returned. The permutation
   is a perfect shuffle such that the local parts of A, B, C, D remain in the local part of
   G for the same process.

   Matrix G must be destroyed by the user.

   Level: developer
@*/
PetscErrorCode SlepcMatTile(PetscScalar a,Mat A,PetscScalar b,Mat B,PetscScalar c,Mat C,PetscScalar d,Mat D,Mat *G)
{
  PetscErrorCode ierr;
  PetscInt       M1,M2,N1,N2,M,N,m1,m2,n1,n2,m,n;
  PetscBool      flg1,flg2;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(A,MAT_CLASSID,2);
  PetscValidHeaderSpecific(B,MAT_CLASSID,4);
  PetscValidHeaderSpecific(C,MAT_CLASSID,6);
  PetscValidHeaderSpecific(D,MAT_CLASSID,8);
  PetscCheckSameTypeAndComm(A,2,B,4);
  PetscCheckSameTypeAndComm(A,2,C,6);
  PetscCheckSameTypeAndComm(A,2,D,8);
  PetscValidPointer(G,9);

  /* check row 1 */
  ierr = MatGetSize(A,&M1,NULL);CHKERRQ(ierr);
  ierr = MatGetLocalSize(A,&m1,NULL);CHKERRQ(ierr);
  ierr = MatGetSize(B,&M,NULL);CHKERRQ(ierr);
  ierr = MatGetLocalSize(B,&m,NULL);CHKERRQ(ierr);
  if (M!=M1 || m!=m1) SETERRQ(PetscObjectComm((PetscObject)A),PETSC_ERR_ARG_INCOMP,"Incompatible dimensions");
  /* check row 2 */
  ierr = MatGetSize(C,&M2,NULL);CHKERRQ(ierr);
  ierr = MatGetLocalSize(C,&m2,NULL);CHKERRQ(ierr);
  ierr = MatGetSize(D,&M,NULL);CHKERRQ(ierr);
  ierr = MatGetLocalSize(D,&m,NULL);CHKERRQ(ierr);
  if (M!=M2 || m!=m2) SETERRQ(PetscObjectComm((PetscObject)A),PETSC_ERR_ARG_INCOMP,"Incompatible dimensions");
  /* check column 1 */
  ierr = MatGetSize(A,NULL,&N1);CHKERRQ(ierr);
  ierr = MatGetLocalSize(A,NULL,&n1);CHKERRQ(ierr);
  ierr = MatGetSize(C,NULL,&N);CHKERRQ(ierr);
  ierr = MatGetLocalSize(C,NULL,&n);CHKERRQ(ierr);
  if (N!=N1 || n!=n1) SETERRQ(PetscObjectComm((PetscObject)A),PETSC_ERR_ARG_INCOMP,"Incompatible dimensions");
  /* check column 2 */
  ierr = MatGetSize(B,NULL,&N2);CHKERRQ(ierr);
  ierr = MatGetLocalSize(B,NULL,&n2);CHKERRQ(ierr);
  ierr = MatGetSize(D,NULL,&N);CHKERRQ(ierr);
  ierr = MatGetLocalSize(D,NULL,&n);CHKERRQ(ierr);
  if (N!=N2 || n!=n2) SETERRQ(PetscObjectComm((PetscObject)A),PETSC_ERR_ARG_INCOMP,"Incompatible dimensions");

  ierr = MatCreate(PetscObjectComm((PetscObject)A),G);CHKERRQ(ierr);
  ierr = MatSetSizes(*G,m1+m2,n1+n2,M1+M2,N1+N2);CHKERRQ(ierr);
  ierr = MatSetFromOptions(*G);CHKERRQ(ierr);
  ierr = MatSetUp(*G);CHKERRQ(ierr);

  ierr = PetscObjectTypeCompare((PetscObject)*G,MATMPIAIJ,&flg1);CHKERRQ(ierr);
  ierr = PetscObjectTypeCompare((PetscObject)A,MATMPIAIJ,&flg2);CHKERRQ(ierr);
  if (flg1 && flg2) {
    ierr = SlepcMatTile_MPIAIJ(a,A,b,B,c,C,d,D,*G);CHKERRQ(ierr);
  } else {
    ierr = PetscObjectTypeCompare((PetscObject)*G,MATSEQAIJ,&flg1);CHKERRQ(ierr);
    ierr = PetscObjectTypeCompare((PetscObject)A,MATSEQAIJ,&flg2);CHKERRQ(ierr);
    if (flg1 && flg2) {
      ierr = SlepcMatTile_SeqAIJ(a,A,b,B,c,C,d,D,*G);CHKERRQ(ierr);
    } else SETERRQ(PetscObjectComm((PetscObject)A),PETSC_ERR_SUP,"Not implemented for this matrix type");
  }
  ierr = MatAssemblyBegin(*G,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(*G,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "SlepcCheckOrthogonality"
/*@C
   SlepcCheckOrthogonality - Checks (or prints) the level of orthogonality
   of a set of vectors.

   Collective on Vec

   Input parameters:
+  V  - a set of vectors
.  nv - number of V vectors
.  W  - an alternative set of vectors (optional)
.  nw - number of W vectors
.  B  - matrix defining the inner product (optional)
-  viewer - optional visualization context

   Output parameter:
.  lev - level of orthogonality (optional)

   Notes:
   This function computes W'*V and prints the result. It is intended to check
   the level of bi-orthogonality of the vectors in the two sets. If W is equal
   to NULL then V is used, thus checking the orthogonality of the V vectors.

   If matrix B is provided then the check uses the B-inner product, W'*B*V.

   If lev is not NULL, it will contain the maximum entry of matrix
   W'*V - I (in absolute value). Otherwise, the matrix W'*V is printed.

   Level: developer
@*/
PetscErrorCode SlepcCheckOrthogonality(Vec *V,PetscInt nv,Vec *W,PetscInt nw,Mat B,PetscViewer viewer,PetscReal *lev)
{
  PetscErrorCode ierr;
  PetscInt       i,j;
  PetscScalar    *vals;
  PetscBool      isascii;
  Vec            w;

  PetscFunctionBegin;
  if (nv<=0 || nw<=0) PetscFunctionReturn(0);
  PetscValidPointer(V,1);
  PetscValidHeaderSpecific(*V,VEC_CLASSID,1);
  if (!lev) {
    if (!viewer) viewer = PETSC_VIEWER_STDOUT_(PetscObjectComm((PetscObject)*V));
    PetscValidHeaderSpecific(viewer,PETSC_VIEWER_CLASSID,2);
    PetscCheckSameComm(*V,1,viewer,6);
    ierr = PetscObjectTypeCompare((PetscObject)viewer,PETSCVIEWERASCII,&isascii);CHKERRQ(ierr);
    if (!isascii) PetscFunctionReturn(0);
  }

  ierr = PetscMalloc1(nv,&vals);CHKERRQ(ierr);
  if (B) {
    ierr = VecDuplicate(V[0],&w);CHKERRQ(ierr);
  }
  if (lev) *lev = 0.0;
  for (i=0;i<nw;i++) {
    if (B) {
      if (W) {
        ierr = MatMultTranspose(B,W[i],w);CHKERRQ(ierr);
      } else {
        ierr = MatMultTranspose(B,V[i],w);CHKERRQ(ierr);
      }
    } else {
      if (W) w = W[i];
      else w = V[i];
    }
    ierr = VecMDot(w,nv,V,vals);CHKERRQ(ierr);
    for (j=0;j<nv;j++) {
      if (lev) *lev = PetscMax(*lev,PetscAbsScalar((j==i)? (vals[j]-1.0): vals[j]));
      else {
#if !defined(PETSC_USE_COMPLEX)
        ierr = PetscViewerASCIIPrintf(viewer," %12g  ",(double)vals[j]);CHKERRQ(ierr);
#else
        ierr = PetscViewerASCIIPrintf(viewer," %12g%+12gi ",(double)PetscRealPart(vals[j]),(double)PetscImaginaryPart(vals[j]));CHKERRQ(ierr);
#endif
      }
    }
    if (!lev) { ierr = PetscViewerASCIIPrintf(viewer,"\n");CHKERRQ(ierr); }
  }
  ierr = PetscFree(vals);CHKERRQ(ierr);
  if (B) {
    ierr = VecDestroy(&w);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "SlepcConvMonitorCreate"
/*@C
   SlepcConvMonitorCreate - Creates a SlepcConvMonitor context.

   Collective on PetscViewer

   Input Parameters:
+  viewer - the viewer where the monitor must send data
-  format - the format 

   Output Parameter:
.  ctx - the created context

   Notes:
   The created context is used for EPS, SVD, PEP, and NEP monitor functions that just
   print the iteration numbers at which convergence takes place (XXXMonitorConverged).

   This function increases the reference count of the viewer so you can destroy the
   viewer object after this call.

   Level: developer

.seealso: SlepcConvMonitorDestroy()
@*/
PetscErrorCode SlepcConvMonitorCreate(PetscViewer viewer,PetscViewerFormat format,SlepcConvMonitor *ctx)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscObjectReference((PetscObject)viewer);CHKERRQ(ierr);
  ierr = PetscNew(ctx);CHKERRQ(ierr);
  (*ctx)->viewer = viewer;
  (*ctx)->format = format;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "SlepcConvMonitorDestroy"
/*@C
   SlepcConvMonitorDestroy - Destroys a SlepcConvMonitor context.

   Collective on PetscViewer

   Input Parameters:
.  ctx - the SlepcConvMonitor context to be destroyed.

   Level: developer

.seealso: SlepcConvMonitorCreate()
@*/
PetscErrorCode SlepcConvMonitorDestroy(SlepcConvMonitor *ctx)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (!*ctx) PetscFunctionReturn(0);
  ierr = PetscViewerDestroy(&(*ctx)->viewer);CHKERRQ(ierr);
  ierr = PetscFree(*ctx);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "SlepcBasisReference_Private"
/*
   Given n vectors in V, this function gets references of them into W.
   If m<0 then some previous non-processed vectors remain in W and must be freed.
 */
PetscErrorCode SlepcBasisReference_Private(PetscInt n,Vec *V,PetscInt *m,Vec **W)
{
  PetscErrorCode ierr;
  PetscInt       i;

  PetscFunctionBegin;
  ierr = SlepcBasisDestroy_Private(m,W);CHKERRQ(ierr);
  if (n>0) {
    ierr = PetscMalloc1(n,W);CHKERRQ(ierr);
    for (i=0;i<n;i++) {
      ierr = PetscObjectReference((PetscObject)V[i]);CHKERRQ(ierr);
      (*W)[i] = V[i];
    }
    *m = -n;
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "SlepcBasisDestroy_Private"
/*
   Destroys a set of vectors.
   A negative value of m indicates that W contains vectors to be destroyed.
 */
PetscErrorCode SlepcBasisDestroy_Private(PetscInt *m,Vec **W)
{
  PetscErrorCode ierr;
  PetscInt       i;

  PetscFunctionBegin;
  if (*m<0) {
    for (i=0;i<-(*m);i++) {
      ierr = VecDestroy(&(*W)[i]);CHKERRQ(ierr);
    }
    ierr = PetscFree(*W);CHKERRQ(ierr);
  }
  *m = 0;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "SlepcSNPrintfScalar"
/*@C
   SlepcSNPrintfScalar - Prints a PetscScalar variable to a string of
   given length.

   Not Collective

   Input Parameters:
+  str - the string to print to
.  len - the length of str
.  val - scalar value to be printed
-  exp - to be used within an expression, print leading sign and parentheses
         in case of nonzero imaginary part

   Level: developer
@*/
PetscErrorCode SlepcSNPrintfScalar(char *str,size_t len,PetscScalar val,PetscBool exp)
{
  PetscErrorCode ierr;
#if defined(PETSC_USE_COMPLEX)
  PetscReal      re,im;
#endif

  PetscFunctionBegin;
#if !defined(PETSC_USE_COMPLEX)
  if (exp) {
    ierr = PetscSNPrintf(str,len,"%+g",(double)val);CHKERRQ(ierr);
  } else {
    ierr = PetscSNPrintf(str,len,"%g",(double)val);CHKERRQ(ierr);
  }
#else
  re = PetscRealPart(val);
  im = PetscImaginaryPart(val);
  if (im!=0.0) {
    if (exp) {
      ierr = PetscSNPrintf(str,len,"+(%g%+gi)",(double)re,(double)im);CHKERRQ(ierr);
    } else {
      ierr = PetscSNPrintf(str,len,"%g%+gi",(double)re,(double)im);CHKERRQ(ierr);
    }
  } else {
    if (exp) {
      ierr = PetscSNPrintf(str,len,"%+g",(double)re);CHKERRQ(ierr);
    } else {
      ierr = PetscSNPrintf(str,len,"%g",(double)re);CHKERRQ(ierr);
    }
  }
#endif
  PetscFunctionReturn(0);
}

