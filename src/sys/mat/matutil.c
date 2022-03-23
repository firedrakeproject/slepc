/*
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
   SLEPc - Scalable Library for Eigenvalue Problem Computations
   Copyright (c) 2002-2021, Universitat Politecnica de Valencia, Spain

   This file is part of SLEPc.
   SLEPc is distributed under a 2-clause BSD license (see LICENSE).
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
*/

#include <slepc/private/slepcimpl.h>            /*I "slepcsys.h" I*/

static PetscErrorCode MatCreateTile_Seq(PetscScalar a,Mat A,PetscScalar b,Mat B,PetscScalar c,Mat C,PetscScalar d,Mat D,Mat G)
{
  PetscInt          i,j,M1,M2,N1,N2,*nnz,ncols,*scols,bs;
  PetscScalar       *svals,*buf;
  const PetscInt    *cols;
  const PetscScalar *vals;

  PetscFunctionBegin;
  CHKERRQ(MatGetSize(A,&M1,&N1));
  CHKERRQ(MatGetSize(D,&M2,&N2));
  CHKERRQ(MatGetBlockSize(A,&bs));

  CHKERRQ(PetscCalloc1((M1+M2)/bs,&nnz));
  /* Preallocate for A */
  if (a!=0.0) {
    for (i=0;i<(M1+bs-1)/bs;i++) {
      CHKERRQ(MatGetRow(A,i*bs,&ncols,NULL,NULL));
      nnz[i] += ncols/bs;
      CHKERRQ(MatRestoreRow(A,i*bs,&ncols,NULL,NULL));
    }
  }
  /* Preallocate for B */
  if (b!=0.0) {
    for (i=0;i<(M1+bs-1)/bs;i++) {
      CHKERRQ(MatGetRow(B,i*bs,&ncols,NULL,NULL));
      nnz[i] += ncols/bs;
      CHKERRQ(MatRestoreRow(B,i*bs,&ncols,NULL,NULL));
    }
  }
  /* Preallocate for C */
  if (c!=0.0) {
    for (i=0;i<(M2+bs-1)/bs;i++) {
      CHKERRQ(MatGetRow(C,i*bs,&ncols,NULL,NULL));
      nnz[i+M1/bs] += ncols/bs;
      CHKERRQ(MatRestoreRow(C,i*bs,&ncols,NULL,NULL));
    }
  }
  /* Preallocate for D */
  if (d!=0.0) {
    for (i=0;i<(M2+bs-1)/bs;i++) {
      CHKERRQ(MatGetRow(D,i*bs,&ncols,NULL,NULL));
      nnz[i+M1/bs] += ncols/bs;
      CHKERRQ(MatRestoreRow(D,i*bs,&ncols,NULL,NULL));
    }
  }
  CHKERRQ(MatXAIJSetPreallocation(G,bs,nnz,NULL,NULL,NULL));
  CHKERRQ(PetscFree(nnz));

  CHKERRQ(PetscMalloc2(PetscMax(N1,N2),&buf,PetscMax(N1,N2),&scols));
  /* Transfer A */
  if (a!=0.0) {
    for (i=0;i<M1;i++) {
      CHKERRQ(MatGetRow(A,i,&ncols,&cols,&vals));
      if (a!=1.0) {
        svals=buf;
        for (j=0;j<ncols;j++) svals[j] = vals[j]*a;
      } else svals=(PetscScalar*)vals;
      CHKERRQ(MatSetValues(G,1,&i,ncols,cols,svals,INSERT_VALUES));
      CHKERRQ(MatRestoreRow(A,i,&ncols,&cols,&vals));
    }
  }
  /* Transfer B */
  if (b!=0.0) {
    for (i=0;i<M1;i++) {
      CHKERRQ(MatGetRow(B,i,&ncols,&cols,&vals));
      if (b!=1.0) {
        svals=buf;
        for (j=0;j<ncols;j++) svals[j] = vals[j]*b;
      } else svals=(PetscScalar*)vals;
      for (j=0;j<ncols;j++) scols[j] = cols[j]+N1;
      CHKERRQ(MatSetValues(G,1,&i,ncols,scols,svals,INSERT_VALUES));
      CHKERRQ(MatRestoreRow(B,i,&ncols,&cols,&vals));
    }
  }
  /* Transfer C */
  if (c!=0.0) {
    for (i=0;i<M2;i++) {
      CHKERRQ(MatGetRow(C,i,&ncols,&cols,&vals));
      if (c!=1.0) {
        svals=buf;
        for (j=0;j<ncols;j++) svals[j] = vals[j]*c;
      } else svals=(PetscScalar*)vals;
      j = i+M1;
      CHKERRQ(MatSetValues(G,1,&j,ncols,cols,svals,INSERT_VALUES));
      CHKERRQ(MatRestoreRow(C,i,&ncols,&cols,&vals));
    }
  }
  /* Transfer D */
  if (d!=0.0) {
    for (i=0;i<M2;i++) {
      CHKERRQ(MatGetRow(D,i,&ncols,&cols,&vals));
      if (d!=1.0) {
        svals=buf;
        for (j=0;j<ncols;j++) svals[j] = vals[j]*d;
      } else svals=(PetscScalar*)vals;
      for (j=0;j<ncols;j++) scols[j] = cols[j]+N1;
      j = i+M1;
      CHKERRQ(MatSetValues(G,1,&j,ncols,scols,svals,INSERT_VALUES));
      CHKERRQ(MatRestoreRow(D,i,&ncols,&cols,&vals));
    }
  }
  CHKERRQ(PetscFree2(buf,scols));
  PetscFunctionReturn(0);
}

static PetscErrorCode MatCreateTile_MPI(PetscScalar a,Mat A,PetscScalar b,Mat B,PetscScalar c,Mat C,PetscScalar d,Mat D,Mat G)
{
  PetscErrorCode    ierr;
  PetscMPIInt       np;
  PetscInt          p,i,j,N1,N2,m1,m2,n1,n2,*map1,*map2;
  PetscInt          *dnz,*onz,ncols,*scols,start,gstart;
  PetscScalar       *svals,*buf;
  const PetscInt    *cols,*mapptr1,*mapptr2;
  const PetscScalar *vals;

  PetscFunctionBegin;
  CHKERRQ(MatGetSize(A,NULL,&N1));
  CHKERRQ(MatGetLocalSize(A,&m1,&n1));
  CHKERRQ(MatGetSize(D,NULL,&N2));
  CHKERRQ(MatGetLocalSize(D,&m2,&n2));

  /* Create mappings */
  CHKERRMPI(MPI_Comm_size(PetscObjectComm((PetscObject)G),&np));
  CHKERRQ(MatGetOwnershipRangesColumn(A,&mapptr1));
  CHKERRQ(MatGetOwnershipRangesColumn(B,&mapptr2));
  CHKERRQ(PetscMalloc4(PetscMax(N1,N2),&buf,PetscMax(N1,N2),&scols,N1,&map1,N2,&map2));
  for (p=0;p<np;p++) {
    for (i=mapptr1[p];i<mapptr1[p+1];i++) map1[i] = i+mapptr2[p];
  }
  for (p=0;p<np;p++) {
    for (i=mapptr2[p];i<mapptr2[p+1];i++) map2[i] = i+mapptr1[p+1];
  }

  ierr = MatPreallocateInitialize(PetscObjectComm((PetscObject)G),m1+m2,n1+n2,dnz,onz);CHKERRQ(ierr);
  CHKERRQ(MatGetOwnershipRange(G,&gstart,NULL));
  /* Preallocate for A */
  if (a!=0.0) {
    CHKERRQ(MatGetOwnershipRange(A,&start,NULL));
    for (i=0;i<m1;i++) {
      CHKERRQ(MatGetRow(A,i+start,&ncols,&cols,NULL));
      for (j=0;j<ncols;j++) scols[j] = map1[cols[j]];
      CHKERRQ(MatPreallocateSet(gstart+i,ncols,scols,dnz,onz));
      CHKERRQ(MatRestoreRow(A,i+start,&ncols,&cols,NULL));
    }
  }
  /* Preallocate for B */
  if (b!=0.0) {
    CHKERRQ(MatGetOwnershipRange(B,&start,NULL));
    for (i=0;i<m1;i++) {
      CHKERRQ(MatGetRow(B,i+start,&ncols,&cols,NULL));
      for (j=0;j<ncols;j++) scols[j] = map2[cols[j]];
      CHKERRQ(MatPreallocateSet(gstart+i,ncols,scols,dnz,onz));
      CHKERRQ(MatRestoreRow(B,i+start,&ncols,&cols,NULL));
    }
  }
  /* Preallocate for C */
  if (c!=0.0) {
    CHKERRQ(MatGetOwnershipRange(C,&start,NULL));
    for (i=0;i<m2;i++) {
      CHKERRQ(MatGetRow(C,i+start,&ncols,&cols,NULL));
      for (j=0;j<ncols;j++) scols[j] = map1[cols[j]];
      CHKERRQ(MatPreallocateSet(gstart+m1+i,ncols,scols,dnz,onz));
      CHKERRQ(MatRestoreRow(C,i+start,&ncols,&cols,NULL));
    }
  }
  /* Preallocate for D */
  if (d!=0.0) {
    CHKERRQ(MatGetOwnershipRange(D,&start,NULL));
    for (i=0;i<m2;i++) {
      CHKERRQ(MatGetRow(D,i+start,&ncols,&cols,NULL));
      for (j=0;j<ncols;j++) scols[j] = map2[cols[j]];
      CHKERRQ(MatPreallocateSet(gstart+m1+i,ncols,scols,dnz,onz));
      CHKERRQ(MatRestoreRow(D,i+start,&ncols,&cols,NULL));
    }
  }
  CHKERRQ(MatMPIAIJSetPreallocation(G,0,dnz,0,onz));
  ierr = MatPreallocateFinalize(dnz,onz);CHKERRQ(ierr);

  /* Transfer A */
  if (a!=0.0) {
    CHKERRQ(MatGetOwnershipRange(A,&start,NULL));
    for (i=0;i<m1;i++) {
      CHKERRQ(MatGetRow(A,i+start,&ncols,&cols,&vals));
      if (a!=1.0) {
        svals=buf;
        for (j=0;j<ncols;j++) svals[j] = vals[j]*a;
      } else svals=(PetscScalar*)vals;
      for (j=0;j<ncols;j++) scols[j] = map1[cols[j]];
      j = gstart+i;
      CHKERRQ(MatSetValues(G,1,&j,ncols,scols,svals,INSERT_VALUES));
      CHKERRQ(MatRestoreRow(A,i+start,&ncols,&cols,&vals));
    }
  }
  /* Transfer B */
  if (b!=0.0) {
    CHKERRQ(MatGetOwnershipRange(B,&start,NULL));
    for (i=0;i<m1;i++) {
      CHKERRQ(MatGetRow(B,i+start,&ncols,&cols,&vals));
      if (b!=1.0) {
        svals=buf;
        for (j=0;j<ncols;j++) svals[j] = vals[j]*b;
      } else svals=(PetscScalar*)vals;
      for (j=0;j<ncols;j++) scols[j] = map2[cols[j]];
      j = gstart+i;
      CHKERRQ(MatSetValues(G,1,&j,ncols,scols,svals,INSERT_VALUES));
      CHKERRQ(MatRestoreRow(B,i+start,&ncols,&cols,&vals));
    }
  }
  /* Transfer C */
  if (c!=0.0) {
    CHKERRQ(MatGetOwnershipRange(C,&start,NULL));
    for (i=0;i<m2;i++) {
      CHKERRQ(MatGetRow(C,i+start,&ncols,&cols,&vals));
      if (c!=1.0) {
        svals=buf;
        for (j=0;j<ncols;j++) svals[j] = vals[j]*c;
      } else svals=(PetscScalar*)vals;
      for (j=0;j<ncols;j++) scols[j] = map1[cols[j]];
      j = gstart+m1+i;
      CHKERRQ(MatSetValues(G,1,&j,ncols,scols,svals,INSERT_VALUES));
      CHKERRQ(MatRestoreRow(C,i+start,&ncols,&cols,&vals));
    }
  }
  /* Transfer D */
  if (d!=0.0) {
    CHKERRQ(MatGetOwnershipRange(D,&start,NULL));
    for (i=0;i<m2;i++) {
      CHKERRQ(MatGetRow(D,i+start,&ncols,&cols,&vals));
      if (d!=1.0) {
        svals=buf;
        for (j=0;j<ncols;j++) svals[j] = vals[j]*d;
      } else svals=(PetscScalar*)vals;
      for (j=0;j<ncols;j++) scols[j] = map2[cols[j]];
      j = gstart+m1+i;
      CHKERRQ(MatSetValues(G,1,&j,ncols,scols,svals,INSERT_VALUES));
      CHKERRQ(MatRestoreRow(D,i+start,&ncols,&cols,&vals));
    }
  }
  CHKERRQ(PetscFree4(buf,scols,map1,map2));
  PetscFunctionReturn(0);
}

/*@
   MatCreateTile - Explicitly build a matrix from four blocks, G = [ a*A b*B; c*C d*D ].

   Collective on A

   Input Parameters:
+  A - matrix for top-left block
.  a - scaling factor for block A
.  B - matrix for top-right block
.  b - scaling factor for block B
.  C - matrix for bottom-left block
.  c - scaling factor for block C
.  D - matrix for bottom-right block
-  d - scaling factor for block D

   Output Parameter:
.  G  - the resulting matrix

   Notes:
   In the case of a parallel matrix, a permuted version of G is returned. The permutation
   is a perfect shuffle such that the local parts of A, B, C, D remain in the local part of
   G for the same process.

   Matrix G must be destroyed by the user.

   The blocks can be of different type. They can be either ConstantDiagonal, or a standard
   type such as AIJ, or any other type provided that it supports the MatGetRow operation.
   The type of the output matrix will be the same as the first block that is not
   ConstantDiagonal (checked in the A,B,C,D order).

   Level: developer

.seealso: MatCreateNest()
@*/
PetscErrorCode MatCreateTile(PetscScalar a,Mat A,PetscScalar b,Mat B,PetscScalar c,Mat C,PetscScalar d,Mat D,Mat *G)
{
  PetscInt       i,k,M1,M2,N1,N2,M,N,m1,m2,n1,n2,m,n,bs;
  PetscBool      diag[4];
  Mat            block[4] = {A,B,C,D};
  MatType        type[4];
  PetscMPIInt    size;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(A,MAT_CLASSID,2);
  PetscValidHeaderSpecific(B,MAT_CLASSID,4);
  PetscValidHeaderSpecific(C,MAT_CLASSID,6);
  PetscValidHeaderSpecific(D,MAT_CLASSID,8);
  PetscCheckSameTypeAndComm(A,2,B,4);
  PetscCheckSameTypeAndComm(A,2,C,6);
  PetscCheckSameTypeAndComm(A,2,D,8);
  PetscValidLogicalCollectiveScalar(A,a,1);
  PetscValidLogicalCollectiveScalar(A,b,3);
  PetscValidLogicalCollectiveScalar(A,c,5);
  PetscValidLogicalCollectiveScalar(A,d,7);
  PetscValidPointer(G,9);

  /* check row 1 */
  CHKERRQ(MatGetSize(A,&M1,NULL));
  CHKERRQ(MatGetLocalSize(A,&m1,NULL));
  CHKERRQ(MatGetSize(B,&M,NULL));
  CHKERRQ(MatGetLocalSize(B,&m,NULL));
  PetscCheck(M==M1 && m==m1,PetscObjectComm((PetscObject)A),PETSC_ERR_ARG_INCOMP,"Incompatible dimensions");
  /* check row 2 */
  CHKERRQ(MatGetSize(C,&M2,NULL));
  CHKERRQ(MatGetLocalSize(C,&m2,NULL));
  CHKERRQ(MatGetSize(D,&M,NULL));
  CHKERRQ(MatGetLocalSize(D,&m,NULL));
  PetscCheck(M==M2 && m==m2,PetscObjectComm((PetscObject)A),PETSC_ERR_ARG_INCOMP,"Incompatible dimensions");
  /* check column 1 */
  CHKERRQ(MatGetSize(A,NULL,&N1));
  CHKERRQ(MatGetLocalSize(A,NULL,&n1));
  CHKERRQ(MatGetSize(C,NULL,&N));
  CHKERRQ(MatGetLocalSize(C,NULL,&n));
  PetscCheck(N==N1 && n==n1,PetscObjectComm((PetscObject)A),PETSC_ERR_ARG_INCOMP,"Incompatible dimensions");
  /* check column 2 */
  CHKERRQ(MatGetSize(B,NULL,&N2));
  CHKERRQ(MatGetLocalSize(B,NULL,&n2));
  CHKERRQ(MatGetSize(D,NULL,&N));
  CHKERRQ(MatGetLocalSize(D,NULL,&n));
  PetscCheck(N==N2 && n==n2,PetscObjectComm((PetscObject)A),PETSC_ERR_ARG_INCOMP,"Incompatible dimensions");

  /* check matrix types */
  for (i=0;i<4;i++) {
    CHKERRQ(MatGetType(block[i],&type[i]));
    CHKERRQ(PetscStrcmp(type[i],MATCONSTANTDIAGONAL,&diag[i]));
  }
  for (k=0;k<4;k++) if (!diag[k]) break;
  PetscCheck(k<4,PetscObjectComm((PetscObject)A),PETSC_ERR_SUP,"Not implemented for 4 diagonal blocks");

  CHKERRQ(MatGetBlockSize(block[k],&bs));
  CHKERRQ(MatCreate(PetscObjectComm((PetscObject)block[k]),G));
  CHKERRQ(MatSetSizes(*G,m1+m2,n1+n2,M1+M2,N1+N2));
  CHKERRQ(MatSetType(*G,type[k]));
  CHKERRQ(MatSetBlockSize(*G,bs));
  CHKERRQ(MatSetUp(*G));

  CHKERRMPI(MPI_Comm_size(PetscObjectComm((PetscObject)*G),&size));
  if (size>1) CHKERRQ(MatCreateTile_MPI(a,A,b,B,c,C,d,D,*G));
  else CHKERRQ(MatCreateTile_Seq(a,A,b,B,c,C,d,D,*G));
  CHKERRQ(MatAssemblyBegin(*G,MAT_FINAL_ASSEMBLY));
  CHKERRQ(MatAssemblyEnd(*G,MAT_FINAL_ASSEMBLY));
  PetscFunctionReturn(0);
}

/*@C
   MatCreateVecsEmpty - Get vector(s) compatible with the matrix, i.e. with the same
   parallel layout, but without internal array.

   Collective on mat

   Input Parameter:
.  mat - the matrix

   Output Parameters:
+  right - (optional) vector that the matrix can be multiplied against
-  left - (optional) vector that the matrix vector product can be stored in

   Note:
   This is similar to MatCreateVecs(), but the new vectors do not have an internal
   array, so the intended usage is with VecPlaceArray().

   Level: developer

.seealso: VecDuplicateEmpty()
@*/
PetscErrorCode MatCreateVecsEmpty(Mat mat,Vec *right,Vec *left)
{
  PetscBool      standard,cuda=PETSC_FALSE,skip=PETSC_FALSE;
  PetscInt       M,N,mloc,nloc,rbs,cbs;
  PetscMPIInt    size;
  Vec            v;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(mat,MAT_CLASSID,1);
  PetscValidType(mat,1);

  CHKERRQ(PetscObjectTypeCompareAny((PetscObject)mat,&standard,MATSEQAIJ,MATMPIAIJ,MATSEQBAIJ,MATMPIBAIJ,MATSEQSBAIJ,MATMPISBAIJ,MATSEQDENSE,MATMPIDENSE,""));
  CHKERRQ(PetscObjectTypeCompareAny((PetscObject)mat,&cuda,MATSEQAIJCUSPARSE,MATMPIAIJCUSPARSE,""));
  if (!standard && !cuda) {
    CHKERRQ(MatCreateVecs(mat,right,left));
    v = right? *right: *left;
    if (v) {
      CHKERRQ(PetscObjectTypeCompareAny((PetscObject)v,&standard,VECSEQ,VECMPI,""));
      CHKERRQ(PetscObjectTypeCompareAny((PetscObject)v,&cuda,VECSEQCUDA,VECMPICUDA,""));
    }
    if (!standard && !cuda) skip = PETSC_TRUE;
    else {
      if (right) CHKERRQ(VecDestroy(right));
      if (left) CHKERRQ(VecDestroy(left));
    }
  }
  if (!skip) {
    CHKERRMPI(MPI_Comm_size(PetscObjectComm((PetscObject)mat),&size));
    CHKERRQ(MatGetLocalSize(mat,&mloc,&nloc));
    CHKERRQ(MatGetSize(mat,&M,&N));
    CHKERRQ(MatGetBlockSizes(mat,&rbs,&cbs));
    if (right) {
      if (cuda) {
#if defined(PETSC_HAVE_CUDA)
        if (size>1) CHKERRQ(VecCreateMPICUDAWithArray(PetscObjectComm((PetscObject)mat),cbs,nloc,N,NULL,right));
        else CHKERRQ(VecCreateSeqCUDAWithArray(PetscObjectComm((PetscObject)mat),cbs,N,NULL,right));
#endif
      } else {
        if (size>1) CHKERRQ(VecCreateMPIWithArray(PetscObjectComm((PetscObject)mat),cbs,nloc,N,NULL,right));
        else CHKERRQ(VecCreateSeqWithArray(PetscObjectComm((PetscObject)mat),cbs,N,NULL,right));
      }
    }
    if (left) {
      if (cuda) {
#if defined(PETSC_HAVE_CUDA)
        if (size>1) CHKERRQ(VecCreateMPICUDAWithArray(PetscObjectComm((PetscObject)mat),rbs,mloc,M,NULL,left));
        else CHKERRQ(VecCreateSeqCUDAWithArray(PetscObjectComm((PetscObject)mat),rbs,M,NULL,left));
#endif
      } else {
        if (size>1) CHKERRQ(VecCreateMPIWithArray(PetscObjectComm((PetscObject)mat),rbs,mloc,M,NULL,left));
        else CHKERRQ(VecCreateSeqWithArray(PetscObjectComm((PetscObject)mat),rbs,M,NULL,left));
      }
    }
  }
  PetscFunctionReturn(0);
}

/*@C
   MatNormEstimate - Estimate the 2-norm of a matrix.

   Collective on A

   Input Parameters:
+  A   - the matrix
.  vrn - random vector with normally distributed entries (can be NULL)
-  w   - workspace vector (can be NULL)

   Output Parameter:
.  nrm - the norm estimate

   Notes:
   Does not need access to the matrix entries, just performs a matrix-vector product.
   Based on work by I. Ipsen and coworkers https://ipsen.math.ncsu.edu/ps/slides_ima.pdf

   The input vector vrn must have unit 2-norm.
   If vrn is NULL, then it is created internally and filled with VecSetRandomNormal().

   Level: developer

.seealso: VecSetRandomNormal()
@*/
PetscErrorCode MatNormEstimate(Mat A,Vec vrn,Vec w,PetscReal *nrm)
{
  PetscInt       n;
  Vec            vv=NULL,ww=NULL;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(A,MAT_CLASSID,1);
  PetscValidType(A,1);
  if (vrn) PetscValidHeaderSpecific(vrn,VEC_CLASSID,2);
  if (w) PetscValidHeaderSpecific(w,VEC_CLASSID,3);
  PetscValidRealPointer(nrm,4);

  if (!vrn) {
    CHKERRQ(MatCreateVecs(A,&vv,NULL));
    vrn = vv;
    CHKERRQ(VecSetRandomNormal(vv,NULL,NULL,NULL));
    CHKERRQ(VecNormalize(vv,NULL));
  }
  if (!w) {
    CHKERRQ(MatCreateVecs(A,&ww,NULL));
    w = ww;
  }

  CHKERRQ(MatGetSize(A,&n,NULL));
  CHKERRQ(MatMult(A,vrn,w));
  CHKERRQ(VecNorm(w,NORM_2,nrm));
  *nrm *= PetscSqrtReal((PetscReal)n);

  CHKERRQ(VecDestroy(&vv));
  CHKERRQ(VecDestroy(&ww));
  PetscFunctionReturn(0);
}
