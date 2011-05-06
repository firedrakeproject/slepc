/*
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

#include <slepcsys.h>            /*I "slepcsys.h" I*/

#undef __FUNCT__  
#define __FUNCT__ "SlepcIsHermitian"
/*@
   SlepcIsHermitian - Checks if a matrix is Hermitian or not.

   Collective on Mat

   Input parameter:
.  A  - the matrix

   Output parameter:
.  is  - flag indicating if the matrix is Hermitian

   Notes: 
   The result of Ax and A^Hx (with a random x) is compared, but they 
   could be equal also for some non-Hermitian matrices.

   This routine will not work with matrix formats MATSEQSBAIJ or MATMPISBAIJ,
   or when PETSc is configured with complex scalars.
   
   Level: developer

@*/
PetscErrorCode SlepcIsHermitian(Mat A,PetscBool *is)
{
  PetscErrorCode ierr;
  PetscInt       M,N,m,n;
  Vec            x,w1,w2;
  MPI_Comm       comm;
  PetscReal      norm;
  PetscBool      has;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(A,MAT_CLASSID,1);
  PetscValidPointer(is,2);
#if !defined(PETSC_USE_COMPLEX)
  ierr = PetscTypeCompare((PetscObject)A,MATSEQSBAIJ,is);CHKERRQ(ierr);
  if (*is) PetscFunctionReturn(0);
  ierr = PetscTypeCompare((PetscObject)A,MATMPISBAIJ,is);CHKERRQ(ierr);
  if (*is) PetscFunctionReturn(0);
#endif

  *is = PETSC_FALSE;
  ierr = MatGetSize(A,&M,&N);CHKERRQ(ierr);
  ierr = MatGetLocalSize(A,&m,&n);CHKERRQ(ierr);
  if (M!=N) PetscFunctionReturn(0);
  ierr = MatHasOperation(A,MATOP_MULT,&has);CHKERRQ(ierr);
  if (!has) PetscFunctionReturn(0);
  ierr = MatHasOperation(A,MATOP_MULT_TRANSPOSE,&has);CHKERRQ(ierr);
  if (!has) PetscFunctionReturn(0);

  ierr = PetscObjectGetComm((PetscObject)A,&comm);CHKERRQ(ierr);
  ierr = VecCreate(comm,&x);CHKERRQ(ierr);
  ierr = VecSetSizes(x,n,N);CHKERRQ(ierr);
  ierr = VecSetFromOptions(x);CHKERRQ(ierr);
  ierr = SlepcVecSetRandom(x,PETSC_NULL);CHKERRQ(ierr);
  ierr = VecDuplicate(x,&w1);CHKERRQ(ierr);
  ierr = VecDuplicate(x,&w2);CHKERRQ(ierr);
  ierr = MatMult(A,x,w1);CHKERRQ(ierr);
  ierr = MatMultTranspose(A,x,w2);CHKERRQ(ierr);
  ierr = VecConjugate(w2);CHKERRQ(ierr);
  ierr = VecAXPY(w2,-1.0,w1);CHKERRQ(ierr);
  ierr = VecNorm(w2,NORM_2,&norm);CHKERRQ(ierr);
  if (norm<1.0e-6) *is = PETSC_TRUE;
  ierr = VecDestroy(&x);CHKERRQ(ierr);
  ierr = VecDestroy(&w1);CHKERRQ(ierr);
  ierr = VecDestroy(&w2);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#if !defined(PETSC_USE_COMPLEX)

#undef __FUNCT__  
#define __FUNCT__ "SlepcAbsEigenvalue"
/*@C
   SlepcAbsEigenvalue - Returns the absolute value of a complex number given
   its real and imaginary parts.

   Not Collective

   Input parameters:
+  x  - the real part of the complex number
-  y  - the imaginary part of the complex number

   Notes: 
   This function computes sqrt(x**2+y**2), taking care not to cause unnecessary
   overflow. It is based on LAPACK's DLAPY2.

   Level: developer

@*/
PetscReal SlepcAbsEigenvalue(PetscScalar x,PetscScalar y)
{
  PetscReal xabs,yabs,w,z,t;

  PetscFunctionBegin;
  xabs = PetscAbsReal(x);
  yabs = PetscAbsReal(y);
  w = PetscMax(xabs,yabs);
  z = PetscMin(xabs,yabs);
  if (z == 0.0) PetscFunctionReturn(w);
  t = z/w;
  PetscFunctionReturn(w*sqrt(1.0+t*t));  
}

#endif

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
  MPI_Comm       comm;
  Mat            *M;
  IS             isrow,iscol;
  PetscBool      flg;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(mat,MAT_CLASSID,1);
  PetscValidPointer(newmat,2);
  ierr = PetscObjectGetComm((PetscObject)mat,&comm);CHKERRQ(ierr);
  ierr = MPI_Comm_size(comm,&size);CHKERRQ(ierr);

  if (size > 1) {
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
    *newmat = *M;
    ierr = PetscFree(M);CHKERRQ(ierr);     
  
    /* convert matrix to MatSeqDense */
    ierr = PetscTypeCompare((PetscObject)*newmat,MATSEQDENSE,&flg);CHKERRQ(ierr);
    if (!flg) {
      ierr = MatConvert(*newmat,MATSEQDENSE,MAT_INITIAL_MATRIX,newmat);CHKERRQ(ierr);
    } 
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

  ierr = PetscMalloc((M1+M2)*sizeof(PetscInt),&nnz);CHKERRQ(ierr);
  ierr = PetscMemzero(nnz,(M1+M2)*sizeof(PetscInt));CHKERRQ(ierr);
  /* Preallocate for A */
  if (a!=0.0) {
    for (i=0;i<M1;i++) {
      ierr = MatGetRow(A,i,&ncols,PETSC_NULL,PETSC_NULL);CHKERRQ(ierr);
      nnz[i] += ncols;
      ierr = MatRestoreRow(A,i,&ncols,PETSC_NULL,PETSC_NULL);CHKERRQ(ierr);
    }
  }
  /* Preallocate for B */
  if (b!=0.0) {
    for (i=0;i<M1;i++) {
      ierr = MatGetRow(B,i,&ncols,PETSC_NULL,PETSC_NULL);CHKERRQ(ierr);
      nnz[i] += ncols;
      ierr = MatRestoreRow(B,i,&ncols,PETSC_NULL,PETSC_NULL);CHKERRQ(ierr);
    }
  }
  /* Preallocate for C */
  if (c!=0.0) {
    for (i=0;i<M2;i++) {
      ierr = MatGetRow(C,i,&ncols,PETSC_NULL,PETSC_NULL);CHKERRQ(ierr);
      nnz[i+M1] += ncols;
      ierr = MatRestoreRow(C,i,&ncols,PETSC_NULL,PETSC_NULL);CHKERRQ(ierr);
    }
  }
  /* Preallocate for D */
  if (d!=0.0) {
    for (i=0;i<M2;i++) {
      ierr = MatGetRow(D,i,&ncols,PETSC_NULL,PETSC_NULL);CHKERRQ(ierr);
      nnz[i+M1] += ncols;
      ierr = MatRestoreRow(D,i,&ncols,PETSC_NULL,PETSC_NULL);CHKERRQ(ierr);
    }
  }
  ierr = MatSeqAIJSetPreallocation(G,0,nnz);CHKERRQ(ierr);
  ierr = PetscFree(nnz);CHKERRQ(ierr);
  
  ierr = PetscMalloc(sizeof(PetscScalar)*PetscMax(N1,N2),&buf);CHKERRQ(ierr);
  ierr = PetscMalloc(sizeof(PetscInt)*PetscMax(N1,N2),&scols);CHKERRQ(ierr);
  /* Transfer A */
  if (a!=0.0) {
    for (i=0;i<M1;i++) {
      ierr = MatGetRow(A,i,&ncols,&cols,&vals);CHKERRQ(ierr);
      if (a!=1.0) { svals=buf; for (j=0;j<ncols;j++) svals[j] = vals[j]*a; }
      else svals=(PetscScalar*)vals;
      ierr = MatSetValues(G,1,&i,ncols,cols,svals,INSERT_VALUES);CHKERRQ(ierr);
      ierr = MatRestoreRow(A,i,&ncols,&cols,&vals);CHKERRQ(ierr);
    }
  }
  /* Transfer B */
  if (b!=0.0) {
    for (i=0;i<M1;i++) {
      ierr = MatGetRow(B,i,&ncols,&cols,&vals);CHKERRQ(ierr);
      if (b!=1.0) { svals=buf; for (j=0;j<ncols;j++) svals[j] = vals[j]*b; }
      else svals=(PetscScalar*)vals;
      for (j=0;j<ncols;j++) scols[j] = cols[j]+N1;
      ierr = MatSetValues(G,1,&i,ncols,scols,svals,INSERT_VALUES);CHKERRQ(ierr);
      ierr = MatRestoreRow(B,i,&ncols,&cols,&vals);CHKERRQ(ierr);
    }
  }
  /* Transfer C */
  if (c!=0.0) {
    for (i=0;i<M2;i++) {
      ierr = MatGetRow(C,i,&ncols,&cols,&vals);CHKERRQ(ierr);
      if (c!=1.0) { svals=buf; for (j=0;j<ncols;j++) svals[j] = vals[j]*c; }
      else svals=(PetscScalar*)vals;
      j = i+M1;
      ierr = MatSetValues(G,1,&j,ncols,cols,svals,INSERT_VALUES);CHKERRQ(ierr);
      ierr = MatRestoreRow(C,i,&ncols,&cols,&vals);CHKERRQ(ierr);
    }
  }
  /* Transfer D */
  if (d!=0.0) {
    for (i=0;i<M2;i++) {
      ierr = MatGetRow(D,i,&ncols,&cols,&vals);CHKERRQ(ierr);
      if (d!=1.0) { svals=buf; for (j=0;j<ncols;j++) svals[j] = vals[j]*d; }
      else svals=(PetscScalar*)vals;
      for (j=0;j<ncols;j++) scols[j] = cols[j]+N1;
      j = i+M1;
      ierr = MatSetValues(G,1,&j,ncols,scols,svals,INSERT_VALUES);CHKERRQ(ierr);
      ierr = MatRestoreRow(D,i,&ncols,&cols,&vals);CHKERRQ(ierr);
    }
  }
  ierr = PetscFree(buf);CHKERRQ(ierr);
  ierr = PetscFree(scols);CHKERRQ(ierr);
  PetscFunctionReturn(0);  
}

#undef __FUNCT__  
#define __FUNCT__ "SlepcMatTile_MPIAIJ"
static PetscErrorCode SlepcMatTile_MPIAIJ(PetscScalar a,Mat A,PetscScalar b,Mat B,PetscScalar c,Mat C,PetscScalar d,Mat D,Mat G)
{
  PetscErrorCode ierr;
  PetscInt          p,i,j,N1,N2,m1,m2,n1,n2,*map1,*map2,np;
  PetscInt          *dnz,*onz,ncols,*scols,start,gstart;
  PetscScalar       *svals,*buf;
  const PetscInt    *cols,*mapptr1,*mapptr2;
  const PetscScalar *vals;

  PetscFunctionBegin;
  ierr = MatGetSize(A,PETSC_NULL,&N1);CHKERRQ(ierr);
  ierr = MatGetLocalSize(A,&m1,&n1);CHKERRQ(ierr);
  ierr = MatGetSize(D,PETSC_NULL,&N2);CHKERRQ(ierr);
  ierr = MatGetLocalSize(D,&m2,&n2);CHKERRQ(ierr);

  /* Create mappings */
  MPI_Comm_size(((PetscObject)G)->comm,&np);
  ierr = MatGetOwnershipRangesColumn(A,&mapptr1);CHKERRQ(ierr);
  ierr = MatGetOwnershipRangesColumn(B,&mapptr2);CHKERRQ(ierr);
  ierr = PetscMalloc(sizeof(PetscInt)*N1,&map1);CHKERRQ(ierr);
  for (p=0;p<np;p++) {
    for (i=mapptr1[p];i<mapptr1[p+1];i++) map1[i] = i+mapptr2[p];
  }
  ierr = PetscMalloc(sizeof(PetscInt)*N2,&map2);CHKERRQ(ierr);
  for (p=0;p<np;p++) {
    for (i=mapptr2[p];i<mapptr2[p+1];i++) map2[i] = i+mapptr1[p+1];
  }

  ierr = PetscMalloc(sizeof(PetscScalar)*PetscMax(N1,N2),&buf);CHKERRQ(ierr);
  ierr = PetscMalloc(sizeof(PetscInt)*PetscMax(N1,N2),&scols);CHKERRQ(ierr);

  ierr = MatPreallocateInitialize(((PetscObject)G)->comm,m1+m2,n1+n2,dnz,onz);CHKERRQ(ierr);
  ierr = MatGetOwnershipRange(G,&gstart,PETSC_NULL);CHKERRQ(ierr);
  /* Preallocate for A */
  if (a!=0.0) {
    ierr = MatGetOwnershipRange(A,&start,PETSC_NULL);CHKERRQ(ierr);
    for (i=0;i<m1;i++) {
      ierr = MatGetRow(A,i+start,&ncols,&cols,PETSC_NULL);CHKERRQ(ierr);
      for (j=0;j<ncols;j++) scols[j] = map1[cols[j]];
      ierr = MatPreallocateSet(gstart+i,ncols,scols,dnz,onz);CHKERRQ(ierr);
      ierr = MatRestoreRow(A,i+start,&ncols,&cols,PETSC_NULL);CHKERRQ(ierr);
    }
  }
  /* Preallocate for B */
  if (b!=0.0) {
    ierr = MatGetOwnershipRange(B,&start,PETSC_NULL);CHKERRQ(ierr);
    for (i=0;i<m1;i++) {
      ierr = MatGetRow(B,i+start,&ncols,&cols,PETSC_NULL);CHKERRQ(ierr);
      for (j=0;j<ncols;j++) scols[j] = map2[cols[j]];
      ierr = MatPreallocateSet(gstart+i,ncols,scols,dnz,onz);CHKERRQ(ierr);
      ierr = MatRestoreRow(B,i+start,&ncols,&cols,PETSC_NULL);CHKERRQ(ierr);
    }
  }
  /* Preallocate for C */
  if (c!=0.0) {
    ierr = MatGetOwnershipRange(C,&start,PETSC_NULL);CHKERRQ(ierr);
    for (i=0;i<m2;i++) {
      ierr = MatGetRow(C,i+start,&ncols,&cols,PETSC_NULL);CHKERRQ(ierr);
      for (j=0;j<ncols;j++) scols[j] = map1[cols[j]];
      ierr = MatPreallocateSet(gstart+m1+i,ncols,scols,dnz,onz);CHKERRQ(ierr);
      ierr = MatRestoreRow(C,i+start,&ncols,&cols,PETSC_NULL);CHKERRQ(ierr);
    }
  }
  /* Preallocate for D */
  if (d!=0.0) {
    ierr = MatGetOwnershipRange(D,&start,PETSC_NULL);CHKERRQ(ierr);
    for (i=0;i<m2;i++) {
      ierr = MatGetRow(D,i+start,&ncols,&cols,PETSC_NULL);CHKERRQ(ierr);
      for (j=0;j<ncols;j++) scols[j] = map2[cols[j]];
      ierr = MatPreallocateSet(gstart+m1+i,ncols,scols,dnz,onz);CHKERRQ(ierr);
      ierr = MatRestoreRow(D,i+start,&ncols,&cols,PETSC_NULL);CHKERRQ(ierr);
    }
  }
  ierr = MatMPIAIJSetPreallocation(G,0,dnz,0,onz);CHKERRQ(ierr);
  ierr = MatPreallocateFinalize(dnz,onz);CHKERRQ(ierr);
  
  /* Transfer A */
  if (a!=0.0) {
    ierr = MatGetOwnershipRange(A,&start,PETSC_NULL);CHKERRQ(ierr);
    for (i=0;i<m1;i++) {
      ierr = MatGetRow(A,i+start,&ncols,&cols,&vals);CHKERRQ(ierr);
      if (a!=1.0) { svals=buf; for (j=0;j<ncols;j++) svals[j] = vals[j]*a; }
      else svals=(PetscScalar*)vals;
      for (j=0;j<ncols;j++) scols[j] = map1[cols[j]];
      j = gstart+i;
      ierr = MatSetValues(G,1,&j,ncols,scols,svals,INSERT_VALUES);CHKERRQ(ierr);
      ierr = MatRestoreRow(A,i+start,&ncols,&cols,&vals);CHKERRQ(ierr);
    }
  }
  /* Transfer B */
  if (b!=0.0) {
    ierr = MatGetOwnershipRange(B,&start,PETSC_NULL);CHKERRQ(ierr);
    for (i=0;i<m1;i++) {
      ierr = MatGetRow(B,i+start,&ncols,&cols,&vals);CHKERRQ(ierr);
      if (b!=1.0) { svals=buf; for (j=0;j<ncols;j++) svals[j] = vals[j]*b; }
      else svals=(PetscScalar*)vals;
      for (j=0;j<ncols;j++) scols[j] = map2[cols[j]];
      j = gstart+i;
      ierr = MatSetValues(G,1,&j,ncols,scols,svals,INSERT_VALUES);CHKERRQ(ierr);
      ierr = MatRestoreRow(B,i+start,&ncols,&cols,&vals);CHKERRQ(ierr);
    }
  }
  /* Transfer C */
  if (c!=0.0) {
    ierr = MatGetOwnershipRange(C,&start,PETSC_NULL);CHKERRQ(ierr);
    for (i=0;i<m2;i++) {
      ierr = MatGetRow(C,i+start,&ncols,&cols,&vals);CHKERRQ(ierr);
      if (c!=1.0) { svals=buf; for (j=0;j<ncols;j++) svals[j] = vals[j]*c; }
      else svals=(PetscScalar*)vals;
      for (j=0;j<ncols;j++) scols[j] = map1[cols[j]];
      j = gstart+m1+i;
      ierr = MatSetValues(G,1,&j,ncols,scols,svals,INSERT_VALUES);CHKERRQ(ierr);
      ierr = MatRestoreRow(C,i+start,&ncols,&cols,&vals);CHKERRQ(ierr);
    }
  }
  /* Transfer D */
  if (d!=0.0) {
    ierr = MatGetOwnershipRange(D,&start,PETSC_NULL);CHKERRQ(ierr);
    for (i=0;i<m2;i++) {
      ierr = MatGetRow(D,i+start,&ncols,&cols,&vals);CHKERRQ(ierr);
      if (d!=1.0) { svals=buf; for (j=0;j<ncols;j++) svals[j] = vals[j]*d; }
      else svals=(PetscScalar*)vals;
      for (j=0;j<ncols;j++) scols[j] = map2[cols[j]];
      j = gstart+m1+i;
      ierr = MatSetValues(G,1,&j,ncols,scols,svals,INSERT_VALUES);CHKERRQ(ierr);
      ierr = MatRestoreRow(D,i+start,&ncols,&cols,&vals);CHKERRQ(ierr);
    }
  }
  ierr = PetscFree(buf);CHKERRQ(ierr);
  ierr = PetscFree(scols);CHKERRQ(ierr);
  ierr = PetscFree(map1);CHKERRQ(ierr);
  ierr = PetscFree(map2);CHKERRQ(ierr);
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
  ierr = MatGetSize(A,&M1,PETSC_NULL);CHKERRQ(ierr);
  ierr = MatGetLocalSize(A,&m1,PETSC_NULL);CHKERRQ(ierr);
  ierr = MatGetSize(B,&M,PETSC_NULL);CHKERRQ(ierr);
  ierr = MatGetLocalSize(B,&m,PETSC_NULL);CHKERRQ(ierr);
  if (M!=M1 || m!=m1) SETERRQ(((PetscObject)A)->comm,PETSC_ERR_ARG_INCOMP,"Incompatible dimensions");
  /* check row 2 */
  ierr = MatGetSize(C,&M2,PETSC_NULL);CHKERRQ(ierr);
  ierr = MatGetLocalSize(C,&m2,PETSC_NULL);CHKERRQ(ierr);
  ierr = MatGetSize(D,&M,PETSC_NULL);CHKERRQ(ierr);
  ierr = MatGetLocalSize(D,&m,PETSC_NULL);CHKERRQ(ierr);
  if (M!=M2 || m!=m2) SETERRQ(((PetscObject)A)->comm,PETSC_ERR_ARG_INCOMP,"Incompatible dimensions");
  /* check column 1 */
  ierr = MatGetSize(A,PETSC_NULL,&N1);CHKERRQ(ierr);
  ierr = MatGetLocalSize(A,PETSC_NULL,&n1);CHKERRQ(ierr);
  ierr = MatGetSize(C,PETSC_NULL,&N);CHKERRQ(ierr);
  ierr = MatGetLocalSize(C,PETSC_NULL,&n);CHKERRQ(ierr);
  if (N!=N1 || n!=n1) SETERRQ(((PetscObject)A)->comm,PETSC_ERR_ARG_INCOMP,"Incompatible dimensions");
  /* check column 2 */
  ierr = MatGetSize(B,PETSC_NULL,&N2);CHKERRQ(ierr);
  ierr = MatGetLocalSize(B,PETSC_NULL,&n2);CHKERRQ(ierr);
  ierr = MatGetSize(D,PETSC_NULL,&N);CHKERRQ(ierr);
  ierr = MatGetLocalSize(D,PETSC_NULL,&n);CHKERRQ(ierr);
  if (N!=N2 || n!=n2) SETERRQ(((PetscObject)A)->comm,PETSC_ERR_ARG_INCOMP,"Incompatible dimensions");

  ierr = MatCreate(((PetscObject)A)->comm,G);CHKERRQ(ierr);
  ierr = MatSetSizes(*G,m1+m2,n1+n2,M1+M2,N1+N2);CHKERRQ(ierr);
  ierr = MatSetFromOptions(*G);CHKERRQ(ierr);

  ierr = PetscTypeCompare((PetscObject)*G,MATMPIAIJ,&flg1);CHKERRQ(ierr);
  ierr = PetscTypeCompare((PetscObject)A,MATMPIAIJ,&flg2);CHKERRQ(ierr);
  if (flg1 && flg2) {
    ierr = SlepcMatTile_MPIAIJ(a,A,b,B,c,C,d,D,*G);CHKERRQ(ierr);
  }
  else {
    ierr = PetscTypeCompare((PetscObject)*G,MATSEQAIJ,&flg1);CHKERRQ(ierr);
    ierr = PetscTypeCompare((PetscObject)A,MATSEQAIJ,&flg2);CHKERRQ(ierr);
    if (flg1 && flg2) {
      ierr = SlepcMatTile_SeqAIJ(a,A,b,B,c,C,d,D,*G);CHKERRQ(ierr);
    }
    else SETERRQ(((PetscObject)A)->comm,PETSC_ERR_SUP,"Not implemented for this matrix type");
  }
  ierr = MatAssemblyBegin(*G,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(*G,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  PetscFunctionReturn(0);  
}

#undef __FUNCT__  
#define __FUNCT__ "SlepcCheckOrthogonality"
/*@
   SlepcCheckOrthogonality - Checks (or prints) the level of orthogonality
   of a set of vectors.

   Collective on Vec

   Input parameters:
+  V  - a set of vectors
.  nv - number of V vectors
.  W  - an alternative set of vectors (optional)
.  nw - number of W vectors
-  B  - matrix defining the inner product (optional)

   Output parameter:
.  lev - level of orthogonality (optional)

   Notes: 
   This function computes W'*V and prints the result. It is intended to check
   the level of bi-orthogonality of the vectors in the two sets. If W is equal
   to PETSC_NULL then V is used, thus checking the orthogonality of the V vectors.

   If matrix B is provided then the check uses the B-inner product, W'*B*V.

   If lev is not PETSC_NULL, it will contain the level of orthogonality
   computed as ||W'*V - I|| in the Frobenius norm. Otherwise, the matrix W'*V
   is printed.

   Level: developer

@*/
PetscErrorCode SlepcCheckOrthogonality(Vec *V,PetscInt nv,Vec *W,PetscInt nw,Mat B,PetscScalar *lev)
{
  PetscErrorCode ierr;
  PetscInt       i,j;
  PetscScalar    *vals;
  Vec            w;
  MPI_Comm       comm;

  PetscFunctionBegin;
  if (nv<=0 || nw<=0) PetscFunctionReturn(0);
  ierr = PetscObjectGetComm((PetscObject)V[0],&comm);CHKERRQ(ierr);
  ierr = PetscMalloc(nv*sizeof(PetscScalar),&vals);CHKERRQ(ierr);
  if (B) { ierr = VecDuplicate(V[0],&w);CHKERRQ(ierr); }
  if (lev) *lev = 0.0;
  for (i=0;i<nw;i++) {
    if (B) {
      if (W) { ierr = MatMultTranspose(B,W[i],w);CHKERRQ(ierr); }
      else { ierr = MatMultTranspose(B,V[i],w);CHKERRQ(ierr); }
    }
    else {
      if (W) w = W[i];
      else w = V[i];
    }
    ierr = VecMDot(w,nv,V,vals);CHKERRQ(ierr);
    for (j=0;j<nv;j++) {
      if (lev) *lev += (j==i)? (vals[j]-1.0)*(vals[j]-1.0): vals[j]*vals[j];
      else { 
#if !defined(PETSC_USE_COMPLEX)
        ierr = PetscPrintf(comm," %12g  ",vals[j]);CHKERRQ(ierr); 
#else
        ierr = PetscPrintf(comm," %12g%+12gi ",PetscRealPart(vals[j]),PetscImaginaryPart(vals[j]));CHKERRQ(ierr);     
#endif
      }
    }
    if (!lev) { ierr = PetscPrintf(comm,"\n");CHKERRQ(ierr); }
  }
  ierr = PetscFree(vals);CHKERRQ(ierr);
  if (B) { ierr = VecDestroy(&w);CHKERRQ(ierr); }
  if (lev) *lev = PetscSqrtScalar(*lev);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "SlepcConvMonitorDestroy"
/*
  Clean up context used in monitors of type XXXMonitorConverged.
  This function is shared by EPS, SVD, QEP
*/
PetscErrorCode SlepcConvMonitorDestroy(SlepcConvMonitor *ctx)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (!*ctx) PetscFunctionReturn(0);
  ierr = PetscViewerASCIIMonitorDestroy(&(*ctx)->viewer);CHKERRQ(ierr);
  ierr = PetscFree(*ctx);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

