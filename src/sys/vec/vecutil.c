/*
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
   SLEPc - Scalable Library for Eigenvalue Problem Computations
   Copyright (c) 2002-2020, Universitat Politecnica de Valencia, Spain

   This file is part of SLEPc.
   SLEPc is distributed under a 2-clause BSD license (see LICENSE).
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
*/

#include <slepc/private/vecimplslepc.h>       /*I "slepcvec.h" I*/

/*@
   VecNormalizeComplex - Normalizes a possibly complex vector by the 2-norm.

   Collective on xr

   Input parameters:
+  xr - the real part of the vector (overwritten on output)
.  xi - the imaginary part of the vector (not referenced if iscomplex is false)
-  iscomplex - a flag indicating if the vector is complex

   Output parameter:
.  norm - the vector norm before normalization (can be set to NULL)

   Level: developer
@*/
PetscErrorCode VecNormalizeComplex(Vec xr,Vec xi,PetscBool iscomplex,PetscReal *norm)
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

static PetscErrorCode VecCheckOrthogonality_Private(Vec V[],PetscInt nv,Vec W[],PetscInt nw,Mat B,PetscViewer viewer,PetscReal *lev,PetscBool norm)
{
  PetscErrorCode ierr;
  PetscInt       i,j;
  PetscScalar    *vals;
  PetscBool      isascii;
  Vec            w;

  PetscFunctionBegin;
  if (!lev) {
    if (!viewer) {
      ierr = PetscViewerASCIIGetStdout(PetscObjectComm((PetscObject)*V),&viewer);CHKERRQ(ierr);
    }
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
      if (lev) {
        if (i!=j) *lev = PetscMax(*lev,PetscAbsScalar(vals[j]));
        else if (norm) *lev = PetscMax(*lev,PetscAbsScalar(vals[j]-PetscRealConstant(1.0)));
      } else {
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

/*@
   VecCheckOrthogonality - Checks (or prints) the level of (bi-)orthogonality
   of a set of vectors.

   Collective on V

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
   W'*V - I (in absolute value) omitting the diagonal. Otherwise, the matrix W'*V
   is printed.

   Level: developer

.seealso: VecCheckOrthonormality()
@*/
PetscErrorCode VecCheckOrthogonality(Vec V[],PetscInt nv,Vec W[],PetscInt nw,Mat B,PetscViewer viewer,PetscReal *lev)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidPointer(V,1);
  PetscValidHeaderSpecific(*V,VEC_CLASSID,1);
  PetscValidLogicalCollectiveInt(*V,nv,2);
  PetscValidLogicalCollectiveInt(*V,nw,4);
  if (nv<=0 || nw<=0) PetscFunctionReturn(0);
  if (W) {
    PetscValidPointer(W,3);
    PetscValidHeaderSpecific(*W,VEC_CLASSID,3);
    PetscCheckSameComm(*V,1,*W,3);
  }
  ierr = VecCheckOrthogonality_Private(V,nv,W,nw,B,viewer,lev,PETSC_FALSE);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@
   VecCheckOrthonormality - Checks (or prints) the level of (bi-)orthonormality
   of a set of vectors.

   Collective on V

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
   This function is equivalent to VecCheckOrthonormality(), but in addition it checks
   that the diagonal of W'*V (or W'*B*V) is equal to all ones.

   Level: developer

.seealso: VecCheckOrthogonality()
@*/
PetscErrorCode VecCheckOrthonormality(Vec V[],PetscInt nv,Vec W[],PetscInt nw,Mat B,PetscViewer viewer,PetscReal *lev)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidPointer(V,1);
  PetscValidHeaderSpecific(*V,VEC_CLASSID,1);
  PetscValidLogicalCollectiveInt(*V,nv,2);
  PetscValidLogicalCollectiveInt(*V,nw,4);
  if (nv<=0 || nw<=0) PetscFunctionReturn(0);
  if (W) {
    PetscValidPointer(W,3);
    PetscValidHeaderSpecific(*W,VEC_CLASSID,3);
    PetscCheckSameComm(*V,1,*W,3);
  }
  ierr = VecCheckOrthogonality_Private(V,nv,W,nw,B,viewer,lev,PETSC_TRUE);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@
   VecDuplicateEmpty - Creates a new vector of the same type as an existing vector,
   but without internal array.

   Collective on v

   Input parameters:
.  v - a vector to mimic

   Output parameter:
.  newv - location to put new vector

   Note:
   This is similar to VecDuplicate(), but the new vector does not have an internal
   array, so the intended usage is with VecPlaceArray().

   Level: developer
@*/
PetscErrorCode VecDuplicateEmpty(Vec v,Vec *newv)
{
  PetscErrorCode ierr;
  PetscBool      standard,cuda,mpi;
  PetscInt       N,nloc,bs;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(v,VEC_CLASSID,1);
  PetscValidPointer(newv,2);
  PetscValidType(v,1);

  ierr = PetscObjectTypeCompareAny((PetscObject)v,&standard,VECSEQ,VECMPI,"");CHKERRQ(ierr);
  ierr = PetscObjectTypeCompareAny((PetscObject)v,&cuda,VECSEQCUDA,VECMPICUDA,"");CHKERRQ(ierr);
  if (standard || cuda) {
    ierr = PetscObjectTypeCompareAny((PetscObject)v,&mpi,VECMPI,VECMPICUDA,"");CHKERRQ(ierr);
    ierr = VecGetLocalSize(v,&nloc);CHKERRQ(ierr);
    ierr = VecGetSize(v,&N);CHKERRQ(ierr);
    ierr = VecGetBlockSize(v,&bs);CHKERRQ(ierr);
    if (cuda) {
#if defined(PETSC_HAVE_CUDA)
      if (mpi) {
        ierr = VecCreateMPICUDAWithArray(PetscObjectComm((PetscObject)v),bs,nloc,N,NULL,newv);CHKERRQ(ierr);
      } else {
        ierr = VecCreateSeqCUDAWithArray(PetscObjectComm((PetscObject)v),bs,N,NULL,newv);CHKERRQ(ierr);
      }
#endif
    } else {
      if (mpi) {
        ierr = VecCreateMPIWithArray(PetscObjectComm((PetscObject)v),bs,nloc,N,NULL,newv);CHKERRQ(ierr);
      } else {
        ierr = VecCreateSeqWithArray(PetscObjectComm((PetscObject)v),bs,N,NULL,newv);CHKERRQ(ierr);
      }
    }
  } else {  /* standard duplicate, with internal array */
    ierr = VecDuplicate(v,newv);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

