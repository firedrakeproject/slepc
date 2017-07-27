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

#include <slepc/private/vecimplslepc.h>       /*I "slepcvec.h" I*/

/*@
   VecNormalizeComplex - Normalizes a possibly complex vector by the 2-norm.

   Collective on Vec

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

/*@C
   VecCheckOrthogonality - Checks (or prints) the level of orthogonality
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
PetscErrorCode VecCheckOrthogonality(Vec *V,PetscInt nv,Vec *W,PetscInt nw,Mat B,PetscViewer viewer,PetscReal *lev)
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

/*@
   VecDuplicateEmpty - Creates a new vector of the same type as an existing vector,
   but without internal array.

   Collective on Vec

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
  PetscBool      sup,cuda,mpi;
  PetscInt       N,nloc,bs;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(v,VEC_CLASSID,1);
  PetscValidPointer(newv,2);
  PetscValidType(v,1);

  ierr = PetscObjectTypeCompareAny((PetscObject)v,&sup,VECSEQ,VECMPI,VECSEQCUDA,VECMPICUDA,"");CHKERRQ(ierr);
  if (!sup) SETERRQ1(PetscObjectComm((PetscObject)v),PETSC_ERR_SUP,"Vector type %s not supported",((PetscObject)v)->type_name);
  ierr = PetscObjectTypeCompareAny((PetscObject)v,&cuda,VECSEQCUDA,VECMPICUDA,"");CHKERRQ(ierr);
  ierr = PetscObjectTypeCompareAny((PetscObject)v,&mpi,VECMPI,VECMPICUDA,"");CHKERRQ(ierr);
  ierr = VecGetLocalSize(v,&nloc);CHKERRQ(ierr);
  ierr = VecGetSize(v,&N);CHKERRQ(ierr);
  ierr = VecGetBlockSize(v,&bs);CHKERRQ(ierr);

  if (cuda) {
#if defined(PETSC_HAVE_VECCUDA)
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
  PetscFunctionReturn(0);
}

