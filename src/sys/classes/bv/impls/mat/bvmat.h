/*
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
   SLEPc - Scalable Library for Eigenvalue Problem Computations
   Copyright (c) 2002-, Universitat Politecnica de Valencia, Spain

   This file is part of SLEPc.
   SLEPc is distributed under a 2-clause BSD license (see LICENSE).
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
*/

#pragma once

typedef struct {
  Mat       A;
  PetscBool mpi;    /* true if either VECMPI or VECMPICUDA */
} BV_MAT;

#if defined(PETSC_HAVE_CUDA)
SLEPC_INTERN PetscErrorCode BVMult_Mat_CUDA(BV,PetscScalar,PetscScalar,BV,Mat);
SLEPC_INTERN PetscErrorCode BVMultVec_Mat_CUDA(BV,PetscScalar,PetscScalar,Vec,PetscScalar*);
SLEPC_INTERN PetscErrorCode BVMultInPlace_Mat_CUDA(BV,Mat,PetscInt,PetscInt);
SLEPC_INTERN PetscErrorCode BVMultInPlaceHermitianTranspose_Mat_CUDA(BV,Mat,PetscInt,PetscInt);
SLEPC_INTERN PetscErrorCode BVDot_Mat_CUDA(BV,BV,Mat);
SLEPC_INTERN PetscErrorCode BVDotVec_Mat_CUDA(BV,Vec,PetscScalar*);
SLEPC_INTERN PetscErrorCode BVDotVec_Local_Mat_CUDA(BV,Vec,PetscScalar*);
SLEPC_INTERN PetscErrorCode BVScale_Mat_CUDA(BV,PetscInt,PetscScalar);
SLEPC_INTERN PetscErrorCode BVMatMult_Mat_CUDA(BV,Mat,BV);
SLEPC_INTERN PetscErrorCode BVCopy_Mat_CUDA(BV,BV);
SLEPC_INTERN PetscErrorCode BVCopyColumn_Mat_CUDA(BV,PetscInt,PetscInt);
SLEPC_INTERN PetscErrorCode BVResize_Mat_CUDA(BV,PetscInt,PetscBool);
SLEPC_INTERN PetscErrorCode BVGetColumn_Mat_CUDA(BV,PetscInt,Vec*);
SLEPC_INTERN PetscErrorCode BVRestoreColumn_Mat_CUDA(BV,PetscInt,Vec*);
SLEPC_INTERN PetscErrorCode BVRestoreSplit_Mat_CUDA(BV,BV*,BV*);
SLEPC_INTERN PetscErrorCode BVGetMat_Mat_CUDA(BV,Mat*);
SLEPC_INTERN PetscErrorCode BVRestoreMat_Mat_CUDA(BV,Mat*);
#endif
