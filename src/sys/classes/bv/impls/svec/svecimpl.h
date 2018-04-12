/*
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
   SLEPc - Scalable Library for Eigenvalue Problem Computations
   Copyright (c) 2002-2018, Universitat Politecnica de Valencia, Spain

   This file is part of SLEPc.
   SLEPc is distributed under a 2-clause BSD license (see LICENSE).
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
*/

#if !defined(__SVECIMPL_H)
#define __SVECIMPL_H

typedef struct {
  Vec       v;
  PetscBool mpi;    /* true if either VECMPI or VECMPICUDA */
} BV_SVEC;

PETSC_INTERN PetscErrorCode BVMult_Svec_CUDA(BV,PetscScalar,PetscScalar,BV,Mat);
PETSC_INTERN PetscErrorCode BVMultVec_Svec_CUDA(BV,PetscScalar,PetscScalar,Vec,PetscScalar*);
PETSC_INTERN PetscErrorCode BVMultInPlace_Svec_CUDA(BV,Mat,PetscInt,PetscInt);
PETSC_INTERN PetscErrorCode BVMultInPlaceTranspose_Svec_CUDA(BV,Mat,PetscInt,PetscInt);
PETSC_INTERN PetscErrorCode BVDot_Svec_CUDA(BV,BV,Mat);
PETSC_INTERN PetscErrorCode BVDotVec_Svec_CUDA(BV,Vec,PetscScalar*);
PETSC_INTERN PetscErrorCode BVDotVec_Local_Svec_CUDA(BV,Vec,PetscScalar*);
PETSC_INTERN PetscErrorCode BVScale_Svec_CUDA(BV,PetscInt,PetscScalar);
PETSC_INTERN PetscErrorCode BVMatMult_Svec_CUDA(BV,Mat,BV);
PETSC_INTERN PetscErrorCode BVCopy_Svec_CUDA(BV,BV);
PETSC_INTERN PetscErrorCode BVCopyColumn_Svec_CUDA(BV,PetscInt,PetscInt);
PETSC_INTERN PetscErrorCode BVResize_Svec_CUDA(BV,PetscInt,PetscBool);
PETSC_INTERN PetscErrorCode BVGetColumn_Svec_CUDA(BV,PetscInt,Vec*);
PETSC_INTERN PetscErrorCode BVRestoreColumn_Svec_CUDA(BV,PetscInt,Vec*);
PETSC_INTERN PetscErrorCode BVRestoreSplit_Svec_CUDA(BV,BV*,BV*);

#endif

