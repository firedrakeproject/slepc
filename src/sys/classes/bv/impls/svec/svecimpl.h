/*
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
   SLEPc - Scalable Library for Eigenvalue Problem Computations
   Copyright (c) 2002-2015, Universitat Politecnica de Valencia, Spain

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

#if !defined(__SVECIMPL_H)
#define __SVECIMPL_H

typedef struct {
  Vec       v;
  PetscBool mpi;    /* true if either VECMPI or VECMPICUSP */
  PetscBool cuda;   /* true if either VECSEQCUDA or VECMPICUDA */
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
PETSC_INTERN PetscErrorCode BVResize_Svec_CUDA(BV,PetscInt,PetscBool);
PETSC_INTERN PetscErrorCode BVGetColumn_Svec_CUDA(BV,PetscInt,Vec*);
PETSC_INTERN PetscErrorCode BVRestoreColumn_Svec_CUDA(BV,PetscInt,Vec*);

#endif

