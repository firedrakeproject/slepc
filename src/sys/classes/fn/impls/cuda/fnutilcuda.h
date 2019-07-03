/*
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
   SLEPc - Scalable Library for Eigenvalue Problem Computations
   Copyright (c) 2002-2019, Universitat Politecnica de Valencia, Spain

   This file is part of SLEPc.
   SLEPc is distributed under a 2-clause BSD license (see LICENSE).
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
*/
/*
   Utility subroutines common to several impls
*/

#include <petsccuda.h>
#include <slepcsys.h>

#if !defined(__FNUTILCUDA_H)
#define __FNUTILCUDA_H

#if defined(PETSC_HAVE_CUDA)

#define X_AXIS 0
#define TILE_SIZE_X  1
#define BLOCK_SIZE_X 1

__global__ void clean_offdiagonal_kernel(PetscScalar *,PetscInt,PetscInt,PetscScalar,PetscInt);
SLEPC_INTERN __host__ PetscErrorCode clean_offdiagonal(PetscScalar *,PetscInt,PetscInt,PetscScalar);
__global__ void set_diagonal_kernel(PetscScalar *,PetscInt,PetscInt,PetscScalar,PetscInt);
SLEPC_INTERN __host__ PetscErrorCode set_diagonal(PetscScalar *,PetscInt,PetscInt,PetscScalar);
__global__ void shift_diagonal_kernel(PetscScalar *,PetscInt,PetscInt,PetscScalar,PetscInt);
SLEPC_INTERN __host__ PetscErrorCode shift_diagonal(PetscScalar *,PetscInt,PetscInt,PetscScalar);
__global__ void mult_diagonal_kernel(PetscScalar *,PetscInt,PetscInt,PetscScalar *,PetscInt);
SLEPC_INTERN __host__ PetscErrorCode mult_diagonal(PetscScalar *,PetscInt, PetscInt,PetscScalar *);
SLEPC_INTERN __host__ PetscErrorCode get_params_1D(PetscInt,dim3 *,dim3 *,PetscInt *);
#endif /* PETSC_HAVE_CUDA */

#endif /* __FNUTILCUDA_H */
