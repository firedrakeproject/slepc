/*
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
   SLEPc - Scalable Library for Eigenvalue Problem Computations
   Copyright (c) 2002-2018, Universitat Politecnica de Valencia, Spain

   This file is part of SLEPc.
   SLEPc is distributed under a 2-clause BSD license (see LICENSE).
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
*/
/*
   User interface for various matrix operations added in SLEPc
*/

#if !defined(__SLEPCMAT_H)
#define __SLEPCMAT_H
#include <petscmat.h>

PETSC_EXTERN PetscErrorCode MatCreateTile(PetscScalar,Mat,PetscScalar,Mat,PetscScalar,Mat,PetscScalar,Mat,Mat*);
PETSC_EXTERN PetscErrorCode MatCreateVecsEmpty(Mat,Vec*,Vec*);

#endif

