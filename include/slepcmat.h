/*
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
   SLEPc - Scalable Library for Eigenvalue Problem Computations
   Copyright (c) 2002-2019, Universitat Politecnica de Valencia, Spain

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

SLEPC_EXTERN PetscErrorCode MatCreateTile(PetscScalar,Mat,PetscScalar,Mat,PetscScalar,Mat,PetscScalar,Mat,Mat*);
SLEPC_EXTERN PetscErrorCode MatCreateVecsEmpty(Mat,Vec*,Vec*);

/* Deprecated functions */
PETSC_DEPRECATED("Use MatCreateRedundantMatrix() followed by MatConvert()") PETSC_STATIC_INLINE PetscErrorCode SlepcMatConvertSeqDense(Mat mat,Mat *newmat) {
  PetscErrorCode ierr; Mat Ar;
  ierr = MatCreateRedundantMatrix(mat,0,PETSC_COMM_SELF,MAT_INITIAL_MATRIX,&Ar);CHKERRQ(ierr);
  ierr = MatConvert(Ar,MATSEQDENSE,MAT_INITIAL_MATRIX,newmat);CHKERRQ(ierr);
  ierr = MatDestroy(&Ar);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
PETSC_DEPRECATED("Use MatCreateTile()") PETSC_STATIC_INLINE PetscErrorCode SlepcMatTile(PetscScalar a,Mat A,PetscScalar b,Mat B,PetscScalar c,Mat C,PetscScalar d,Mat D,Mat *G) {return MatCreateTile(a,A,b,B,c,C,d,D,G);}

#endif

