/*
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
   SLEPc - Scalable Library for Eigenvalue Problem Computations
   Copyright (c) 2002-2018, Universitat Politecnica de Valencia, Spain

   This file is part of SLEPc.
   SLEPc is distributed under a 2-clause BSD license (see LICENSE).
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
*/
/*
   User interface for various vector operations added in SLEPc
*/

#if !defined(__SLEPCVEC_H)
#define __SLEPCVEC_H
#include <petscmat.h>

/* VecComp: Vec composed of several smaller Vecs */
#define VECCOMP  "comp"

PETSC_EXTERN PetscErrorCode VecCreateComp(MPI_Comm,PetscInt*,PetscInt,VecType,Vec,Vec*);
PETSC_EXTERN PetscErrorCode VecCreateCompWithVecs(Vec*,PetscInt,Vec,Vec*);
PETSC_EXTERN PetscErrorCode VecCompGetSubVecs(Vec,PetscInt*,const Vec**);
PETSC_EXTERN PetscErrorCode VecCompSetSubVecs(Vec,PetscInt,Vec*);

/* Some auxiliary functions */
PETSC_EXTERN PetscErrorCode VecNormalizeComplex(Vec,Vec,PetscBool,PetscReal*);
PETSC_EXTERN PetscErrorCode VecCheckOrthogonality(Vec*,PetscInt,Vec*,PetscInt,Mat,PetscViewer,PetscReal*);
PETSC_EXTERN PetscErrorCode VecDuplicateEmpty(Vec,Vec*);

#endif

