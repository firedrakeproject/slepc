/*
   User interface for various vector operations added in SLEPc.

   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
   SLEPc - Scalable Library for Eigenvalue Problem Computations
   Copyright (c) 2002-2012, Universitat Politecnica de Valencia, Spain

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

#if !defined(__SLEPCVEC_H)
#define __SLEPCVEC_H
#include "petscmat.h"
#include "petscvec.h"

/* VecComp: Vec composed of several smaller Vecs */
#define VECCOMP  "comp"
PETSC_EXTERN PetscErrorCode VecRegister_Comp(const char[]);
PETSC_EXTERN PetscErrorCode VecCreateComp(MPI_Comm,PetscInt*,PetscInt,VecType,Vec,Vec*);
PETSC_EXTERN PetscErrorCode VecCreateCompWithVecs(Vec*,PetscInt,Vec,Vec*);
PETSC_EXTERN PetscErrorCode VecCompGetSubVecs(Vec,PetscInt*,const Vec**);
PETSC_EXTERN PetscErrorCode VecCompSetSubVecs(Vec,PetscInt,Vec*);

/* Vecs with contiguous array storage */
PETSC_EXTERN PetscErrorCode SlepcVecSetTemplate(Vec);
PETSC_EXTERN PetscErrorCode SlepcMatGetVecsTemplate(Mat,Vec*,Vec*);

/* Vec-related operations that have two versions, for contiguous and regular Vecs */
PETSC_EXTERN PetscErrorCode SlepcUpdateVectors(PetscInt,Vec*,PetscInt,PetscInt,const PetscScalar*,PetscInt,PetscBool);
PETSC_EXTERN PetscErrorCode SlepcUpdateStrideVectors(PetscInt n_,Vec *V,PetscInt s,PetscInt d,PetscInt e,const PetscScalar *Q,PetscInt ldq_,PetscBool qtrans);
PETSC_EXTERN PetscErrorCode SlepcVecMAXPBY(Vec,PetscScalar,PetscScalar,PetscInt,PetscScalar*,Vec*);

/* Miscellaneous functions related to Vec */
PETSC_EXTERN PetscErrorCode SlepcVecSetRandom(Vec,PetscRandom);
PETSC_EXTERN PetscErrorCode SlepcVecNormalize(Vec,Vec,PetscBool,PetscReal*);

#endif

