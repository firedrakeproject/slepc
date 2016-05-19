/*
   User interface for various vector operations added in SLEPc.

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

#if !defined(__SLEPCVEC_H)
#define __SLEPCVEC_H
#include <slepcsys.h>
#include <petscmat.h>

/* VecComp: Vec composed of several smaller Vecs */
#define VECCOMP  "comp"

PETSC_EXTERN PetscErrorCode VecCreateComp(MPI_Comm,PetscInt*,PetscInt,VecType,Vec,Vec*);
PETSC_EXTERN PetscErrorCode VecCreateCompWithVecs(Vec*,PetscInt,Vec,Vec*);
PETSC_EXTERN PetscErrorCode VecCompGetSubVecs(Vec,PetscInt*,const Vec**);
PETSC_EXTERN PetscErrorCode VecCompSetSubVecs(Vec,PetscInt,Vec*);

#endif

