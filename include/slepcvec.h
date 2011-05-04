/*
   User interface for vectors composed by vectors. 

   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
   SLEPc - Scalable Library for Eigenvalue Problem Computations
   Copyright (c) 2002-2010, Universidad Politecnica de Valencia, Spain

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
#include "petscvec.h"

PETSC_EXTERN_CXX_BEGIN

/* VecComp: Vec composed of several smaller Vecs */
#define VECCOMP  "comp"
extern PetscErrorCode VecRegister_Comp(const char[]);
extern PetscErrorCode VecCreateComp(MPI_Comm,PetscInt*,PetscInt,const VecType,Vec,Vec*);
extern PetscErrorCode VecCreateCompWithVecs(Vec*,PetscInt,Vec,Vec*);
extern PetscErrorCode VecCompGetVecs(Vec,const Vec**,PetscInt*);
extern PetscErrorCode VecCompSetVecs(Vec,Vec*,PetscInt);

/* Vecs with contiguous array storage */
extern PetscErrorCode SlepcVecDuplicateVecs(Vec,PetscInt,Vec**);
extern PetscErrorCode SlepcVecDestroyVecs(PetscInt,Vec**);
extern PetscErrorCode SlepcVecsContiguous(Vec**V,PetscBool*);

PETSC_EXTERN_CXX_END
#endif

