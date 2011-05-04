/*
   User interface for various vector operations added in SLEPc.

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

#if !defined(PETSC_USE_DEBUG)
#define SlepcValidVecsContiguous(V,m,arg) do {} while (0)
#else
#define SlepcValidVecsContiguous(V,m,arg) \
  do { \
    PetscErrorCode __ierr; \
    PetscInt       __i; \
    PetscContainer __container; \
    for (__i=0;__i<m;__i++) { \
      PetscValidHeaderSpecific((V)[__i],VEC_CLASSID,arg); \
      __ierr = PetscObjectQuery((PetscObject)((V)[__i]),"contiguous",(PetscObject*)&__container);CHKERRQ(__ierr); \
      if (!__container) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONG,"Contiguous check failed in argument # %d",arg); \
    } \
  } while (0)
#endif

PETSC_EXTERN_CXX_END
#endif

