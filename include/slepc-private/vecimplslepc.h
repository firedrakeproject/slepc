/*
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

#ifndef _VECIMPLSLEPC
#define _VECIMPLSLEPC

#include <slepcvec.h>
#include <slepc-private/slepcimpl.h>

PETSC_EXTERN PetscLogEvent SLEPC_UpdateVectors,SLEPC_VecMAXPBY;

/* context for the storage of contiguous Vecs */
typedef struct {
  PetscScalar *array;    /* pointer to common storage */
  PetscInt    nvecs;     /* number of vectors that share this array */
} Vecs_Contiguous;

#if !defined(PETSC_USE_DEBUG)

#define SlepcValidVecsContiguous(V,m,arg) do {} while (0)
#define SlepcValidVecComp(y) do {} while (0)

#else

#define SlepcValidVecsContiguous(V,m,arg) \
  do { \
    PetscErrorCode __ierr; \
    PetscInt       __i; \
    PetscContainer __container; \
    for (__i=0;__i<(m);__i++) { \
      PetscValidHeaderSpecific((V)[__i],VEC_CLASSID,(arg)); \
      __ierr = PetscObjectQuery((PetscObject)((V)[__i]),"contiguous",(PetscObject*)&__container);CHKERRQ(__ierr); \
      if (!__container && (m)>1) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONG,"Contiguous check failed in argument # %d",(arg)); \
    } \
  } while (0)

#define SlepcValidVecComp(y) \
  do { \
    if (((Vec_Comp*)(y)->data)->nx < ((Vec_Comp*)(y)->data)->n->n) \
      SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONG,"Invalid number of subvectors required!"); \
  } while (0)

#endif

/* Contexts for VecComp */
typedef struct {
  PetscInt      n;        /* number of active subvectors */
  PetscInt      N;        /* virtual global size */
  PetscInt      lN;       /* virtual local size */
  PetscInt      friends;  /* number of vectors sharing this structure */
} Vec_Comp_N;

typedef struct {
  Vec           *x;       /* the vectors */
  PetscInt      nx;       /* number of available subvectors */
  Vec_Comp_N    *n;       /* structure shared by friend vectors */
} Vec_Comp;

/* Operations implemented in VecComp */
PETSC_INTERN PetscErrorCode VecDuplicateVecs_Comp(Vec,PetscInt,Vec*[]);
PETSC_INTERN PetscErrorCode VecDestroyVecs_Comp(PetscInt,Vec[]);
PETSC_INTERN PetscErrorCode VecDuplicate_Comp(Vec,Vec*);
PETSC_INTERN PetscErrorCode VecDestroy_Comp(Vec);
PETSC_INTERN PetscErrorCode VecSet_Comp(Vec,PetscScalar);
PETSC_INTERN PetscErrorCode VecView_Comp(Vec,PetscViewer);
PETSC_INTERN PetscErrorCode VecScale_Comp(Vec,PetscScalar);
PETSC_INTERN PetscErrorCode VecCopy_Comp(Vec,Vec);
PETSC_INTERN PetscErrorCode VecSwap_Comp(Vec,Vec);
PETSC_INTERN PetscErrorCode VecAXPY_Comp(Vec,PetscScalar,Vec);
PETSC_INTERN PetscErrorCode VecAYPX_Comp(Vec,PetscScalar,Vec);
PETSC_INTERN PetscErrorCode VecAXPBY_Comp(Vec,PetscScalar,PetscScalar,Vec);
PETSC_INTERN PetscErrorCode VecMAXPY_Comp(Vec,PetscInt,const PetscScalar*,Vec*);
PETSC_INTERN PetscErrorCode VecWAXPY_Comp(Vec,PetscScalar,Vec,Vec);
PETSC_INTERN PetscErrorCode VecAXPBYPCZ_Comp(Vec,PetscScalar,PetscScalar,PetscScalar,Vec,Vec);
PETSC_INTERN PetscErrorCode VecPointwiseMult_Comp(Vec,Vec,Vec);
PETSC_INTERN PetscErrorCode VecPointwiseDivide_Comp(Vec,Vec,Vec);
PETSC_INTERN PetscErrorCode VecGetSize_Comp(Vec,PetscInt*);
PETSC_INTERN PetscErrorCode VecGetLocalSize_Comp(Vec,PetscInt*);
PETSC_INTERN PetscErrorCode VecMax_Comp(Vec,PetscInt*,PetscReal*);
PETSC_INTERN PetscErrorCode VecMin_Comp(Vec,PetscInt*,PetscReal*);
PETSC_INTERN PetscErrorCode VecSetRandom_Comp(Vec,PetscRandom);
PETSC_INTERN PetscErrorCode VecConjugate_Comp(Vec);
PETSC_INTERN PetscErrorCode VecReciprocal_Comp(Vec);
PETSC_INTERN PetscErrorCode VecMaxPointwiseDivide_Comp(Vec,Vec,PetscReal*);
PETSC_INTERN PetscErrorCode VecPointwiseMax_Comp(Vec,Vec,Vec);
PETSC_INTERN PetscErrorCode VecPointwiseMaxAbs_Comp(Vec,Vec,Vec);
PETSC_INTERN PetscErrorCode VecPointwiseMin_Comp(Vec,Vec,Vec);
PETSC_INTERN PetscErrorCode VecDotNorm2_Comp_Seq(Vec,Vec,PetscScalar*,PetscScalar*);
PETSC_INTERN PetscErrorCode VecDotNorm2_Comp_MPI(Vec,Vec,PetscScalar*,PetscScalar*);
PETSC_INTERN PetscErrorCode VecSqrtAbs_Comp(Vec);
PETSC_INTERN PetscErrorCode VecAbs_Comp(Vec);
PETSC_INTERN PetscErrorCode VecExp_Comp(Vec);
PETSC_INTERN PetscErrorCode VecLog_Comp(Vec);
PETSC_INTERN PetscErrorCode VecShift_Comp(Vec,PetscScalar);
PETSC_EXTERN PetscErrorCode VecCreate_Comp(Vec);

#endif
