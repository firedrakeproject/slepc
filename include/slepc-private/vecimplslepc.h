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
PetscErrorCode VecDuplicateVecs_Comp(Vec,PetscInt,Vec*[]);
PetscErrorCode VecDestroyVecs_Comp(PetscInt,Vec[]);
PetscErrorCode VecDuplicate_Comp(Vec,Vec*);
PetscErrorCode VecDestroy_Comp(Vec);
PetscErrorCode VecSet_Comp(Vec,PetscScalar);
PetscErrorCode VecView_Comp(Vec,PetscViewer);
PetscErrorCode VecScale_Comp(Vec,PetscScalar);
PetscErrorCode VecCopy_Comp(Vec,Vec);
PetscErrorCode VecSwap_Comp(Vec,Vec);
PetscErrorCode VecAXPY_Comp(Vec,PetscScalar,Vec);
PetscErrorCode VecAYPX_Comp(Vec,PetscScalar,Vec);
PetscErrorCode VecAXPBY_Comp(Vec,PetscScalar,PetscScalar,Vec);
PetscErrorCode VecMAXPY_Comp(Vec,PetscInt,const PetscScalar*,Vec*);
PetscErrorCode VecWAXPY_Comp(Vec,PetscScalar,Vec,Vec);
PetscErrorCode VecAXPBYPCZ_Comp(Vec,PetscScalar,PetscScalar,PetscScalar,Vec,Vec);
PetscErrorCode VecPointwiseMult_Comp(Vec,Vec,Vec);
PetscErrorCode VecPointwiseDivide_Comp(Vec,Vec,Vec);
PetscErrorCode VecGetSize_Comp(Vec,PetscInt*);
PetscErrorCode VecGetLocalSize_Comp(Vec,PetscInt*);
PetscErrorCode VecMax_Comp(Vec,PetscInt*,PetscReal*);
PetscErrorCode VecMin_Comp(Vec,PetscInt*,PetscReal*);
PetscErrorCode VecSetRandom_Comp(Vec,PetscRandom);
PetscErrorCode VecConjugate_Comp(Vec);
PetscErrorCode VecReciprocal_Comp(Vec);
PetscErrorCode VecMaxPointwiseDivide_Comp(Vec,Vec,PetscReal*);
PetscErrorCode VecPointwiseMax_Comp(Vec,Vec,Vec);
PetscErrorCode VecPointwiseMaxAbs_Comp(Vec,Vec,Vec);
PetscErrorCode VecPointwiseMin_Comp(Vec,Vec,Vec);
PetscErrorCode VecDotNorm2_Comp_Seq(Vec,Vec,PetscScalar*,PetscScalar*);
PetscErrorCode VecDotNorm2_Comp_MPI(Vec,Vec,PetscScalar*,PetscScalar*);
PetscErrorCode VecSqrtAbs_Comp(Vec);
PetscErrorCode VecAbs_Comp(Vec);
PetscErrorCode VecExp_Comp(Vec);
PetscErrorCode VecLog_Comp(Vec);
PetscErrorCode VecShift_Comp(Vec,PetscScalar);

#endif
