/*
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

#if !defined(__SLEPCIP_H)
#define __SLEPCIP_H
#include "slepcsys.h"
PETSC_EXTERN_CXX_BEGIN

extern PetscClassId IP_CLASSID;

/*E
    IPOrthogonalizationType - determines what type of orthogonalization to use

    Level: advanced

.seealso: IPSetOrthogonalization(), IPGetOrthogonalization(), IPOrthogonalize()
E*/
typedef enum { IP_ORTH_MGS,
               IP_ORTH_CGS } IPOrthogonalizationType;

/*E
    IPOrthogonalizationRefinementType - determines what type of refinement
    to use during orthogonalization

    Level: advanced

.seealso: IPSetOrthogonalization(), IPGetOrthogonalization(), IPOrthogonalize()
E*/
typedef enum { IP_ORTH_REFINE_NEVER,
               IP_ORTH_REFINE_IFNEEDED,
               IP_ORTH_REFINE_ALWAYS } IPOrthogonalizationRefinementType;

/*S
     IP - Abstraction of a vector inner product, that can be defined
     in different ways. Using this object is not required for application
     programmers.

   Level: beginner

.seealso:  IPCreate()
S*/
typedef struct _p_IP* IP;

EXTERN PetscErrorCode IPCreate(MPI_Comm,IP*);
EXTERN PetscErrorCode IPSetOptionsPrefix(IP,const char *);
EXTERN PetscErrorCode IPAppendOptionsPrefix(IP,const char *);
EXTERN PetscErrorCode IPGetOptionsPrefix(IP,const char *[]);
EXTERN PetscErrorCode IPSetFromOptions(IP);
EXTERN PetscErrorCode IPSetOrthogonalization(IP,IPOrthogonalizationType,IPOrthogonalizationRefinementType,PetscReal);
EXTERN PetscErrorCode IPGetOrthogonalization(IP,IPOrthogonalizationType*,IPOrthogonalizationRefinementType*,PetscReal*);
EXTERN PetscErrorCode IPView(IP,PetscViewer);
EXTERN PetscErrorCode IPDestroy(IP);

EXTERN PetscErrorCode IPOrthogonalize(IP,PetscInt,Vec*,PetscInt,PetscTruth*,Vec*,Vec,PetscScalar*,PetscReal*,PetscTruth*);
EXTERN PetscErrorCode IPBiOrthogonalize(IP,PetscInt,Vec*,Vec*,Vec,PetscScalar*,PetscReal*);
EXTERN PetscErrorCode IPQRDecomposition(IP,Vec*,PetscInt,PetscInt,PetscScalar*,PetscInt);

/*E
    IPBilinearForm - determines the type of bilinear/sesquilinear form

    Level: developer

.seealso: IPSetBilinearForm(), IPGetBilinearForm()
E*/
typedef enum { IP_INNER_HERMITIAN,
               IP_INNER_SYMMETRIC } IPBilinearForm;
EXTERN PetscErrorCode IPSetBilinearForm(IP,Mat,IPBilinearForm);
EXTERN PetscErrorCode IPGetBilinearForm(IP,Mat*,IPBilinearForm*);
EXTERN PetscErrorCode IPApplyMatrix(IP,Vec,Vec);

EXTERN PetscErrorCode IPInnerProduct(IP ip,Vec,Vec,PetscScalar*);
EXTERN PetscErrorCode IPInnerProductBegin(IP ip,Vec,Vec,PetscScalar*);
EXTERN PetscErrorCode IPInnerProductEnd(IP ip,Vec,Vec,PetscScalar*);
EXTERN PetscErrorCode IPMInnerProduct(IP ip,Vec,PetscInt,const Vec[],PetscScalar*);
EXTERN PetscErrorCode IPMInnerProductBegin(IP ip,Vec,PetscInt,const Vec[],PetscScalar*);
EXTERN PetscErrorCode IPMInnerProductEnd(IP ip,Vec,PetscInt,const Vec[],PetscScalar*);
EXTERN PetscErrorCode IPNorm(IP ip,Vec,PetscReal*);
EXTERN PetscErrorCode IPNormBegin(IP ip,Vec,PetscReal*);
EXTERN PetscErrorCode IPNormEnd(IP ip,Vec,PetscReal*);

EXTERN PetscErrorCode IPGetOperationCounters(IP,PetscInt*);
EXTERN PetscErrorCode IPResetOperationCounters(IP);

PETSC_EXTERN_CXX_END
#endif
