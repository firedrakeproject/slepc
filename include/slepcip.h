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

extern PetscErrorCode IPCreate(MPI_Comm,IP*);
extern PetscErrorCode IPSetOptionsPrefix(IP,const char *);
extern PetscErrorCode IPAppendOptionsPrefix(IP,const char *);
extern PetscErrorCode IPGetOptionsPrefix(IP,const char *[]);
extern PetscErrorCode IPSetFromOptions(IP);
extern PetscErrorCode IPSetOrthogonalization(IP,IPOrthogonalizationType,IPOrthogonalizationRefinementType,PetscReal);
extern PetscErrorCode IPGetOrthogonalization(IP,IPOrthogonalizationType*,IPOrthogonalizationRefinementType*,PetscReal*);
extern PetscErrorCode IPView(IP,PetscViewer);
extern PetscErrorCode IPDestroy(IP);

extern PetscErrorCode IPOrthogonalize(IP,PetscInt,Vec*,PetscInt,PetscBool*,Vec*,Vec,PetscScalar*,PetscReal*,PetscBool*);
extern PetscErrorCode IPBiOrthogonalize(IP,PetscInt,Vec*,Vec*,Vec,PetscScalar*,PetscReal*);
extern PetscErrorCode IPQRDecomposition(IP,Vec*,PetscInt,PetscInt,PetscScalar*,PetscInt);

/*E
    IPBilinearForm - determines the type of bilinear/sesquilinear form

    Level: developer

.seealso: IPSetBilinearForm(), IPGetBilinearForm()
E*/
typedef enum { IP_INNER_HERMITIAN,
               IP_INNER_SYMMETRIC } IPBilinearForm;
extern PetscErrorCode IPSetBilinearForm(IP,Mat,IPBilinearForm);
extern PetscErrorCode IPGetBilinearForm(IP,Mat*,IPBilinearForm*);
extern PetscErrorCode IPApplyMatrix(IP,Vec,Vec);

extern PetscErrorCode IPInnerProduct(IP ip,Vec,Vec,PetscScalar*);
extern PetscErrorCode IPInnerProductBegin(IP ip,Vec,Vec,PetscScalar*);
extern PetscErrorCode IPInnerProductEnd(IP ip,Vec,Vec,PetscScalar*);
extern PetscErrorCode IPMInnerProduct(IP ip,Vec,PetscInt,const Vec[],PetscScalar*);
extern PetscErrorCode IPMInnerProductBegin(IP ip,Vec,PetscInt,const Vec[],PetscScalar*);
extern PetscErrorCode IPMInnerProductEnd(IP ip,Vec,PetscInt,const Vec[],PetscScalar*);
extern PetscErrorCode IPNorm(IP ip,Vec,PetscReal*);
extern PetscErrorCode IPNormBegin(IP ip,Vec,PetscReal*);
extern PetscErrorCode IPNormEnd(IP ip,Vec,PetscReal*);

extern PetscErrorCode IPGetOperationCounters(IP,PetscInt*);
extern PetscErrorCode IPResetOperationCounters(IP);

PETSC_EXTERN_CXX_END
#endif
