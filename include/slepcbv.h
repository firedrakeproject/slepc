/*
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
   SLEPc - Scalable Library for Eigenvalue Problem Computations
   Copyright (c) 2002-2013, Universitat Politecnica de Valencia, Spain

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

#if !defined(__SLEPCBV_H)
#define __SLEPCBV_H
#include <slepcsys.h>

PETSC_EXTERN PetscErrorCode BVInitializePackage(void);

/*S
    BV - Basis vectors, SLEPc object representing a collection of vectors
    that typically constitute a basis of a subspace.

    Level: beginner

.seealso:  BVCreate()
S*/
typedef struct _p_BV* BV;

/*J
    BVType - String with the name of the type of BV. Each type differs in
    the way data is stored internally.

    Level: beginner

.seealso: BVSetType(), BV
J*/
typedef const char* BVType;
#define BVMAT        "mat"
#define BVSVEC       "svec"
#define BVVECS       "vecs"
#define BVCONTIGUOUS "contiguous"

/* Logging support */
PETSC_EXTERN PetscClassId BV_CLASSID;

PETSC_EXTERN PetscErrorCode BVCreate(MPI_Comm,BV*);
PETSC_EXTERN PetscErrorCode BVDestroy(BV*);
PETSC_EXTERN PetscErrorCode BVSetType(BV,BVType);
PETSC_EXTERN PetscErrorCode BVGetType(BV,BVType*);
PETSC_EXTERN PetscErrorCode BVSetSizes(BV,PetscInt,PetscInt,PetscInt);
PETSC_EXTERN PetscErrorCode BVSetSizesFromVec(BV,Vec,PetscInt);
PETSC_EXTERN PetscErrorCode BVGetSizes(BV,PetscInt*,PetscInt*,PetscInt*);
PETSC_EXTERN PetscErrorCode BVSetFromOptions(BV);
PETSC_EXTERN PetscErrorCode BVView(BV,PetscViewer);

PETSC_EXTERN PetscErrorCode BVGetColumn(BV,PetscInt,Vec*);
PETSC_EXTERN PetscErrorCode BVRestoreColumn(BV,PetscInt,Vec*);
PETSC_EXTERN PetscErrorCode BVGetVec(BV,Vec*);
PETSC_EXTERN PetscErrorCode BVSetActiveColumns(BV,PetscInt,PetscInt);
PETSC_EXTERN PetscErrorCode BVGetActiveColumns(BV,PetscInt*,PetscInt*);

PETSC_EXTERN PetscErrorCode BVMult(BV,PetscScalar,PetscScalar,BV,Mat);
PETSC_EXTERN PetscErrorCode BVMultVec(BV,PetscScalar,PetscScalar,Vec,PetscScalar*);
PETSC_EXTERN PetscErrorCode BVMultInPlace(BV,Mat,PetscInt,PetscInt);
PETSC_EXTERN PetscErrorCode BVDot(BV,BV,Mat);
PETSC_EXTERN PetscErrorCode BVDotVec(BV,Vec,PetscScalar*);

PETSC_EXTERN PetscErrorCode BVSetOptionsPrefix(BV,const char*);
PETSC_EXTERN PetscErrorCode BVAppendOptionsPrefix(BV,const char*);
PETSC_EXTERN PetscErrorCode BVGetOptionsPrefix(BV,const char*[]);

PETSC_EXTERN PetscFunctionList BVList;
PETSC_EXTERN PetscBool         BVRegisterAllCalled;
PETSC_EXTERN PetscErrorCode BVRegisterAll(void);
PETSC_EXTERN PetscErrorCode BVRegister(const char[],PetscErrorCode(*)(BV));

#endif

