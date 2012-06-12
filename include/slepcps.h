/*
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
   SLEPc - Scalable Library for Eigenvalue Problem Computations
   Copyright (c) 2002-2011, Universitat Politecnica de Valencia, Spain

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

#if !defined(__SLEPCPS_H)
#define __SLEPCPS_H
#include "slepcsys.h"

#define PS_MAX_SOLVE 6

PETSC_EXTERN PetscErrorCode PSInitializePackage(const char[]);
/*S
    PS - Projected system, to represent low-dimensional eigenproblems that
    must be solved within iterative solvers. This is an auxiliary object
    and is not normally needed by application programmers.

    Level: advanced

.seealso:  PSCreate()
S*/
typedef struct _p_PS* PS;

/*E
    PSType - String with the name of the type of projected system. Roughly,
    there are as many types as problem types are available within SLEPc,
    with some specific types for particular matrix structures.

    Level: advanced

.seealso: PSSetType(), PS
E*/
#define PSType            char*
#define PSHEP             "hep"
#define PSNHEP            "nhep"
#define PSGHEP            "ghep"
#define PSGHIEP           "ghiep"
#define PSGNHEP           "gnhep"
#define PSSVD             "svd"
#define PSQEP             "qep"

/* Logging support */
PETSC_EXTERN PetscClassId PS_CLASSID;

/*E
    PSStateType - to indicate in which state the projected problem is

    Level: advanced

.seealso: PSSetState()
E*/
typedef enum { PS_STATE_RAW,
               PS_STATE_INTERMEDIATE,
               PS_STATE_CONDENSED,
               PS_STATE_SORTED } PSStateType;

/*E
    PSMatType - to refer to one of the matrices stored internally in PS

    Notes:
    The matrices preferently refer to:
+   PS_MAT_A  - first matrix of eigenproblem/singular value problem
.   PS_MAT_B  - second matrix of a generalized eigenproblem
.   PS_MAT_C  - third matrix of a quadratic eigenproblem
.   PS_MAT_T  - tridiagonal matrix
.   PS_MAT_D  - diagonal matrix
.   PS_MAT_Q  - orthogonal matrix of (right) Schur vectors
.   PS_MAT_Z  - orthogonal matrix of left Schur vectors
.   PS_MAT_X  - right eigenvectors
.   PS_MAT_Y  - left eigenvectors
.   PS_MAT_U  - left singular vectors
.   PS_MAT_VT - right singular vectors
-   PS_MAT_W  - workspace matrix

    All matrices can have space to hold ld x ld elements, except for
    PS_MAT_T that has space for 3 x ld elements (ld = leading dimension)
    and PS_MAT_D that has space for just ld elements.

    Level: advanced

.seealso: PSAllocate(), PSGetArray(), PSGetArrayReal(), PSVectors()
E*/
typedef enum { PS_MAT_A,
               PS_MAT_B,
               PS_MAT_C,
               PS_MAT_T,
               PS_MAT_D,
               PS_MAT_Q,
               PS_MAT_Z,
               PS_MAT_X,
               PS_MAT_Y,
               PS_MAT_U,
               PS_MAT_VT,
               PS_MAT_W,
               PS_NUM_MAT } PSMatType;

PETSC_EXTERN PetscErrorCode PSCreate(MPI_Comm,PS*);
PETSC_EXTERN PetscErrorCode PSSetType(PS,const PSType);
PETSC_EXTERN PetscErrorCode PSGetType(PS,const PSType*);
PETSC_EXTERN PetscErrorCode PSSetOptionsPrefix(PS,const char *);
PETSC_EXTERN PetscErrorCode PSAppendOptionsPrefix(PS,const char *);
PETSC_EXTERN PetscErrorCode PSGetOptionsPrefix(PS,const char *[]);
PETSC_EXTERN PetscErrorCode PSSetFromOptions(PS);
PETSC_EXTERN PetscErrorCode PSView(PS,PetscViewer);
PETSC_EXTERN PetscErrorCode PSDestroy(PS*);
PETSC_EXTERN PetscErrorCode PSReset(PS);

PETSC_EXTERN PetscErrorCode PSAllocate(PS,PetscInt);
PETSC_EXTERN PetscErrorCode PSGetLeadingDimension(PS,PetscInt*);
PETSC_EXTERN PetscErrorCode PSSetState(PS,PSStateType);
PETSC_EXTERN PetscErrorCode PSGetState(PS,PSStateType*);
PETSC_EXTERN PetscErrorCode PSSetDimensions(PS,PetscInt,PetscInt,PetscInt);
PETSC_EXTERN PetscErrorCode PSGetDimensions(PS,PetscInt*,PetscInt*,PetscInt*);
PETSC_EXTERN PetscErrorCode PSTruncate(PS,PetscInt);
PETSC_EXTERN PetscErrorCode PSSetMethod(PS,PetscInt);
PETSC_EXTERN PetscErrorCode PSGetMethod(PS,PetscInt*);
PETSC_EXTERN PetscErrorCode PSSetCompact(PS,PetscBool);
PETSC_EXTERN PetscErrorCode PSGetCompact(PS,PetscBool*);
PETSC_EXTERN PetscErrorCode PSSetExtraRow(PS,PetscBool);
PETSC_EXTERN PetscErrorCode PSGetExtraRow(PS,PetscBool*);
PETSC_EXTERN PetscErrorCode PSSetRefined(PS,PetscBool);
PETSC_EXTERN PetscErrorCode PSGetRefined(PS,PetscBool*);
PETSC_EXTERN PetscErrorCode PSGetArray(PS,PSMatType,PetscScalar *a[]);
PETSC_EXTERN PetscErrorCode PSRestoreArray(PS,PSMatType,PetscScalar *a[]);
PETSC_EXTERN PetscErrorCode PSGetArrayReal(PS,PSMatType,PetscReal *a[]);
PETSC_EXTERN PetscErrorCode PSRestoreArrayReal(PS,PSMatType,PetscReal *a[]);
PETSC_EXTERN PetscErrorCode PSVectors(PS,PSMatType,PetscInt*,PetscReal*);
PETSC_EXTERN PetscErrorCode PSSolve(PS,PetscScalar*,PetscScalar*);
PETSC_EXTERN PetscErrorCode PSSetEigenvalueComparison(PS,PetscErrorCode (*)(PetscScalar,PetscScalar,PetscScalar,PetscScalar,PetscInt*,void*),void*);
PETSC_EXTERN PetscErrorCode PSGetEigenvalueComparison(PS,PetscErrorCode (**)(PetscScalar,PetscScalar,PetscScalar,PetscScalar,PetscInt*,void*),void**);
PETSC_EXTERN PetscErrorCode PSCond(PS,PetscReal*);
PETSC_EXTERN PetscErrorCode PSTranslateHarmonic(PS,PetscScalar,PetscReal,PetscBool,PetscScalar*,PetscReal*);
PETSC_EXTERN PetscErrorCode PSTranslateRKS(PS,PetscScalar);
PETSC_EXTERN PetscErrorCode PSNormalize(PS,PSMatType,PetscInt);
PETSC_EXTERN PetscErrorCode PSSetIdentity(PS,PSMatType);

PETSC_EXTERN PetscFList PSList;
PETSC_EXTERN PetscBool  PSRegisterAllCalled;
PETSC_EXTERN PetscErrorCode PSRegisterAll(const char[]);
PETSC_EXTERN PetscErrorCode PSRegister(const char[],const char[],const char[],PetscErrorCode(*)(PS));
PETSC_EXTERN PetscErrorCode PSRegisterDestroy(void);

/*MC
   PSRegisterDynamic - Adds a projected system to the PS package.

   Synopsis:
   PetscErrorCode PSRegisterDynamic(const char *name,const char *path,const char *name_create,PetscErrorCode (*routine_create)(PS))

   Not collective

   Input Parameters:
+  name - name of a new user-defined PS
.  path - path (either absolute or relative) the library containing this solver
.  name_create - name of routine to create context
-  routine_create - routine to create context

   Notes:
   PSRegisterDynamic() may be called multiple times to add several user-defined
   projected systems.

   If dynamic libraries are used, then the fourth input argument (routine_create)
   is ignored.

   Level: advanced

.seealso: PSRegisterDestroy(), PSRegisterAll()
M*/
#if defined(PETSC_USE_DYNAMIC_LIBRARIES)
#define PSRegisterDynamic(a,b,c,d) PSRegister(a,b,c,0)
#else
#define PSRegisterDynamic(a,b,c,d) PSRegister(a,b,c,d)
#endif

#endif
