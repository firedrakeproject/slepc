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
PETSC_EXTERN_CXX_BEGIN

extern PetscErrorCode PSInitializePackage(const char[]);
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
#define PSGNHEP           "gnhep"
#define PSARROWTRIDSYMM   "arrowtridsymm"
#define PSARROWTRIDPSEUDO "arrowtridpseudo"
#define PSSVD             "svd"
#define PSQEP             "qep"

/* Logging support */
extern PetscClassId PS_CLASSID;

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

    Level: advanced

.seealso: PSGetArray(), PSGetArrayReal(), PSComputeVector()
E*/
typedef enum { PS_MAT_A,
               PS_MAT_B,
               PS_MAT_C,
               PS_MAT_Q,
               PS_MAT_X,
               PS_MAT_Y,
               PS_MAT_U,
               PS_MAT_VT,
               PS_MAT_W,
               PS_NUM_MAT } PSMatType;

extern PetscErrorCode PSCreate(MPI_Comm,PS*);
extern PetscErrorCode PSSetType(PS,const PSType);
extern PetscErrorCode PSGetType(PS,const PSType*);
extern PetscErrorCode PSSetOptionsPrefix(PS,const char *);
extern PetscErrorCode PSAppendOptionsPrefix(PS,const char *);
extern PetscErrorCode PSGetOptionsPrefix(PS,const char *[]);
extern PetscErrorCode PSSetFromOptions(PS);
extern PetscErrorCode PSView(PS,PetscViewer);
extern PetscErrorCode PSDestroy(PS*);
extern PetscErrorCode PSReset(PS);

extern PetscErrorCode PSAllocate(PS,PetscInt);
extern PetscErrorCode PSGetLeadingDimension(PS,PetscInt*);
extern PetscErrorCode PSSetState(PS,PSStateType);
extern PetscErrorCode PSGetState(PS,PSStateType*);
extern PetscErrorCode PSSetDimensions(PS,PetscInt,PetscInt,PetscInt);
extern PetscErrorCode PSGetDimensions(PS,PetscInt*,PetscInt*,PetscInt*);
extern PetscErrorCode PSGetArray(PS,PSMatType,PetscScalar *a[]);
extern PetscErrorCode PSRestoreArray(PS,PSMatType,PetscScalar *a[]);
extern PetscErrorCode PSGetArrayReal(PS,PSMatType,PetscReal *a[]);
extern PetscErrorCode PSRestoreArrayReal(PS,PSMatType,PetscReal *a[]);
extern PetscErrorCode PSComputeVector(PS,PetscInt,PSMatType,PetscBool*);
extern PetscErrorCode PSSolve(PS,PetscScalar*,PetscScalar*);
extern PetscErrorCode PSSort(PS,PetscScalar*,PetscScalar*,PetscErrorCode (*)(PetscScalar,PetscScalar,PetscScalar,PetscScalar,PetscInt*,void*),void*);
extern PetscErrorCode PSCond(PS,PetscReal*);

extern PetscFList PSList;
extern PetscBool  PSRegisterAllCalled;
extern PetscErrorCode PSRegisterAll(const char[]);
extern PetscErrorCode PSRegister(const char[],const char[],const char[],PetscErrorCode(*)(PS));
extern PetscErrorCode PSRegisterDestroy(void);

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

PETSC_EXTERN_CXX_END
#endif
