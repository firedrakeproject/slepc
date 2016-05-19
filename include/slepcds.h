/*
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

#if !defined(__SLEPCDS_H)
#define __SLEPCDS_H
#include <slepcsc.h>
#include <slepcfn.h>

#define DS_MAX_SOLVE 6

PETSC_EXTERN PetscErrorCode DSInitializePackage(void);
/*S
    DS - Direct solver (or dense system), to represent low-dimensional
    eigenproblems that must be solved within iterative solvers. This is an
    auxiliary object and is not normally needed by application programmers.

    Level: beginner

.seealso:  DSCreate()
S*/
typedef struct _p_DS* DS;

/*J
    DSType - String with the name of the type of direct solver. Roughly,
    there are as many types as problem types are available within SLEPc.

    Level: advanced

.seealso: DSSetType(), DS
J*/
typedef const char* DSType;
#define DSHEP             "hep"
#define DSNHEP            "nhep"
#define DSGHEP            "ghep"
#define DSGHIEP           "ghiep"
#define DSGNHEP           "gnhep"
#define DSSVD             "svd"
#define DSPEP             "pep"
#define DSNEP             "nep"

/* Logging support */
PETSC_EXTERN PetscClassId DS_CLASSID;

/*E
    DSStateType - Indicates in which state the direct solver is

    Level: advanced

.seealso: DSSetState()
E*/
typedef enum { DS_STATE_RAW,
               DS_STATE_INTERMEDIATE,
               DS_STATE_CONDENSED,
               DS_STATE_TRUNCATED } DSStateType;

/*E
    DSMatType - Used to refer to one of the matrices stored internally in DS

    Notes:
    The matrices preferently refer to
+   DS_MAT_A  - first matrix of eigenproblem/singular value problem
.   DS_MAT_B  - second matrix of a generalized eigenproblem
.   DS_MAT_C  - third matrix of a quadratic eigenproblem
.   DS_MAT_T  - tridiagonal matrix
.   DS_MAT_D  - diagonal matrix
.   DS_MAT_Q  - orthogonal matrix of (right) Schur vectors
.   DS_MAT_Z  - orthogonal matrix of left Schur vectors
.   DS_MAT_X  - right eigenvectors
.   DS_MAT_Y  - left eigenvectors
.   DS_MAT_U  - left singular vectors
.   DS_MAT_VT - right singular vectors
.   DS_MAT_W  - workspace matrix
-   DS_MAT_Ex - extra matrices (x=0,..,9)

    All matrices can have space to hold ld x ld elements, except for
    DS_MAT_T that has space for 3 x ld elements (ld = leading dimension)
    and DS_MAT_D that has space for just ld elements.

    In DSPEP problems, matrices A, B, W can have space for d*ld x d*ld,
    where d is the polynomial degree, and X can have ld x d*ld.

    Level: advanced

.seealso: DSAllocate(), DSGetArray(), DSGetArrayReal(), DSVectors()
E*/
typedef enum { DS_MAT_A,
               DS_MAT_B,
               DS_MAT_C,
               DS_MAT_T,
               DS_MAT_D,
               DS_MAT_Q,
               DS_MAT_Z,
               DS_MAT_X,
               DS_MAT_Y,
               DS_MAT_U,
               DS_MAT_VT,
               DS_MAT_W,
               DS_MAT_E0,
               DS_MAT_E1,
               DS_MAT_E2,
               DS_MAT_E3,
               DS_MAT_E4,
               DS_MAT_E5,
               DS_MAT_E6,
               DS_MAT_E7,
               DS_MAT_E8,
               DS_MAT_E9,
               DS_NUM_MAT } DSMatType;

/* Convenience for indexing extra matrices */
PETSC_EXTERN DSMatType DSMatExtra[];
#define DS_NUM_EXTRA  10

PETSC_EXTERN PetscErrorCode DSCreate(MPI_Comm,DS*);
PETSC_EXTERN PetscErrorCode DSSetType(DS,DSType);
PETSC_EXTERN PetscErrorCode DSGetType(DS,DSType*);
PETSC_EXTERN PetscErrorCode DSSetOptionsPrefix(DS,const char *);
PETSC_EXTERN PetscErrorCode DSAppendOptionsPrefix(DS,const char *);
PETSC_EXTERN PetscErrorCode DSGetOptionsPrefix(DS,const char *[]);
PETSC_EXTERN PetscErrorCode DSSetFromOptions(DS);
PETSC_EXTERN PetscErrorCode DSView(DS,PetscViewer);
PETSC_EXTERN PetscErrorCode DSViewMat(DS,PetscViewer,DSMatType);
PETSC_EXTERN PetscErrorCode DSDestroy(DS*);
PETSC_EXTERN PetscErrorCode DSReset(DS);

PETSC_EXTERN PetscErrorCode DSAllocate(DS,PetscInt);
PETSC_EXTERN PetscErrorCode DSGetLeadingDimension(DS,PetscInt*);
PETSC_EXTERN PetscErrorCode DSSetState(DS,DSStateType);
PETSC_EXTERN PetscErrorCode DSGetState(DS,DSStateType*);
PETSC_EXTERN PetscErrorCode DSSetDimensions(DS,PetscInt,PetscInt,PetscInt,PetscInt);
PETSC_EXTERN PetscErrorCode DSGetDimensions(DS,PetscInt*,PetscInt*,PetscInt*,PetscInt*,PetscInt*);
PETSC_EXTERN PetscErrorCode DSSetBlockSize(DS,PetscInt);
PETSC_EXTERN PetscErrorCode DSGetBlockSize(DS,PetscInt*);
PETSC_EXTERN PetscErrorCode DSTruncate(DS,PetscInt);
PETSC_EXTERN PetscErrorCode DSSetIdentity(DS,DSMatType);
PETSC_EXTERN PetscErrorCode DSSetMethod(DS,PetscInt);
PETSC_EXTERN PetscErrorCode DSGetMethod(DS,PetscInt*);
PETSC_EXTERN PetscErrorCode DSSetCompact(DS,PetscBool);
PETSC_EXTERN PetscErrorCode DSGetCompact(DS,PetscBool*);
PETSC_EXTERN PetscErrorCode DSSetExtraRow(DS,PetscBool);
PETSC_EXTERN PetscErrorCode DSGetExtraRow(DS,PetscBool*);
PETSC_EXTERN PetscErrorCode DSSetRefined(DS,PetscBool);
PETSC_EXTERN PetscErrorCode DSGetRefined(DS,PetscBool*);
PETSC_EXTERN PetscErrorCode DSGetMat(DS,DSMatType,Mat*);
PETSC_EXTERN PetscErrorCode DSRestoreMat(DS,DSMatType,Mat*);
PETSC_EXTERN PetscErrorCode DSGetArray(DS,DSMatType,PetscScalar*[]);
PETSC_EXTERN PetscErrorCode DSRestoreArray(DS,DSMatType,PetscScalar*[]);
PETSC_EXTERN PetscErrorCode DSGetArrayReal(DS,DSMatType,PetscReal*[]);
PETSC_EXTERN PetscErrorCode DSRestoreArrayReal(DS,DSMatType,PetscReal*[]);
PETSC_EXTERN PetscErrorCode DSVectors(DS,DSMatType,PetscInt*,PetscReal*);
PETSC_EXTERN PetscErrorCode DSSolve(DS,PetscScalar*,PetscScalar*);
PETSC_EXTERN PetscErrorCode DSSort(DS,PetscScalar*,PetscScalar*,PetscScalar*,PetscScalar*,PetscInt*);
PETSC_EXTERN PetscErrorCode DSCopyMat(DS,DSMatType,PetscInt,PetscInt,Mat,PetscInt,PetscInt,PetscInt,PetscInt,PetscBool);
PETSC_EXTERN PetscErrorCode DSSetSlepcSC(DS,SlepcSC);
PETSC_EXTERN PetscErrorCode DSGetSlepcSC(DS,SlepcSC*);
PETSC_EXTERN PetscErrorCode DSUpdateExtraRow(DS);
PETSC_EXTERN PetscErrorCode DSCond(DS,PetscReal*);
PETSC_EXTERN PetscErrorCode DSTranslateHarmonic(DS,PetscScalar,PetscReal,PetscBool,PetscScalar*,PetscReal*);
PETSC_EXTERN PetscErrorCode DSTranslateRKS(DS,PetscScalar);
PETSC_EXTERN PetscErrorCode DSNormalize(DS,DSMatType,PetscInt);
PETSC_EXTERN PetscErrorCode DSOrthogonalize(DS,DSMatType,PetscInt,PetscInt*);
PETSC_EXTERN PetscErrorCode DSPseudoOrthogonalize(DS,DSMatType,PetscInt,PetscReal*,PetscInt*,PetscReal*);

/* --------- options specific to particular solvers -------- */

PETSC_EXTERN PetscErrorCode DSPEPSetDegree(DS,PetscInt);
PETSC_EXTERN PetscErrorCode DSPEPGetDegree(DS,PetscInt*);

PETSC_EXTERN PetscErrorCode DSNEPSetFN(DS,PetscInt,FN*);
PETSC_EXTERN PetscErrorCode DSNEPGetFN(DS,PetscInt,FN*);
PETSC_EXTERN PetscErrorCode DSNEPGetNumFN(DS,PetscInt*);

PETSC_EXTERN PetscFunctionList DSList;
PETSC_EXTERN PetscErrorCode DSRegister(const char[],PetscErrorCode(*)(DS));

#endif
