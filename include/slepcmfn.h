/*
   User interface for the SLEPC matrix function object.

   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
   SLEPc - Scalable Library for Eigenvalue Problem Computations
   Copyright (c) 2002-2014, Universitat Politecnica de Valencia, Spain

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

#if !defined(__SLEPCMFN_H)
#define __SLEPCMFN_H
#include <slepcbv.h>
#include <slepcds.h>

PETSC_EXTERN PetscErrorCode MFNInitializePackage(void);

/*S
    MFN - SLEPc object that encapsulates functionality for matrix functions.

    Level: beginner

.seealso:  MFNCreate()
S*/
typedef struct _p_MFN* MFN;

/*J
    MFNType - String with the name of a method for computing matrix functions.

    Level: beginner

.seealso: MFNSetType(), MFN
J*/
typedef const char* MFNType;
#define MFNKRYLOV   "krylov"

/* Logging support */
PETSC_EXTERN PetscClassId MFN_CLASSID;

PETSC_EXTERN PetscErrorCode MFNCreate(MPI_Comm,MFN *);
PETSC_EXTERN PetscErrorCode MFNDestroy(MFN*);
PETSC_EXTERN PetscErrorCode MFNReset(MFN);
PETSC_EXTERN PetscErrorCode MFNSetType(MFN,MFNType);
PETSC_EXTERN PetscErrorCode MFNGetType(MFN,MFNType*);
PETSC_EXTERN PetscErrorCode MFNSetFunction(MFN,SlepcFunction);
PETSC_EXTERN PetscErrorCode MFNGetFunction(MFN,SlepcFunction*);
PETSC_EXTERN PetscErrorCode MFNSetOperator(MFN,Mat);
PETSC_EXTERN PetscErrorCode MFNGetOperator(MFN,Mat*);
PETSC_EXTERN PetscErrorCode MFNSetFromOptions(MFN);
PETSC_EXTERN PetscErrorCode MFNSetUp(MFN);
PETSC_EXTERN PetscErrorCode MFNSolve(MFN,Vec,Vec);
PETSC_EXTERN PetscErrorCode MFNView(MFN,PetscViewer);

PETSC_EXTERN PetscErrorCode MFNSetBV(MFN,BV);
PETSC_EXTERN PetscErrorCode MFNGetBV(MFN,BV*);
PETSC_EXTERN PetscErrorCode MFNSetDS(MFN,DS);
PETSC_EXTERN PetscErrorCode MFNGetDS(MFN,DS*);
PETSC_EXTERN PetscErrorCode MFNSetTolerances(MFN,PetscReal,PetscInt);
PETSC_EXTERN PetscErrorCode MFNGetTolerances(MFN,PetscReal*,PetscInt*);
PETSC_EXTERN PetscErrorCode MFNSetDimensions(MFN,PetscInt);
PETSC_EXTERN PetscErrorCode MFNGetDimensions(MFN,PetscInt*);
PETSC_EXTERN PetscErrorCode MFNSetScaleFactor(MFN,PetscScalar);
PETSC_EXTERN PetscErrorCode MFNGetScaleFactor(MFN,PetscScalar*);

PETSC_EXTERN PetscErrorCode MFNMonitor(MFN,PetscInt,PetscReal);
PETSC_EXTERN PetscErrorCode MFNMonitorSet(MFN,PetscErrorCode (*)(MFN,PetscInt,PetscReal,void*),void*,PetscErrorCode (*)(void**));
PETSC_EXTERN PetscErrorCode MFNMonitorCancel(MFN);
PETSC_EXTERN PetscErrorCode MFNGetMonitorContext(MFN,void **);
PETSC_EXTERN PetscErrorCode MFNGetIterationNumber(MFN,PetscInt*);

PETSC_EXTERN PetscErrorCode MFNSetErrorIfNotConverged(MFN,PetscBool);
PETSC_EXTERN PetscErrorCode MFNGetErrorIfNotConverged(MFN,PetscBool*);

PETSC_EXTERN PetscErrorCode MFNMonitorDefault(MFN,PetscInt,PetscReal,void*);
PETSC_EXTERN PetscErrorCode MFNMonitorLG(MFN,PetscInt,PetscReal,void*);

PETSC_EXTERN PetscErrorCode MFNSetOptionsPrefix(MFN,const char*);
PETSC_EXTERN PetscErrorCode MFNAppendOptionsPrefix(MFN,const char*);
PETSC_EXTERN PetscErrorCode MFNGetOptionsPrefix(MFN,const char*[]);

/*E
    MFNConvergedReason - reason a matrix function iteration was said to
         have converged or diverged

    Level: beginner

.seealso: MFNSolve(), MFNGetConvergedReason(), MFNSetTolerances()
E*/
typedef enum {/* converged */
              MFN_CONVERGED_TOL                =  2,
              /* diverged */
              MFN_DIVERGED_ITS                 = -3,
              MFN_DIVERGED_BREAKDOWN           = -4,
              MFN_CONVERGED_ITERATING          =  0} MFNConvergedReason;

PETSC_EXTERN PetscErrorCode MFNGetConvergedReason(MFN,MFNConvergedReason *);

PETSC_EXTERN PetscFunctionList MFNList;
PETSC_EXTERN PetscBool         MFNRegisterAllCalled;
PETSC_EXTERN PetscErrorCode MFNRegisterAll(void);
PETSC_EXTERN PetscErrorCode MFNRegister(const char[],PetscErrorCode(*)(MFN));

PETSC_EXTERN PetscErrorCode MFNAllocateSolution(MFN,PetscInt);

#endif

