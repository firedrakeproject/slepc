/*
   User interface for SLEPc's quadratic eigenvalue solvers.

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

#if !defined(__SLEPCQEP_H)
#define __SLEPCQEP_H
#include <slepceps.h>
#include <slepcip.h>
#include <slepcds.h>

PETSC_EXTERN PetscErrorCode QEPInitializePackage(void);

/*S
     QEP - Abstract SLEPc object that manages all the quadratic eigenvalue
     problem solvers.

   Level: beginner

.seealso:  QEPCreate()
S*/
typedef struct _p_QEP* QEP;

/*J
    QEPType - String with the name of a quadratic eigensolver

   Level: beginner

.seealso: QEPSetType(), QEP
J*/
typedef const char* QEPType;
#define QEPLINEAR    "linear"
#define QEPQARNOLDI  "qarnoldi"
#define QEPSTOAR     "stoar"

/* Logging support */
PETSC_EXTERN PetscClassId QEP_CLASSID;

/*E
    QEPProblemType - Determines the type of the quadratic eigenproblem

    Level: intermediate

.seealso: QEPSetProblemType(), QEPGetProblemType()
E*/
typedef enum { QEP_GENERAL=1,
               QEP_HERMITIAN,   /* M, C, K  Hermitian */
               QEP_GYROSCOPIC   /* M, K  Hermitian, M>0, C skew-Hermitian */
             } QEPProblemType;

/*E
    QEPWhich - Determines which part of the spectrum is requested

    Level: intermediate

.seealso: QEPSetWhichEigenpairs(), QEPGetWhichEigenpairs()
E*/
typedef enum { QEP_LARGEST_MAGNITUDE=1,
               QEP_SMALLEST_MAGNITUDE,
               QEP_LARGEST_REAL,
               QEP_SMALLEST_REAL,
               QEP_LARGEST_IMAGINARY,
               QEP_SMALLEST_IMAGINARY,
               QEP_TARGET_MAGNITUDE,
               QEP_TARGET_REAL,
               QEP_TARGET_IMAGINARY} QEPWhich;

PETSC_EXTERN PetscErrorCode QEPCreate(MPI_Comm,QEP*);
PETSC_EXTERN PetscErrorCode QEPDestroy(QEP*);
PETSC_EXTERN PetscErrorCode QEPReset(QEP);
PETSC_EXTERN PetscErrorCode QEPSetType(QEP,QEPType);
PETSC_EXTERN PetscErrorCode QEPGetType(QEP,QEPType*);
PETSC_EXTERN PetscErrorCode QEPSetProblemType(QEP,QEPProblemType);
PETSC_EXTERN PetscErrorCode QEPGetProblemType(QEP,QEPProblemType*);
PETSC_EXTERN PetscErrorCode QEPSetOperators(QEP,Mat,Mat,Mat);
PETSC_EXTERN PetscErrorCode QEPGetOperators(QEP,Mat*,Mat*,Mat*);
PETSC_EXTERN PetscErrorCode QEPSetTarget(QEP,PetscScalar);
PETSC_EXTERN PetscErrorCode QEPGetTarget(QEP,PetscScalar*);
PETSC_EXTERN PetscErrorCode QEPSetST(QEP,ST);
PETSC_EXTERN PetscErrorCode QEPGetST(QEP,ST*);
PETSC_EXTERN PetscErrorCode QEPSetFromOptions(QEP);
PETSC_EXTERN PetscErrorCode QEPSetUp(QEP);
PETSC_EXTERN PetscErrorCode QEPSolve(QEP);
PETSC_EXTERN PetscErrorCode QEPView(QEP,PetscViewer);
PETSC_EXTERN PetscErrorCode QEPPrintSolution(QEP,PetscViewer);

PETSC_EXTERN PetscErrorCode QEPSetBV(QEP,BV);
PETSC_EXTERN PetscErrorCode QEPGetBV(QEP,BV*);
PETSC_EXTERN PetscErrorCode QEPSetDS(QEP,DS);
PETSC_EXTERN PetscErrorCode QEPGetDS(QEP,DS*);
PETSC_EXTERN PetscErrorCode QEPSetTolerances(QEP,PetscReal,PetscInt);
PETSC_EXTERN PetscErrorCode QEPGetTolerances(QEP,PetscReal*,PetscInt*);
PETSC_EXTERN PetscErrorCode QEPSetConvergenceTest(QEP,PetscErrorCode (*)(QEP,PetscScalar,PetscScalar,PetscReal,PetscReal*,void*),void*);
PETSC_EXTERN PetscErrorCode QEPConvergedDefault(QEP,PetscScalar,PetscScalar,PetscReal,PetscReal*,void*);
PETSC_EXTERN PetscErrorCode QEPConvergedAbsolute(QEP,PetscScalar,PetscScalar,PetscReal,PetscReal*,void*);
PETSC_EXTERN PetscErrorCode QEPSetDimensions(QEP,PetscInt,PetscInt,PetscInt);
PETSC_EXTERN PetscErrorCode QEPGetDimensions(QEP,PetscInt*,PetscInt*,PetscInt*);
PETSC_EXTERN PetscErrorCode QEPSetScaleFactor(QEP,PetscReal);
PETSC_EXTERN PetscErrorCode QEPGetScaleFactor(QEP,PetscReal*);

PETSC_EXTERN PetscErrorCode QEPGetConverged(QEP,PetscInt*);
PETSC_EXTERN PetscErrorCode QEPGetEigenpair(QEP,PetscInt,PetscScalar*,PetscScalar*,Vec,Vec);
PETSC_EXTERN PetscErrorCode QEPComputeRelativeError(QEP,PetscInt,PetscReal*);
PETSC_EXTERN PetscErrorCode QEPComputeResidualNorm(QEP,PetscInt,PetscReal*);
PETSC_EXTERN PetscErrorCode QEPGetErrorEstimate(QEP,PetscInt,PetscReal*);

PETSC_EXTERN PetscErrorCode QEPMonitor(QEP,PetscInt,PetscInt,PetscScalar*,PetscScalar*,PetscReal*,PetscInt);
PETSC_EXTERN PetscErrorCode QEPMonitorSet(QEP,PetscErrorCode (*)(QEP,PetscInt,PetscInt,PetscScalar*,PetscScalar*,PetscReal*,PetscInt,void*),void*,PetscErrorCode (*)(void**));
PETSC_EXTERN PetscErrorCode QEPMonitorCancel(QEP);
PETSC_EXTERN PetscErrorCode QEPGetMonitorContext(QEP,void **);
PETSC_EXTERN PetscErrorCode QEPGetIterationNumber(QEP,PetscInt*);
PETSC_EXTERN PetscErrorCode QEPGetOperationCounters(QEP,PetscInt*,PetscInt*,PetscInt*);

PETSC_EXTERN PetscErrorCode QEPSetInitialSpace(QEP,PetscInt,Vec*);
PETSC_EXTERN PetscErrorCode QEPSetInitialSpaceLeft(QEP,PetscInt,Vec*);
PETSC_EXTERN PetscErrorCode QEPSetWhichEigenpairs(QEP,QEPWhich);
PETSC_EXTERN PetscErrorCode QEPGetWhichEigenpairs(QEP,QEPWhich*);
PETSC_EXTERN PetscErrorCode QEPSetLeftVectorsWanted(QEP,PetscBool);
PETSC_EXTERN PetscErrorCode QEPGetLeftVectorsWanted(QEP,PetscBool*);
PETSC_EXTERN PetscErrorCode QEPSetEigenvalueComparison(QEP,PetscErrorCode (*func)(QEP,PetscScalar,PetscScalar,PetscScalar,PetscScalar,PetscInt*,void*),void*);

PETSC_EXTERN PetscErrorCode QEPMonitorAll(QEP,PetscInt,PetscInt,PetscScalar*,PetscScalar*,PetscReal*,PetscInt,void*);
PETSC_EXTERN PetscErrorCode QEPMonitorFirst(QEP,PetscInt,PetscInt,PetscScalar*,PetscScalar*,PetscReal*,PetscInt,void*);
PETSC_EXTERN PetscErrorCode QEPMonitorConverged(QEP,PetscInt,PetscInt,PetscScalar*,PetscScalar*,PetscReal*,PetscInt,void*);
PETSC_EXTERN PetscErrorCode QEPMonitorLG(QEP,PetscInt,PetscInt,PetscScalar*,PetscScalar*,PetscReal*,PetscInt,void*);
PETSC_EXTERN PetscErrorCode QEPMonitorLGAll(QEP,PetscInt,PetscInt,PetscScalar*,PetscScalar*,PetscReal*,PetscInt,void*);

PETSC_EXTERN PetscErrorCode QEPSetTrackAll(QEP,PetscBool);
PETSC_EXTERN PetscErrorCode QEPGetTrackAll(QEP,PetscBool*);

PETSC_EXTERN PetscErrorCode QEPSetOptionsPrefix(QEP,const char*);
PETSC_EXTERN PetscErrorCode QEPAppendOptionsPrefix(QEP,const char*);
PETSC_EXTERN PetscErrorCode QEPGetOptionsPrefix(QEP,const char*[]);

/*E
    QEPConvergedReason - Reason an eigensolver was said to
         have converged or diverged

    Level: beginner

.seealso: QEPSolve(), QEPGetConvergedReason(), QEPSetTolerances()
E*/
typedef enum {/* converged */
              QEP_CONVERGED_TOL                =  2,
              /* diverged */
              QEP_DIVERGED_ITS                 = -3,
              QEP_DIVERGED_BREAKDOWN           = -4,
              QEP_CONVERGED_ITERATING          =  0} QEPConvergedReason;

PETSC_EXTERN PetscErrorCode QEPGetConvergedReason(QEP,QEPConvergedReason *);

PETSC_EXTERN PetscErrorCode QEPSortEigenvalues(QEP,PetscInt,PetscScalar*,PetscScalar*,PetscInt*);
PETSC_EXTERN PetscErrorCode QEPCompareEigenvalues(QEP,PetscScalar,PetscScalar,PetscScalar,PetscScalar,PetscInt*);

PETSC_EXTERN PetscFunctionList QEPList;
PETSC_EXTERN PetscBool         QEPRegisterAllCalled;
PETSC_EXTERN PetscErrorCode QEPRegisterAll(void);
PETSC_EXTERN PetscErrorCode QEPRegister(const char[],PetscErrorCode(*)(QEP));

PETSC_EXTERN PetscErrorCode QEPSetWorkVecs(QEP,PetscInt);

/* --------- options specific to particular eigensolvers -------- */

PETSC_EXTERN PetscErrorCode QEPLinearSetCompanionForm(QEP,PetscInt);
PETSC_EXTERN PetscErrorCode QEPLinearGetCompanionForm(QEP,PetscInt*);
PETSC_EXTERN PetscErrorCode QEPLinearSetExplicitMatrix(QEP,PetscBool);
PETSC_EXTERN PetscErrorCode QEPLinearGetExplicitMatrix(QEP,PetscBool*);
PETSC_EXTERN PetscErrorCode QEPLinearSetEPS(QEP,EPS);
PETSC_EXTERN PetscErrorCode QEPLinearGetEPS(QEP,EPS*);

PETSC_EXTERN PetscErrorCode QEPSTOARSetMonic(QEP,PetscBool);
PETSC_EXTERN PetscErrorCode QEPSTOARGetMonic(QEP,PetscBool*);

#endif

