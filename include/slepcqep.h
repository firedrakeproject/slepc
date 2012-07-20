/*
   User interface for SLEPc's quadratic eigenvalue solvers. 

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

#if !defined(__SLEPCQEP_H)
#define __SLEPCQEP_H
#include "slepcsys.h"
#include "slepceps.h"

PETSC_EXTERN PetscErrorCode QEPInitializePackage(const char[]);

/*S
     QEP - Abstract SLEPc object that manages all the quadratic eigenvalue 
     problem solvers.

   Level: beginner

.seealso:  QEPCreate()
S*/
typedef struct _p_QEP* QEP;

/*E
    QEPType - String with the name of a quadratic eigensolver

   Level: beginner

.seealso: QEPSetType(), QEP
E*/
#define QEPType      char*
#define QEPLINEAR    "linear"
#define QEPQARNOLDI  "qarnoldi"

/* Logging support */
PETSC_EXTERN PetscClassId QEP_CLASSID;

/*E
    QEPProblemType - determines the type of the quadratic eigenproblem

    Level: intermediate

.seealso: QEPSetProblemType(), QEPGetProblemType()
E*/
typedef enum { QEP_GENERAL=1,
               QEP_HERMITIAN,   /* M, C, K  Hermitian */
               QEP_GYROSCOPIC   /* M, K  Hermitian, M>0, C skew-Hermitian */
             } QEPProblemType;

/*E
    QEPWhich - determines which part of the spectrum is requested

    Level: intermediate

.seealso: QEPSetWhichEigenpairs(), QEPGetWhichEigenpairs()
E*/
typedef enum { QEP_LARGEST_MAGNITUDE=1,
               QEP_SMALLEST_MAGNITUDE,
               QEP_LARGEST_REAL,
               QEP_SMALLEST_REAL,
               QEP_LARGEST_IMAGINARY,
               QEP_SMALLEST_IMAGINARY } QEPWhich;

PETSC_EXTERN PetscErrorCode QEPCreate(MPI_Comm,QEP*);
PETSC_EXTERN PetscErrorCode QEPDestroy(QEP*);
PETSC_EXTERN PetscErrorCode QEPReset(QEP);
PETSC_EXTERN PetscErrorCode QEPSetType(QEP,const QEPType);
PETSC_EXTERN PetscErrorCode QEPGetType(QEP,const QEPType*);
PETSC_EXTERN PetscErrorCode QEPSetProblemType(QEP,QEPProblemType);
PETSC_EXTERN PetscErrorCode QEPGetProblemType(QEP,QEPProblemType*);
PETSC_EXTERN PetscErrorCode QEPSetOperators(QEP,Mat,Mat,Mat);
PETSC_EXTERN PetscErrorCode QEPGetOperators(QEP,Mat*,Mat*,Mat*);
PETSC_EXTERN PetscErrorCode QEPSetFromOptions(QEP);
PETSC_EXTERN PetscErrorCode QEPSetUp(QEP);
PETSC_EXTERN PetscErrorCode QEPSolve(QEP);
PETSC_EXTERN PetscErrorCode QEPView(QEP,PetscViewer);
PETSC_EXTERN PetscErrorCode QEPPrintSolution(QEP,PetscViewer);

PETSC_EXTERN PetscErrorCode QEPSetIP(QEP,IP);
PETSC_EXTERN PetscErrorCode QEPGetIP(QEP,IP*);
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
    QEPConvergedReason - reason an eigensolver was said to 
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

PETSC_EXTERN PetscFList QEPList;
PETSC_EXTERN PetscBool  QEPRegisterAllCalled;
PETSC_EXTERN PetscErrorCode QEPRegisterAll(const char[]);
PETSC_EXTERN PetscErrorCode QEPRegisterDestroy(void);
PETSC_EXTERN PetscErrorCode QEPRegister(const char[],const char[],const char[],PetscErrorCode(*)(QEP));

/*MC
   QEPRegisterDynamic - Adds a method to the quadratic eigenproblem solver package.

   Synopsis:
   PetscErrorCode QEPRegisterDynamic(const char *name_solver,const char *path,const char *name_create,PetscErrorCode (*routine_create)(QEP))

   Not Collective

   Input Parameters:
+  name_solver - name of a new user-defined solver
.  path - path (either absolute or relative) the library containing this solver
.  name_create - name of routine to create the solver context
-  routine_create - routine to create the solver context

   Notes:
   QEPRegisterDynamic() may be called multiple times to add several user-defined solvers.

   If dynamic libraries are used, then the fourth input argument (routine_create)
   is ignored.

   Sample usage:
.vb
   QEPRegisterDynamic("my_solver",/home/username/my_lib/lib/libO/solaris/mylib.a,
               "MySolverCreate",MySolverCreate);
.ve

   Then, your solver can be chosen with the procedural interface via
$     QEPSetType(qep,"my_solver")
   or at runtime via the option
$     -qep_type my_solver

   Level: advanced

.seealso: QEPRegisterDestroy(), QEPRegisterAll()
M*/
#if defined(PETSC_USE_DYNAMIC_LIBRARIES)
#define QEPRegisterDynamic(a,b,c,d) QEPRegister(a,b,c,0)
#else
#define QEPRegisterDynamic(a,b,c,d) QEPRegister(a,b,c,d)
#endif

/* --------- options specific to particular eigensolvers -------- */

PETSC_EXTERN PetscErrorCode QEPLinearSetCompanionForm(QEP,PetscInt);
PETSC_EXTERN PetscErrorCode QEPLinearGetCompanionForm(QEP,PetscInt*);
PETSC_EXTERN PetscErrorCode QEPLinearSetExplicitMatrix(QEP,PetscBool);
PETSC_EXTERN PetscErrorCode QEPLinearGetExplicitMatrix(QEP,PetscBool*);
PETSC_EXTERN PetscErrorCode QEPLinearSetEPS(QEP,EPS);
PETSC_EXTERN PetscErrorCode QEPLinearGetEPS(QEP,EPS*);

#endif

