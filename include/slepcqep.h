/*
   User interface for SLEPc's quadratic eigenvalue solvers. 

   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
   SLEPc - Scalable Library for Eigenvalue Problem Computations
   Copyright (c) 2002-2009, Universidad Politecnica de Valencia, Spain

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
PETSC_EXTERN_CXX_BEGIN

extern PetscCookie QEP_COOKIE;

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

/*E
    QEPWhich - determines which part of the spectrum is requested

    Level: intermediate

.seealso: QEPSetWhichEigenpairs(), QEPGetWhichEigenpairs()
E*/
typedef enum { QEP_LARGEST_MAGNITUDE, QEP_SMALLEST_MAGNITUDE,
               QEP_LARGEST_REAL,      QEP_SMALLEST_REAL,
               QEP_LARGEST_IMAGINARY, QEP_SMALLEST_IMAGINARY } QEPWhich;

EXTERN PetscErrorCode QEPCreate(MPI_Comm,QEP*);
EXTERN PetscErrorCode QEPDestroy(QEP);
EXTERN PetscErrorCode QEPSetType(QEP,const QEPType);
EXTERN PetscErrorCode QEPGetType(QEP,const QEPType*);
EXTERN PetscErrorCode QEPSetOperators(QEP,Mat,Mat,Mat);
EXTERN PetscErrorCode QEPGetOperators(QEP,Mat*,Mat*,Mat*);
EXTERN PetscErrorCode QEPSetFromOptions(QEP);
EXTERN PetscErrorCode QEPSetUp(QEP);
EXTERN PetscErrorCode QEPSolve(QEP);
EXTERN PetscErrorCode QEPView(QEP,PetscViewer);

EXTERN PetscErrorCode QEPSetIP(QEP,IP);
EXTERN PetscErrorCode QEPGetIP(QEP,IP*);
EXTERN PetscErrorCode QEPSetTolerances(QEP,PetscReal,PetscInt);
EXTERN PetscErrorCode QEPGetTolerances(QEP,PetscReal*,PetscInt*);
EXTERN PetscErrorCode QEPSetConvergenceTest(QEP,PetscErrorCode (*)(QEP,PetscInt,PetscInt,PetscScalar*,PetscScalar*,PetscReal*,PetscTruth*,void*),void*);
EXTERN PetscErrorCode QEPDefaultConverged(QEP,PetscInt,PetscInt,PetscScalar*,PetscScalar*,PetscReal*,PetscTruth*,void*);
EXTERN PetscErrorCode QEPAbsoluteConverged(QEP,PetscInt,PetscInt,PetscScalar*,PetscScalar*,PetscReal*,PetscTruth*,void*);
EXTERN PetscErrorCode QEPResidualConverged(QEP,PetscInt,PetscInt,PetscScalar*,PetscScalar*,PetscReal*,PetscTruth*,void*);
EXTERN PetscErrorCode QEPSetDimensions(QEP,PetscInt,PetscInt,PetscInt);
EXTERN PetscErrorCode QEPGetDimensions(QEP,PetscInt*,PetscInt*,PetscInt*);

EXTERN PetscErrorCode QEPGetConverged(QEP,PetscInt*);
EXTERN PetscErrorCode QEPGetEigenpair(QEP,PetscInt,PetscScalar*,PetscScalar*,Vec,Vec);
EXTERN PetscErrorCode QEPComputeRelativeError(QEP,PetscInt,PetscReal*);
EXTERN PetscErrorCode QEPComputeResidualNorm(QEP,PetscInt,PetscReal*);
EXTERN PetscErrorCode QEPGetErrorEstimate(QEP,PetscInt,PetscReal*);

EXTERN PetscErrorCode QEPMonitorSet(QEP,PetscErrorCode (*)(QEP,PetscInt,PetscInt,PetscScalar*,PetscScalar*,PetscReal*,PetscInt,void*),
                                    void*,PetscErrorCode (*monitordestroy)(void*));
EXTERN PetscErrorCode QEPMonitorCancel(QEP);
EXTERN PetscErrorCode QEPGetMonitorContext(QEP,void **);
EXTERN PetscErrorCode QEPGetIterationNumber(QEP,PetscInt*);
EXTERN PetscErrorCode QEPGetOperationCounters(QEP,PetscInt*,PetscInt*,PetscInt*);

EXTERN PetscErrorCode QEPSetInitialVector(QEP,Vec);
EXTERN PetscErrorCode QEPGetInitialVector(QEP,Vec*);
EXTERN PetscErrorCode QEPSetWhichEigenpairs(QEP,QEPWhich);
EXTERN PetscErrorCode QEPSetEigenvalueComparison(QEP,PetscErrorCode (*func)(QEP,PetscScalar,PetscScalar,PetscScalar,PetscScalar,PetscInt*,void*),void*);
EXTERN PetscErrorCode QEPGetWhichEigenpairs(QEP,QEPWhich*);

EXTERN PetscErrorCode QEPMonitorDefault(QEP,PetscInt,PetscInt,PetscScalar*,PetscScalar*,PetscReal*,PetscInt,void*);
EXTERN PetscErrorCode QEPMonitorFirst(QEP,PetscInt,PetscInt,PetscScalar*,PetscScalar*,PetscReal*,PetscInt,void*);
EXTERN PetscErrorCode QEPMonitorConverged(QEP,PetscInt,PetscInt,PetscScalar*,PetscScalar*,PetscReal*,PetscInt,void*);
EXTERN PetscErrorCode QEPMonitorLG(QEP,PetscInt,PetscInt,PetscScalar*,PetscScalar*,PetscReal*,PetscInt,void*);

EXTERN PetscErrorCode QEPSetOptionsPrefix(QEP,const char*);
EXTERN PetscErrorCode QEPAppendOptionsPrefix(QEP,const char*);
EXTERN PetscErrorCode QEPGetOptionsPrefix(QEP,const char*[]);

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

EXTERN PetscErrorCode QEPGetConvergedReason(QEP,QEPConvergedReason *);

EXTERN PetscErrorCode QEPSortEigenvalues(QEP,PetscInt,PetscScalar*,PetscScalar*,PetscInt*);
EXTERN PetscErrorCode QEPSortEigenvaluesReal(QEP,PetscInt,PetscReal*,PetscInt*);
EXTERN PetscErrorCode QEPCompareEigenvalues(QEP,PetscScalar,PetscScalar,PetscScalar,PetscScalar,PetscInt*);

EXTERN PetscErrorCode QEPRegister(const char*,const char*,const char*,PetscErrorCode(*)(QEP));
#if defined(PETSC_USE_DYNAMIC_LIBRARIES)
#define QEPRegisterDynamic(a,b,c,d) QEPRegister(a,b,c,0)
#else
#define QEPRegisterDynamic(a,b,c,d) QEPRegister(a,b,c,d)
#endif
EXTERN PetscErrorCode QEPRegisterDestroy(void);

/* --------- options specific to particular eigensolvers -------- */

EXTERN PetscErrorCode QEPLinearSetExplicitMatrix(QEP,PetscTruth);
EXTERN PetscErrorCode QEPLinearGetExplicitMatrix(QEP,PetscTruth*);
EXTERN PetscErrorCode QEPLinearSetEPS(QEP,EPS);
EXTERN PetscErrorCode QEPLinearGetEPS(QEP,EPS*);

PETSC_EXTERN_CXX_END
#endif

