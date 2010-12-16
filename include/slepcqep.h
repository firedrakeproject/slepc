/*
   User interface for SLEPc's quadratic eigenvalue solvers. 

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

#if !defined(__SLEPCQEP_H)
#define __SLEPCQEP_H
#include "slepcsys.h"
#include "slepceps.h"
PETSC_EXTERN_CXX_BEGIN

extern PetscClassId QEP_CLASSID;

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

extern PetscErrorCode QEPCreate(MPI_Comm,QEP*);
extern PetscErrorCode QEPDestroy(QEP);
extern PetscErrorCode QEPSetType(QEP,const QEPType);
extern PetscErrorCode QEPGetType(QEP,const QEPType*);
extern PetscErrorCode QEPSetProblemType(QEP,QEPProblemType);
extern PetscErrorCode QEPGetProblemType(QEP,QEPProblemType*);
extern PetscErrorCode QEPSetOperators(QEP,Mat,Mat,Mat);
extern PetscErrorCode QEPGetOperators(QEP,Mat*,Mat*,Mat*);
extern PetscErrorCode QEPSetFromOptions(QEP);
extern PetscErrorCode QEPSetUp(QEP);
extern PetscErrorCode QEPSolve(QEP);
extern PetscErrorCode QEPView(QEP,PetscViewer);

extern PetscErrorCode QEPSetIP(QEP,IP);
extern PetscErrorCode QEPGetIP(QEP,IP*);
extern PetscErrorCode QEPSetTolerances(QEP,PetscReal,PetscInt);
extern PetscErrorCode QEPGetTolerances(QEP,PetscReal*,PetscInt*);
extern PetscErrorCode QEPSetConvergenceTest(QEP,PetscErrorCode (*)(QEP,PetscScalar,PetscScalar,PetscReal,PetscReal*,void*),void*);
extern PetscErrorCode QEPDefaultConverged(QEP,PetscScalar,PetscScalar,PetscReal,PetscReal*,void*);
extern PetscErrorCode QEPAbsoluteConverged(QEP,PetscScalar,PetscScalar,PetscReal,PetscReal*,void*);
extern PetscErrorCode QEPSetDimensions(QEP,PetscInt,PetscInt,PetscInt);
extern PetscErrorCode QEPGetDimensions(QEP,PetscInt*,PetscInt*,PetscInt*);
extern PetscErrorCode QEPSetScaleFactor(QEP,PetscReal);
extern PetscErrorCode QEPGetScaleFactor(QEP,PetscReal*);

extern PetscErrorCode QEPGetConverged(QEP,PetscInt*);
extern PetscErrorCode QEPGetEigenpair(QEP,PetscInt,PetscScalar*,PetscScalar*,Vec,Vec);
extern PetscErrorCode QEPComputeRelativeError(QEP,PetscInt,PetscReal*);
extern PetscErrorCode QEPComputeResidualNorm(QEP,PetscInt,PetscReal*);
extern PetscErrorCode QEPGetErrorEstimate(QEP,PetscInt,PetscReal*);

extern PetscErrorCode QEPMonitorSet(QEP,PetscErrorCode (*)(QEP,PetscInt,PetscInt,PetscScalar*,PetscScalar*,PetscReal*,PetscInt,void*),
                                    void*,PetscErrorCode (*monitordestroy)(void*));
extern PetscErrorCode QEPMonitorCancel(QEP);
extern PetscErrorCode QEPGetMonitorContext(QEP,void **);
extern PetscErrorCode QEPGetIterationNumber(QEP,PetscInt*);
extern PetscErrorCode QEPGetOperationCounters(QEP,PetscInt*,PetscInt*,PetscInt*);

extern PetscErrorCode QEPSetInitialSpace(QEP,PetscInt,Vec*);
extern PetscErrorCode QEPSetInitialSpaceLeft(QEP,PetscInt,Vec*);
extern PetscErrorCode QEPSetWhichEigenpairs(QEP,QEPWhich);
extern PetscErrorCode QEPGetWhichEigenpairs(QEP,QEPWhich*);
extern PetscErrorCode QEPSetLeftVectorsWanted(QEP,PetscBool);
extern PetscErrorCode QEPGetLeftVectorsWanted(QEP,PetscBool*);
extern PetscErrorCode QEPSetEigenvalueComparison(QEP,PetscErrorCode (*func)(QEP,PetscScalar,PetscScalar,PetscScalar,PetscScalar,PetscInt*,void*),void*);

extern PetscErrorCode QEPMonitorAll(QEP,PetscInt,PetscInt,PetscScalar*,PetscScalar*,PetscReal*,PetscInt,void*);
extern PetscErrorCode QEPMonitorFirst(QEP,PetscInt,PetscInt,PetscScalar*,PetscScalar*,PetscReal*,PetscInt,void*);
extern PetscErrorCode QEPMonitorConverged(QEP,PetscInt,PetscInt,PetscScalar*,PetscScalar*,PetscReal*,PetscInt,void*);
extern PetscErrorCode QEPMonitorLG(QEP,PetscInt,PetscInt,PetscScalar*,PetscScalar*,PetscReal*,PetscInt,void*);
extern PetscErrorCode QEPMonitorLGAll(QEP,PetscInt,PetscInt,PetscScalar*,PetscScalar*,PetscReal*,PetscInt,void*);

extern PetscErrorCode QEPSetTrackAll(QEP,PetscBool);
extern PetscErrorCode QEPGetTrackAll(QEP,PetscBool*);

extern PetscErrorCode QEPSetOptionsPrefix(QEP,const char*);
extern PetscErrorCode QEPAppendOptionsPrefix(QEP,const char*);
extern PetscErrorCode QEPGetOptionsPrefix(QEP,const char*[]);

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

extern PetscErrorCode QEPGetConvergedReason(QEP,QEPConvergedReason *);

extern PetscErrorCode QEPSortEigenvalues(QEP,PetscInt,PetscScalar*,PetscScalar*,PetscInt*);
extern PetscErrorCode QEPSortEigenvaluesReal(QEP,PetscInt,PetscReal*,PetscInt*);
extern PetscErrorCode QEPCompareEigenvalues(QEP,PetscScalar,PetscScalar,PetscScalar,PetscScalar,PetscInt*);
extern PetscErrorCode QEPSortDenseSchur(QEP,PetscInt,PetscInt,PetscScalar*,PetscInt,PetscScalar*,PetscScalar*,PetscScalar*);

extern PetscErrorCode QEPRegister(const char*,const char*,const char*,PetscErrorCode(*)(QEP));
#if defined(PETSC_USE_DYNAMIC_LIBRARIES)
#define QEPRegisterDynamic(a,b,c,d) QEPRegister(a,b,c,0)
#else
#define QEPRegisterDynamic(a,b,c,d) QEPRegister(a,b,c,d)
#endif
extern PetscErrorCode QEPRegisterDestroy(void);

/* --------- options specific to particular eigensolvers -------- */

extern PetscErrorCode QEPLinearSetCompanionForm(QEP,PetscInt);
extern PetscErrorCode QEPLinearGetCompanionForm(QEP,PetscInt*);
extern PetscErrorCode QEPLinearSetExplicitMatrix(QEP,PetscBool);
extern PetscErrorCode QEPLinearGetExplicitMatrix(QEP,PetscBool*);
extern PetscErrorCode QEPLinearSetEPS(QEP,EPS);
extern PetscErrorCode QEPLinearGetEPS(QEP,EPS*);

PETSC_EXTERN_CXX_END
#endif

