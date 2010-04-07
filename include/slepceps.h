/*
   User interface for the SLEPC eigenproblem solvers. 

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

#if !defined(__SLEPCEPS_H)
#define __SLEPCEPS_H
#include "slepcsys.h"
#include "slepcst.h"
#include "slepcip.h"
PETSC_EXTERN_CXX_BEGIN

extern PetscCookie EPS_COOKIE;

/*S
     EPS - Abstract SLEPc object that manages all the eigenvalue 
     problem solvers.

   Level: beginner

.seealso:  EPSCreate(), ST
S*/
typedef struct _p_EPS* EPS;

/*E
    EPSType - String with the name of a SLEPc eigensolver

   Level: beginner

.seealso: EPSSetType(), EPS
E*/
#define EPSType      char*
#define EPSPOWER     "power"
#define EPSSUBSPACE  "subspace"
#define EPSARNOLDI   "arnoldi"
#define EPSLANCZOS   "lanczos"
#define EPSKRYLOVSCHUR "krylovschur"
#define EPSDSITRLANCZOS "dsitrlanczos"
#define EPSLAPACK    "lapack"
/* the next ones are interfaces to external libraries */
#define EPSARPACK    "arpack"
#define EPSBLZPACK   "blzpack"
#define EPSTRLAN     "trlan"
#define EPSBLOPEX    "blopex"
#define EPSPRIMME    "primme"

/*E
    EPSProblemType - determines the type of eigenvalue problem

    Level: beginner

.seealso: EPSSetProblemType(), EPSGetProblemType()
E*/
typedef enum { EPS_HEP=1,
               EPS_GHEP,
               EPS_NHEP,
               EPS_GNHEP,
               EPS_PGNHEP,
               EPS_GHIEP } EPSProblemType;

/*E
    EPSExtraction - determines the type of extraction technique employed
    by the eigensolver

    Level: beginner

.seealso: EPSSetExtraction(), EPSGetExtraction()
E*/
typedef enum { EPS_RITZ=1,
               EPS_HARMONIC,
               EPS_REFINED,
               EPS_REFINED_HARMONIC } EPSExtraction;

/*E
    EPSWhich - determines which part of the spectrum is requested

    Level: intermediate

.seealso: EPSSetWhichEigenpairs(), EPSGetWhichEigenpairs()
E*/
typedef enum { EPS_LARGEST_MAGNITUDE=1,
               EPS_SMALLEST_MAGNITUDE,
               EPS_LARGEST_REAL,
               EPS_SMALLEST_REAL,
               EPS_LARGEST_IMAGINARY,
               EPS_SMALLEST_IMAGINARY,
               EPS_TARGET_MAGNITUDE,
               EPS_TARGET_REAL, 
               EPS_TARGET_IMAGINARY,
               EPS_WHICH_USER } EPSWhich;

/*E
    EPSBalance - the type of balancing used for non-Hermitian problems

    Level: intermediate

.seealso: EPSSetBalance()
E*/
typedef enum { EPS_BALANCE_NONE=1,
               EPS_BALANCE_ONESIDE,
               EPS_BALANCE_TWOSIDE,
               EPS_BALANCE_USER } EPSBalance;

EXTERN PetscErrorCode EPSCreate(MPI_Comm,EPS *);
EXTERN PetscErrorCode EPSDestroy(EPS);
EXTERN PetscErrorCode EPSSetType(EPS,const EPSType);
EXTERN PetscErrorCode EPSGetType(EPS,const EPSType*);
EXTERN PetscErrorCode EPSSetProblemType(EPS,EPSProblemType);
EXTERN PetscErrorCode EPSGetProblemType(EPS,EPSProblemType*);
EXTERN PetscErrorCode EPSSetExtraction(EPS,EPSExtraction);
EXTERN PetscErrorCode EPSGetExtraction(EPS,EPSExtraction*);
EXTERN PetscErrorCode EPSSetBalance(EPS,EPSBalance,PetscInt,PetscReal);
EXTERN PetscErrorCode EPSGetBalance(EPS,EPSBalance*,PetscInt*,PetscReal*);
EXTERN PetscErrorCode EPSSetOperators(EPS,Mat,Mat);
EXTERN PetscErrorCode EPSGetOperators(EPS,Mat*,Mat*);
EXTERN PetscErrorCode EPSSetFromOptions(EPS);
EXTERN PetscErrorCode EPSSetUp(EPS);
EXTERN PetscErrorCode EPSSolve(EPS);
EXTERN PetscErrorCode EPSView(EPS,PetscViewer);

EXTERN PetscErrorCode EPSSetTarget(EPS,PetscScalar);
EXTERN PetscErrorCode EPSGetTarget(EPS,PetscScalar*);
EXTERN PetscErrorCode EPSSetST(EPS,ST);
EXTERN PetscErrorCode EPSGetST(EPS,ST*);
EXTERN PetscErrorCode EPSSetIP(EPS,IP);
EXTERN PetscErrorCode EPSGetIP(EPS,IP*);
EXTERN PetscErrorCode EPSSetTolerances(EPS,PetscReal,PetscInt);
EXTERN PetscErrorCode EPSGetTolerances(EPS,PetscReal*,PetscInt*);
EXTERN PetscErrorCode EPSSetConvergenceTest(EPS,PetscErrorCode (*)(EPS,PetscInt,PetscInt,PetscScalar*,PetscScalar*,PetscReal*,PetscTruth*,void*),void*);
EXTERN PetscErrorCode EPSDefaultConverged(EPS,PetscInt,PetscInt,PetscScalar*,PetscScalar*,PetscReal*,PetscTruth*,void*);
EXTERN PetscErrorCode EPSAbsoluteConverged(EPS,PetscInt,PetscInt,PetscScalar*,PetscScalar*,PetscReal*,PetscTruth*,void*);
EXTERN PetscErrorCode EPSResidualConverged(EPS,PetscInt,PetscInt,PetscScalar*,PetscScalar*,PetscReal*,PetscTruth*,void*);
EXTERN PetscErrorCode EPSSetDimensions(EPS,PetscInt,PetscInt,PetscInt);
EXTERN PetscErrorCode EPSGetDimensions(EPS,PetscInt*,PetscInt*,PetscInt*);

EXTERN PetscErrorCode EPSGetConverged(EPS,PetscInt*);
EXTERN PetscErrorCode EPSGetEigenpair(EPS,PetscInt,PetscScalar*,PetscScalar*,Vec,Vec);
EXTERN PetscErrorCode EPSGetEigenvalue(EPS,PetscInt,PetscScalar*,PetscScalar*);
EXTERN PetscErrorCode EPSGetEigenvector(EPS,PetscInt,Vec,Vec);
EXTERN PetscErrorCode EPSGetEigenvectorLeft(EPS,PetscInt,Vec,Vec);
EXTERN PetscErrorCode EPSComputeRelativeError(EPS,PetscInt,PetscReal*);
EXTERN PetscErrorCode EPSComputeRelativeErrorLeft(EPS,PetscInt,PetscReal*);
EXTERN PetscErrorCode EPSComputeResidualNorm(EPS,PetscInt,PetscReal*);
EXTERN PetscErrorCode EPSComputeResidualNormLeft(EPS,PetscInt,PetscReal*);
EXTERN PetscErrorCode EPSGetInvariantSubspace(EPS,Vec*);
EXTERN PetscErrorCode EPSGetInvariantSubspaceLeft(EPS,Vec*);
EXTERN PetscErrorCode EPSGetErrorEstimate(EPS,PetscInt,PetscReal*);
EXTERN PetscErrorCode EPSGetErrorEstimateLeft(EPS,PetscInt,PetscReal*);

EXTERN PetscErrorCode EPSMonitorSet(EPS,PetscErrorCode (*)(EPS,PetscInt,PetscInt,PetscScalar*,PetscScalar*,PetscReal*,PetscInt,void*),
                                    void*,PetscErrorCode (*monitordestroy)(void*));
EXTERN PetscErrorCode EPSMonitorCancel(EPS);
EXTERN PetscErrorCode EPSGetMonitorContext(EPS,void **);
EXTERN PetscErrorCode EPSGetIterationNumber(EPS,PetscInt*);
EXTERN PetscErrorCode EPSGetOperationCounters(EPS,PetscInt*,PetscInt*,PetscInt*);

EXTERN PetscErrorCode EPSSetWhichEigenpairs(EPS,EPSWhich);
EXTERN PetscErrorCode EPSGetWhichEigenpairs(EPS,EPSWhich*);
EXTERN PetscErrorCode EPSSetLeftVectorsWanted(EPS,PetscTruth);
EXTERN PetscErrorCode EPSGetLeftVectorsWanted(EPS,PetscTruth*);
EXTERN PetscErrorCode EPSSetEigenvalueComparison(EPS,PetscErrorCode (*func)(EPS,PetscScalar,PetscScalar,PetscScalar,PetscScalar,PetscInt*,void*),void*);
EXTERN PetscErrorCode EPSIsGeneralized(EPS,PetscTruth*);
EXTERN PetscErrorCode EPSIsHermitian(EPS,PetscTruth*);

EXTERN PetscErrorCode EPSMonitorDefault(EPS,PetscInt,PetscInt,PetscScalar*,PetscScalar*,PetscReal*,PetscInt,void*);
EXTERN PetscErrorCode EPSMonitorFirst(EPS,PetscInt,PetscInt,PetscScalar*,PetscScalar*,PetscReal*,PetscInt,void*);
EXTERN PetscErrorCode EPSMonitorConverged(EPS,PetscInt,PetscInt,PetscScalar*,PetscScalar*,PetscReal*,PetscInt,void*);
EXTERN PetscErrorCode EPSMonitorLG(EPS,PetscInt,PetscInt,PetscScalar*,PetscScalar*,PetscReal*,PetscInt,void*);

EXTERN PetscErrorCode EPSSetDeflationSpace(EPS,PetscInt,Vec*);
EXTERN PetscErrorCode EPSRemoveDeflationSpace(EPS);
EXTERN PetscErrorCode EPSSetInitialSpace(EPS,PetscInt,Vec*);
EXTERN PetscErrorCode EPSSetInitialSpaceLeft(EPS,PetscInt,Vec*);

EXTERN PetscErrorCode EPSSetOptionsPrefix(EPS,const char*);
EXTERN PetscErrorCode EPSAppendOptionsPrefix(EPS,const char*);
EXTERN PetscErrorCode EPSGetOptionsPrefix(EPS,const char*[]);

/*E
    EPSConvergedReason - reason an eigensolver was said to 
         have converged or diverged

    Level: beginner

.seealso: EPSSolve(), EPSGetConvergedReason(), EPSSetTolerances()
E*/
typedef enum {/* converged */
              EPS_CONVERGED_TOL                =  2,
              /* diverged */
              EPS_DIVERGED_ITS                 = -3,
              EPS_DIVERGED_BREAKDOWN           = -4,
              EPS_DIVERGED_NONSYMMETRIC        = -5,
              EPS_CONVERGED_ITERATING          =  0} EPSConvergedReason;

EXTERN PetscErrorCode EPSGetConvergedReason(EPS,EPSConvergedReason *);

EXTERN PetscErrorCode EPSSortEigenvalues(EPS,PetscInt,PetscScalar*,PetscScalar*,PetscInt*);
EXTERN PetscErrorCode EPSSortEigenvaluesReal(EPS,PetscInt,PetscReal*,PetscInt*);
EXTERN PetscErrorCode EPSCompareEigenvalues(EPS,PetscScalar,PetscScalar,PetscScalar,PetscScalar,PetscInt*);
EXTERN PetscErrorCode EPSDenseNHEP(PetscInt,PetscScalar*,PetscScalar*,PetscScalar*,PetscScalar*,PetscScalar*);
EXTERN PetscErrorCode EPSDenseGNHEP(PetscInt,PetscScalar*,PetscScalar*,PetscScalar*,PetscScalar*,PetscScalar*,PetscScalar*);
EXTERN PetscErrorCode EPSDenseHEP(PetscInt,PetscScalar*,PetscInt,PetscReal*,PetscScalar*);
EXTERN PetscErrorCode EPSDenseGHEP(PetscInt,PetscScalar*,PetscScalar*,PetscReal*,PetscScalar*);
EXTERN PetscErrorCode EPSDenseHessenberg(PetscInt,PetscInt,PetscScalar*,PetscInt,PetscScalar*);
EXTERN PetscErrorCode EPSDenseSchur(PetscInt,PetscInt,PetscScalar*,PetscInt,PetscScalar*,PetscScalar*,PetscScalar*);
EXTERN PetscErrorCode EPSSortDenseSchur(EPS,PetscInt,PetscInt,PetscScalar*,PetscInt,PetscScalar*,PetscScalar*,PetscScalar*);
EXTERN PetscErrorCode EPSSortDenseSchurGeneralized(EPS,PetscInt,PetscInt,PetscScalar*,PetscScalar*,PetscInt,PetscScalar*,PetscScalar*,PetscScalar*,PetscScalar*);
EXTERN PetscErrorCode EPSDenseTridiagonal(PetscInt,PetscReal*,PetscReal*,PetscReal*,PetscScalar*);

EXTERN PetscErrorCode EPSGetStartVector(EPS,PetscInt,Vec,PetscTruth*);
EXTERN PetscErrorCode EPSGetStartVectorLeft(EPS,PetscInt,Vec,PetscTruth*);

EXTERN PetscErrorCode EPSRegister(const char*,const char*,const char*,PetscErrorCode(*)(EPS));
#if defined(PETSC_USE_DYNAMIC_LIBRARIES)
#define EPSRegisterDynamic(a,b,c,d) EPSRegister(a,b,c,0)
#else
#define EPSRegisterDynamic(a,b,c,d) EPSRegister(a,b,c,d)
#endif
EXTERN PetscErrorCode EPSRegisterDestroy(void);

/* --------- options specific to particular eigensolvers -------- */

/*E
    EPSPowerShiftType - determines the type of shift used in the Power iteration

    Level: advanced

.seealso: EPSPowerSetShiftType(), EPSPowerGetShiftType()
E*/
typedef enum { EPS_POWER_SHIFT_CONSTANT,
               EPS_POWER_SHIFT_RAYLEIGH,
               EPS_POWER_SHIFT_WILKINSON } EPSPowerShiftType;

EXTERN PetscErrorCode EPSPowerSetShiftType(EPS,EPSPowerShiftType);
EXTERN PetscErrorCode EPSPowerGetShiftType(EPS,EPSPowerShiftType*);

EXTERN PetscErrorCode EPSArnoldiSetDelayed(EPS,PetscTruth);
EXTERN PetscErrorCode EPSArnoldiGetDelayed(EPS,PetscTruth*);

/*E
    EPSLanczosReorthogType - determines the type of reorthogonalization
    used in the Lanczos method

    Level: advanced

.seealso: EPSLanczosSetReorthog(), EPSLanczosGetReorthog()
E*/
typedef enum { EPS_LANCZOS_REORTHOG_LOCAL, 
               EPS_LANCZOS_REORTHOG_FULL,
               EPS_LANCZOS_REORTHOG_SELECTIVE,
               EPS_LANCZOS_REORTHOG_PERIODIC,
               EPS_LANCZOS_REORTHOG_PARTIAL, 
               EPS_LANCZOS_REORTHOG_DELAYED } EPSLanczosReorthogType;

EXTERN PetscErrorCode EPSLanczosSetReorthog(EPS,EPSLanczosReorthogType);
EXTERN PetscErrorCode EPSLanczosGetReorthog(EPS,EPSLanczosReorthogType*);

EXTERN PetscErrorCode EPSBlzpackSetBlockSize(EPS,PetscInt);
EXTERN PetscErrorCode EPSBlzpackSetInterval(EPS,PetscReal,PetscReal);
EXTERN PetscErrorCode EPSBlzpackSetNSteps(EPS,PetscInt);

/*E
    EPSPRIMMEMethod - determines the method selected in the PRIMME library

    Level: advanced

.seealso: EPSPRIMMESetMethod(), EPSPRIMMEGetMethod()
E*/
typedef enum { EPS_PRIMME_DYNAMIC,
               EPS_PRIMME_DEFAULT_MIN_TIME,
               EPS_PRIMME_DEFAULT_MIN_MATVECS,
               EPS_PRIMME_ARNOLDI,
               EPS_PRIMME_GD,
               EPS_PRIMME_GD_PLUSK,
               EPS_PRIMME_GD_OLSEN_PLUSK,
               EPS_PRIMME_JD_OLSEN_PLUSK,
               EPS_PRIMME_RQI,
               EPS_PRIMME_JDQR,
               EPS_PRIMME_JDQMR,
               EPS_PRIMME_JDQMR_ETOL,
               EPS_PRIMME_SUBSPACE_ITERATION,
               EPS_PRIMME_LOBPCG_ORTHOBASIS,
               EPS_PRIMME_LOBPCG_ORTHOBASISW } EPSPRIMMEMethod;

/*E
    EPSPRIMMEPrecond - determines the type of preconditioning
    used in the PRIMME library

    Level: advanced

.seealso: EPSPRIMMESetPrecond(), EPSPRIMMEGetPrecond()
E*/
typedef enum { EPS_PRIMME_PRECOND_NONE,
               EPS_PRIMME_PRECOND_DIAGONAL } EPSPRIMMEPrecond;

EXTERN PetscErrorCode EPSPRIMMESetBlockSize(EPS eps,PetscInt bs);
EXTERN PetscErrorCode EPSPRIMMESetMethod(EPS eps, EPSPRIMMEMethod method);
EXTERN PetscErrorCode EPSPRIMMESetPrecond(EPS eps, EPSPRIMMEPrecond precond);
EXTERN PetscErrorCode EPSPRIMMEGetBlockSize(EPS eps,PetscInt *bs);
EXTERN PetscErrorCode EPSPRIMMEGetMethod(EPS eps, EPSPRIMMEMethod *method);
EXTERN PetscErrorCode EPSPRIMMEGetPrecond(EPS eps, EPSPRIMMEPrecond *precond);

PETSC_EXTERN_CXX_END
#endif

