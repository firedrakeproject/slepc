/*
   User interface for the SLEPC eigenproblem solvers.

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

#if !defined(__SLEPCEPS_H)
#define __SLEPCEPS_H
#include <slepcst.h>
#include <slepcbv.h>
#include <slepcds.h>
#include <slepcrg.h>

PETSC_EXTERN PetscErrorCode EPSInitializePackage(void);

/*S
    EPS - Abstract SLEPc object that manages all the eigenvalue
    problem solvers.

    Level: beginner

.seealso:  EPSCreate(), ST
S*/
typedef struct _p_EPS* EPS;

/*J
    EPSType - String with the name of a SLEPc eigensolver

    Level: beginner

.seealso: EPSSetType(), EPS
J*/
typedef const char* EPSType;
#define EPSPOWER       "power"
#define EPSSUBSPACE    "subspace"
#define EPSARNOLDI     "arnoldi"
#define EPSLANCZOS     "lanczos"
#define EPSKRYLOVSCHUR "krylovschur"
#define EPSGD          "gd"
#define EPSJD          "jd"
#define EPSRQCG        "rqcg"
#define EPSLOBPCG      "lobpcg"
#define EPSCISS        "ciss"
#define EPSLAPACK      "lapack"
#define EPSARPACK      "arpack"
#define EPSBLZPACK     "blzpack"
#define EPSTRLAN       "trlan"
#define EPSBLOPEX      "blopex"
#define EPSPRIMME      "primme"
#define EPSFEAST       "feast"

/* Logging support */
PETSC_EXTERN PetscClassId EPS_CLASSID;

/*E
    EPSProblemType - Determines the type of eigenvalue problem

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
    EPSExtraction - Determines the type of extraction technique employed
    by the eigensolver

    Level: advanced

.seealso: EPSSetExtraction(), EPSGetExtraction()
E*/
typedef enum { EPS_RITZ,
               EPS_HARMONIC,
               EPS_HARMONIC_RELATIVE,
               EPS_HARMONIC_RIGHT,
               EPS_HARMONIC_LARGEST,
               EPS_REFINED,
               EPS_REFINED_HARMONIC } EPSExtraction;

/*E
    EPSWhich - Determines which part of the spectrum is requested

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
               EPS_ALL,
               EPS_WHICH_USER } EPSWhich;

/*E
    EPSBalance - The type of balancing used for non-Hermitian problems

    Level: intermediate

.seealso: EPSSetBalance()
E*/
typedef enum { EPS_BALANCE_NONE,
               EPS_BALANCE_ONESIDE,
               EPS_BALANCE_TWOSIDE,
               EPS_BALANCE_USER } EPSBalance;
PETSC_EXTERN const char *EPSBalanceTypes[];

/*E
    EPSErrorType - The error type used to assess accuracy of computed solutions

    Level: intermediate

.seealso: EPSComputeError()
E*/
typedef enum { EPS_ERROR_ABSOLUTE,
               EPS_ERROR_RELATIVE,
               EPS_ERROR_BACKWARD } EPSErrorType;
PETSC_EXTERN const char *EPSErrorTypes[];

/*E
    EPSConv - Determines the convergence test

    Level: intermediate

.seealso: EPSSetConvergenceTest(), EPSSetConvergenceTestFunction()
E*/
typedef enum { EPS_CONV_ABS,
               EPS_CONV_REL,
               EPS_CONV_NORM,
               EPS_CONV_USER } EPSConv;

/*E
    EPSStop - Determines the stopping test

    Level: advanced

.seealso: EPSSetStoppingTest(), EPSSetStoppingTestFunction()
E*/
typedef enum { EPS_STOP_BASIC,
               EPS_STOP_USER } EPSStop;

/*E
    EPSConvergedReason - Reason an eigensolver was said to
         have converged or diverged

    Level: intermediate

.seealso: EPSSolve(), EPSGetConvergedReason(), EPSSetTolerances()
E*/
typedef enum {/* converged */
              EPS_CONVERGED_TOL                =  1,
              EPS_CONVERGED_USER               =  2,
              /* diverged */
              EPS_DIVERGED_ITS                 = -1,
              EPS_DIVERGED_BREAKDOWN           = -2,
              EPS_DIVERGED_SYMMETRY_LOST       = -3,
              EPS_CONVERGED_ITERATING          =  0} EPSConvergedReason;
PETSC_EXTERN const char *const*EPSConvergedReasons;

PETSC_EXTERN PetscErrorCode EPSCreate(MPI_Comm,EPS*);
PETSC_EXTERN PetscErrorCode EPSDestroy(EPS*);
PETSC_EXTERN PetscErrorCode EPSReset(EPS);
PETSC_EXTERN PetscErrorCode EPSSetType(EPS,EPSType);
PETSC_EXTERN PetscErrorCode EPSGetType(EPS,EPSType*);
PETSC_EXTERN PetscErrorCode EPSSetProblemType(EPS,EPSProblemType);
PETSC_EXTERN PetscErrorCode EPSGetProblemType(EPS,EPSProblemType*);
PETSC_EXTERN PetscErrorCode EPSSetExtraction(EPS,EPSExtraction);
PETSC_EXTERN PetscErrorCode EPSGetExtraction(EPS,EPSExtraction*);
PETSC_EXTERN PetscErrorCode EPSSetBalance(EPS,EPSBalance,PetscInt,PetscReal);
PETSC_EXTERN PetscErrorCode EPSGetBalance(EPS,EPSBalance*,PetscInt*,PetscReal*);
PETSC_EXTERN PetscErrorCode EPSSetOperators(EPS,Mat,Mat);
PETSC_EXTERN PetscErrorCode EPSGetOperators(EPS,Mat*,Mat*);
PETSC_EXTERN PetscErrorCode EPSSetFromOptions(EPS);
PETSC_EXTERN PetscErrorCode EPSSetUp(EPS);
PETSC_EXTERN PetscErrorCode EPSSolve(EPS);
PETSC_EXTERN PetscErrorCode EPSView(EPS,PetscViewer);
PETSC_STATIC_INLINE PetscErrorCode EPSViewFromOptions(EPS eps,PetscObject obj,const char name[]) {return PetscObjectViewFromOptions((PetscObject)eps,obj,name);}
PETSC_EXTERN PetscErrorCode EPSErrorView(EPS,EPSErrorType,PetscViewer);
PETSC_DEPRECATED("Use EPSErrorView()") PETSC_STATIC_INLINE PetscErrorCode EPSPrintSolution(EPS eps,PetscViewer v) {return EPSErrorView(eps,EPS_ERROR_RELATIVE,v);}
PETSC_EXTERN PetscErrorCode EPSErrorViewFromOptions(EPS);
PETSC_EXTERN PetscErrorCode EPSReasonView(EPS,PetscViewer);
PETSC_EXTERN PetscErrorCode EPSReasonViewFromOptions(EPS);
PETSC_EXTERN PetscErrorCode EPSValuesView(EPS,PetscViewer);
PETSC_EXTERN PetscErrorCode EPSValuesViewFromOptions(EPS);
PETSC_EXTERN PetscErrorCode EPSVectorsView(EPS,PetscViewer);
PETSC_EXTERN PetscErrorCode EPSVectorsViewFromOptions(EPS);

PETSC_EXTERN PetscErrorCode EPSSetTarget(EPS,PetscScalar);
PETSC_EXTERN PetscErrorCode EPSGetTarget(EPS,PetscScalar*);
PETSC_EXTERN PetscErrorCode EPSSetInterval(EPS,PetscReal,PetscReal);
PETSC_EXTERN PetscErrorCode EPSGetInterval(EPS,PetscReal*,PetscReal*);
PETSC_EXTERN PetscErrorCode EPSSetST(EPS,ST);
PETSC_EXTERN PetscErrorCode EPSGetST(EPS,ST*);
PETSC_EXTERN PetscErrorCode EPSSetBV(EPS,BV);
PETSC_EXTERN PetscErrorCode EPSGetBV(EPS,BV*);
PETSC_EXTERN PetscErrorCode EPSSetRG(EPS,RG);
PETSC_EXTERN PetscErrorCode EPSGetRG(EPS,RG*);
PETSC_EXTERN PetscErrorCode EPSSetDS(EPS,DS);
PETSC_EXTERN PetscErrorCode EPSGetDS(EPS,DS*);
PETSC_EXTERN PetscErrorCode EPSSetTolerances(EPS,PetscReal,PetscInt);
PETSC_EXTERN PetscErrorCode EPSGetTolerances(EPS,PetscReal*,PetscInt*);
PETSC_EXTERN PetscErrorCode EPSSetConvergenceTestFunction(EPS,PetscErrorCode (*)(EPS,PetscScalar,PetscScalar,PetscReal,PetscReal*,void*),void*,PetscErrorCode (*)(void*));
PETSC_EXTERN PetscErrorCode EPSSetConvergenceTest(EPS,EPSConv);
PETSC_EXTERN PetscErrorCode EPSGetConvergenceTest(EPS,EPSConv*);
PETSC_EXTERN PetscErrorCode EPSConvergedAbsolute(EPS,PetscScalar,PetscScalar,PetscReal,PetscReal*,void*);
PETSC_EXTERN PetscErrorCode EPSConvergedRelative(EPS,PetscScalar,PetscScalar,PetscReal,PetscReal*,void*);
PETSC_EXTERN PetscErrorCode EPSConvergedNorm(EPS,PetscScalar,PetscScalar,PetscReal,PetscReal*,void*);
PETSC_EXTERN PetscErrorCode EPSSetStoppingTestFunction(EPS,PetscErrorCode (*)(EPS,PetscInt,PetscInt,PetscInt,PetscInt,EPSConvergedReason*,void*),void*,PetscErrorCode (*)(void*));
PETSC_EXTERN PetscErrorCode EPSSetStoppingTest(EPS,EPSStop);
PETSC_EXTERN PetscErrorCode EPSGetStoppingTest(EPS,EPSStop*);
PETSC_EXTERN PetscErrorCode EPSStoppingBasic(EPS,PetscInt,PetscInt,PetscInt,PetscInt,EPSConvergedReason*,void*);
PETSC_EXTERN PetscErrorCode EPSGetConvergedReason(EPS,EPSConvergedReason*);

PETSC_EXTERN PetscErrorCode EPSSetDimensions(EPS,PetscInt,PetscInt,PetscInt);
PETSC_EXTERN PetscErrorCode EPSGetDimensions(EPS,PetscInt*,PetscInt*,PetscInt*);

PETSC_EXTERN PetscErrorCode EPSGetConverged(EPS,PetscInt*);
PETSC_EXTERN PetscErrorCode EPSGetEigenpair(EPS,PetscInt,PetscScalar*,PetscScalar*,Vec,Vec);
PETSC_EXTERN PetscErrorCode EPSGetEigenvalue(EPS,PetscInt,PetscScalar*,PetscScalar*);
PETSC_EXTERN PetscErrorCode EPSGetEigenvector(EPS,PetscInt,Vec,Vec);

PETSC_EXTERN PetscErrorCode EPSComputeError(EPS,PetscInt,EPSErrorType,PetscReal*);
PETSC_DEPRECATED("Use EPSComputeError()") PETSC_STATIC_INLINE PetscErrorCode EPSComputeRelativeError(EPS eps,PetscInt i,PetscReal *r) {return EPSComputeError(eps,i,EPS_ERROR_RELATIVE,r);}
PETSC_DEPRECATED("Use EPSComputeError() with EPS_ERROR_ABSOLUTE") PETSC_STATIC_INLINE PetscErrorCode EPSComputeResidualNorm(EPS eps,PetscInt i,PetscReal *r) {return EPSComputeError(eps,i,EPS_ERROR_ABSOLUTE,r);}
PETSC_EXTERN PetscErrorCode EPSGetInvariantSubspace(EPS,Vec*);
PETSC_EXTERN PetscErrorCode EPSGetErrorEstimate(EPS,PetscInt,PetscReal*);

PETSC_EXTERN PetscErrorCode EPSMonitor(EPS,PetscInt,PetscInt,PetscScalar*,PetscScalar*,PetscReal*,PetscInt);
PETSC_EXTERN PetscErrorCode EPSMonitorSet(EPS,PetscErrorCode (*)(EPS,PetscInt,PetscInt,PetscScalar*,PetscScalar*,PetscReal*,PetscInt,void*),void*,PetscErrorCode (*)(void**));
PETSC_EXTERN PetscErrorCode EPSMonitorSetFromOptions(EPS,const char*,const char*,const char*,PetscErrorCode (*)(EPS,PetscInt,PetscInt,PetscScalar*,PetscScalar*,PetscReal*,PetscInt,PetscViewerAndFormat*),PetscBool);
PETSC_EXTERN PetscErrorCode EPSConvMonitorSetFromOptions(EPS,const char*,const char*,const char*,PetscErrorCode (*)(EPS,PetscInt,PetscInt,PetscScalar*,PetscScalar*,PetscReal*,PetscInt,SlepcConvMonitor));
PETSC_EXTERN PetscErrorCode EPSMonitorCancel(EPS);
PETSC_EXTERN PetscErrorCode EPSGetMonitorContext(EPS,void**);
PETSC_EXTERN PetscErrorCode EPSGetIterationNumber(EPS,PetscInt*);

PETSC_EXTERN PetscErrorCode EPSSetWhichEigenpairs(EPS,EPSWhich);
PETSC_EXTERN PetscErrorCode EPSGetWhichEigenpairs(EPS,EPSWhich*);
PETSC_EXTERN PetscErrorCode EPSSetTrueResidual(EPS,PetscBool);
PETSC_EXTERN PetscErrorCode EPSGetTrueResidual(EPS,PetscBool*);
PETSC_EXTERN PetscErrorCode EPSSetPurify(EPS,PetscBool);
PETSC_EXTERN PetscErrorCode EPSGetPurify(EPS,PetscBool*);
PETSC_EXTERN PetscErrorCode EPSSetEigenvalueComparison(EPS,PetscErrorCode (*func)(PetscScalar,PetscScalar,PetscScalar,PetscScalar,PetscInt*,void*),void*);
PETSC_EXTERN PetscErrorCode EPSSetArbitrarySelection(EPS,PetscErrorCode (*func)(PetscScalar,PetscScalar,Vec,Vec,PetscScalar*,PetscScalar*,void*),void*);
PETSC_EXTERN PetscErrorCode EPSIsGeneralized(EPS,PetscBool*);
PETSC_EXTERN PetscErrorCode EPSIsHermitian(EPS,PetscBool*);
PETSC_EXTERN PetscErrorCode EPSIsPositive(EPS,PetscBool*);

PETSC_EXTERN PetscErrorCode EPSMonitorFirst(EPS,PetscInt,PetscInt,PetscScalar*,PetscScalar*,PetscReal*,PetscInt,PetscViewerAndFormat*);
PETSC_EXTERN PetscErrorCode EPSMonitorAll(EPS,PetscInt,PetscInt,PetscScalar*,PetscScalar*,PetscReal*,PetscInt,PetscViewerAndFormat*);
PETSC_EXTERN PetscErrorCode EPSMonitorConverged(EPS,PetscInt,PetscInt,PetscScalar*,PetscScalar*,PetscReal*,PetscInt,SlepcConvMonitor);
PETSC_EXTERN PetscErrorCode EPSMonitorLGCreate(MPI_Comm,const char[],const char[],int,int,int,int,PetscDrawLG*);
PETSC_EXTERN PetscErrorCode EPSMonitorLG(EPS,PetscInt,PetscInt,PetscScalar*,PetscScalar*,PetscReal*,PetscInt,void*);
PETSC_EXTERN PetscErrorCode EPSMonitorLGAll(EPS,PetscInt,PetscInt,PetscScalar*,PetscScalar*,PetscReal*,PetscInt,void*);

PETSC_EXTERN PetscErrorCode EPSSetTrackAll(EPS,PetscBool);
PETSC_EXTERN PetscErrorCode EPSGetTrackAll(EPS,PetscBool*);

PETSC_EXTERN PetscErrorCode EPSSetDeflationSpace(EPS,PetscInt,Vec*);
PETSC_EXTERN PetscErrorCode EPSSetInitialSpace(EPS,PetscInt,Vec*);

PETSC_EXTERN PetscErrorCode EPSSetOptionsPrefix(EPS,const char*);
PETSC_EXTERN PetscErrorCode EPSAppendOptionsPrefix(EPS,const char*);
PETSC_EXTERN PetscErrorCode EPSGetOptionsPrefix(EPS,const char*[]);

PETSC_EXTERN PetscFunctionList EPSList;
PETSC_EXTERN PetscErrorCode EPSRegister(const char[],PetscErrorCode(*)(EPS));

PETSC_EXTERN PetscErrorCode EPSSetWorkVecs(EPS,PetscInt);
PETSC_EXTERN PetscErrorCode EPSAllocateSolution(EPS,PetscInt);

/* --------- options specific to particular eigensolvers -------- */

/*E
    EPSPowerShiftType - determines the type of shift used in the Power iteration

    Level: advanced

.seealso: EPSPowerSetShiftType(), EPSPowerGetShiftType()
E*/
typedef enum { EPS_POWER_SHIFT_CONSTANT,
               EPS_POWER_SHIFT_RAYLEIGH,
               EPS_POWER_SHIFT_WILKINSON } EPSPowerShiftType;
PETSC_EXTERN const char *EPSPowerShiftTypes[];

PETSC_EXTERN PetscErrorCode EPSPowerSetShiftType(EPS,EPSPowerShiftType);
PETSC_EXTERN PetscErrorCode EPSPowerGetShiftType(EPS,EPSPowerShiftType*);

PETSC_EXTERN PetscErrorCode EPSArnoldiSetDelayed(EPS,PetscBool);
PETSC_EXTERN PetscErrorCode EPSArnoldiGetDelayed(EPS,PetscBool*);

PETSC_EXTERN PetscErrorCode EPSKrylovSchurSetRestart(EPS,PetscReal);
PETSC_EXTERN PetscErrorCode EPSKrylovSchurGetRestart(EPS,PetscReal*);
PETSC_EXTERN PetscErrorCode EPSKrylovSchurSetLocking(EPS,PetscBool);
PETSC_EXTERN PetscErrorCode EPSKrylovSchurGetLocking(EPS,PetscBool*);
PETSC_EXTERN PetscErrorCode EPSKrylovSchurSetPartitions(EPS,PetscInt);
PETSC_EXTERN PetscErrorCode EPSKrylovSchurGetPartitions(EPS,PetscInt*);
PETSC_EXTERN PetscErrorCode EPSKrylovSchurSetDetectZeros(EPS,PetscBool);
PETSC_EXTERN PetscErrorCode EPSKrylovSchurGetDetectZeros(EPS,PetscBool*);
PETSC_EXTERN PetscErrorCode EPSKrylovSchurSetDimensions(EPS,PetscInt,PetscInt,PetscInt);
PETSC_EXTERN PetscErrorCode EPSKrylovSchurGetDimensions(EPS,PetscInt*,PetscInt*,PetscInt*);
PETSC_EXTERN PetscErrorCode EPSKrylovSchurSetSubintervals(EPS,PetscReal*);
PETSC_EXTERN PetscErrorCode EPSKrylovSchurGetSubintervals(EPS,PetscReal**);
PETSC_EXTERN PetscErrorCode EPSKrylovSchurGetInertias(EPS,PetscInt*,PetscReal**,PetscInt**);
PETSC_EXTERN PetscErrorCode EPSKrylovSchurGetSubcommInfo(EPS,PetscInt*,PetscInt*,Vec*);
PETSC_EXTERN PetscErrorCode EPSKrylovSchurGetSubcommPairs(EPS,PetscInt,PetscScalar*,Vec);
PETSC_EXTERN PetscErrorCode EPSKrylovSchurGetSubcommMats(EPS,Mat*,Mat*);
PETSC_EXTERN PetscErrorCode EPSKrylovSchurUpdateSubcommMats(EPS,PetscScalar,PetscScalar,Mat,PetscScalar,PetscScalar, Mat,MatStructure,PetscBool);

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
PETSC_EXTERN const char *EPSLanczosReorthogTypes[];

PETSC_EXTERN PetscErrorCode EPSLanczosSetReorthog(EPS,EPSLanczosReorthogType);
PETSC_EXTERN PetscErrorCode EPSLanczosGetReorthog(EPS,EPSLanczosReorthogType*);

PETSC_EXTERN PetscErrorCode EPSBlzpackSetBlockSize(EPS,PetscInt);
PETSC_EXTERN PetscErrorCode EPSBlzpackSetNSteps(EPS,PetscInt);

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
PETSC_EXTERN const char *EPSPRIMMEMethods[];

PETSC_EXTERN PetscErrorCode EPSPRIMMESetBlockSize(EPS,PetscInt);
PETSC_EXTERN PetscErrorCode EPSPRIMMESetMethod(EPS,EPSPRIMMEMethod);
PETSC_EXTERN PetscErrorCode EPSPRIMMEGetBlockSize(EPS,PetscInt*);
PETSC_EXTERN PetscErrorCode EPSPRIMMEGetMethod(EPS,EPSPRIMMEMethod*);

PETSC_EXTERN PetscErrorCode EPSGDSetKrylovStart(EPS,PetscBool);
PETSC_EXTERN PetscErrorCode EPSGDGetKrylovStart(EPS,PetscBool*);
PETSC_EXTERN PetscErrorCode EPSGDSetBlockSize(EPS,PetscInt);
PETSC_EXTERN PetscErrorCode EPSGDGetBlockSize(EPS,PetscInt*);
PETSC_EXTERN PetscErrorCode EPSGDSetRestart(EPS,PetscInt,PetscInt);
PETSC_EXTERN PetscErrorCode EPSGDGetRestart(EPS,PetscInt*,PetscInt*);
PETSC_EXTERN PetscErrorCode EPSGDSetInitialSize(EPS,PetscInt);
PETSC_EXTERN PetscErrorCode EPSGDGetInitialSize(EPS,PetscInt*);
PETSC_EXTERN PetscErrorCode EPSGDSetBOrth(EPS,PetscBool);
PETSC_EXTERN PetscErrorCode EPSGDGetBOrth(EPS,PetscBool*);
PETSC_EXTERN PetscErrorCode EPSGDGetWindowSizes(EPS,PetscInt*,PetscInt*);
PETSC_EXTERN PetscErrorCode EPSGDSetWindowSizes(EPS,PetscInt,PetscInt);
PETSC_EXTERN PetscErrorCode EPSGDSetDoubleExpansion(EPS,PetscBool);
PETSC_EXTERN PetscErrorCode EPSGDGetDoubleExpansion(EPS,PetscBool*);

PETSC_EXTERN PetscErrorCode EPSJDSetKrylovStart(EPS,PetscBool);
PETSC_EXTERN PetscErrorCode EPSJDGetKrylovStart(EPS,PetscBool*);
PETSC_EXTERN PetscErrorCode EPSJDSetBlockSize(EPS,PetscInt);
PETSC_EXTERN PetscErrorCode EPSJDGetBlockSize(EPS,PetscInt*);
PETSC_EXTERN PetscErrorCode EPSJDSetRestart(EPS,PetscInt,PetscInt);
PETSC_EXTERN PetscErrorCode EPSJDGetRestart(EPS,PetscInt*,PetscInt*);
PETSC_EXTERN PetscErrorCode EPSJDSetInitialSize(EPS,PetscInt);
PETSC_EXTERN PetscErrorCode EPSJDGetInitialSize(EPS,PetscInt*);
PETSC_EXTERN PetscErrorCode EPSJDSetFix(EPS,PetscReal);
PETSC_EXTERN PetscErrorCode EPSJDGetFix(EPS,PetscReal*);
PETSC_EXTERN PetscErrorCode EPSJDSetConstCorrectionTol(EPS,PetscBool);
PETSC_EXTERN PetscErrorCode EPSJDGetConstCorrectionTol(EPS,PetscBool*);
PETSC_EXTERN PetscErrorCode EPSJDSetBOrth(EPS,PetscBool);
PETSC_EXTERN PetscErrorCode EPSJDGetBOrth(EPS,PetscBool*);
PETSC_EXTERN PetscErrorCode EPSJDGetWindowSizes(EPS,PetscInt*,PetscInt*);
PETSC_EXTERN PetscErrorCode EPSJDSetWindowSizes(EPS,PetscInt,PetscInt);

PETSC_EXTERN PetscErrorCode EPSRQCGSetReset(EPS,PetscInt);
PETSC_EXTERN PetscErrorCode EPSRQCGGetReset(EPS,PetscInt*);

PETSC_EXTERN PetscErrorCode EPSLOBPCGSetBlockSize(EPS,PetscInt);
PETSC_EXTERN PetscErrorCode EPSLOBPCGGetBlockSize(EPS,PetscInt*);
PETSC_EXTERN PetscErrorCode EPSLOBPCGSetRestart(EPS,PetscReal);
PETSC_EXTERN PetscErrorCode EPSLOBPCGGetRestart(EPS,PetscReal*);
PETSC_EXTERN PetscErrorCode EPSLOBPCGSetLocking(EPS,PetscBool);
PETSC_EXTERN PetscErrorCode EPSLOBPCGGetLocking(EPS,PetscBool*);

/*E
    EPSCISSQuadRule - determines the quadrature rule in the CISS solver

    Level: advanced

.seealso: EPSCISSSetQuadRule(), EPSCISSGetQuadRule()
E*/
typedef enum { EPS_CISS_QUADRULE_TRAPEZOIDAL=1,
               EPS_CISS_QUADRULE_CHEBYSHEV } EPSCISSQuadRule;
PETSC_EXTERN const char *EPSCISSQuadRules[];

/*E
    EPSCISSExtraction - determines the extraction technique in the CISS solver

    Level: advanced

.seealso: EPSCISSSetExtraction(), EPSCISSGetExtraction()
E*/
typedef enum { EPS_CISS_EXTRACTION_RITZ,
               EPS_CISS_EXTRACTION_HANKEL } EPSCISSExtraction;
PETSC_EXTERN const char *EPSCISSExtractions[];

PETSC_EXTERN PetscErrorCode EPSCISSSetExtraction(EPS,EPSCISSExtraction);
PETSC_EXTERN PetscErrorCode EPSCISSGetExtraction(EPS,EPSCISSExtraction*);
PETSC_EXTERN PetscErrorCode EPSCISSSetQuadRule(EPS,EPSCISSQuadRule);
PETSC_EXTERN PetscErrorCode EPSCISSGetQuadRule(EPS,EPSCISSQuadRule*);
PETSC_EXTERN PetscErrorCode EPSCISSSetRegion(EPS,PetscScalar,PetscReal,PetscReal);
PETSC_EXTERN PetscErrorCode EPSCISSGetRegion(EPS,PetscScalar*,PetscReal*,PetscReal*);
PETSC_EXTERN PetscErrorCode EPSCISSSetSizes(EPS,PetscInt,PetscInt,PetscInt,PetscInt,PetscInt,PetscBool);
PETSC_EXTERN PetscErrorCode EPSCISSGetSizes(EPS,PetscInt*,PetscInt*,PetscInt*,PetscInt*,PetscInt*,PetscBool*);
PETSC_EXTERN PetscErrorCode EPSCISSSetThreshold(EPS,PetscReal,PetscReal);
PETSC_EXTERN PetscErrorCode EPSCISSGetThreshold(EPS,PetscReal*,PetscReal*);
PETSC_EXTERN PetscErrorCode EPSCISSSetRefinement(EPS,PetscInt,PetscInt);
PETSC_EXTERN PetscErrorCode EPSCISSGetRefinement(EPS,PetscInt*,PetscInt*);
PETSC_EXTERN PetscErrorCode EPSCISSSetUseST(EPS,PetscBool);
PETSC_EXTERN PetscErrorCode EPSCISSGetUseST(EPS,PetscBool*);

PETSC_EXTERN PetscErrorCode EPSBLOPEXSetBlockSize(EPS,PetscInt);
PETSC_EXTERN PetscErrorCode EPSBLOPEXGetBlockSize(EPS,PetscInt*);

PETSC_EXTERN PetscErrorCode EPSFEASTSetNumPoints(EPS,PetscInt);
PETSC_EXTERN PetscErrorCode EPSFEASTGetNumPoints(EPS,PetscInt*);

#endif

