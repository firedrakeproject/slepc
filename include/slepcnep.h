/*
   User interface for SLEPc's nonlinear eigenvalue solvers.

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

#if !defined(__SLEPCNEP_H)
#define __SLEPCNEP_H
#include <slepceps.h>
#include <slepcpep.h>
#include <slepcfn.h>

PETSC_EXTERN PetscErrorCode NEPInitializePackage(void);

/*S
     NEP - Abstract SLEPc object that manages all solvers for
     nonlinear eigenvalue problems.

   Level: beginner

.seealso:  NEPCreate()
S*/
typedef struct _p_NEP* NEP;

/*J
    NEPType - String with the name of a nonlinear eigensolver

   Level: beginner

.seealso: NEPSetType(), NEP
J*/
typedef const char* NEPType;
#define NEPRII       "rii"
#define NEPSLP       "slp"
#define NEPNARNOLDI  "narnoldi"
#define NEPCISS      "ciss"
#define NEPINTERPOL  "interpol"
#define NEPNLEIGS    "nleigs"

/* Logging support */
PETSC_EXTERN PetscClassId NEP_CLASSID;

/*E
    NEPWhich - Determines which part of the spectrum is requested

    Level: intermediate

.seealso: NEPSetWhichEigenpairs(), NEPGetWhichEigenpairs()
E*/
typedef enum { NEP_LARGEST_MAGNITUDE=1,
               NEP_SMALLEST_MAGNITUDE,
               NEP_LARGEST_REAL,
               NEP_SMALLEST_REAL,
               NEP_LARGEST_IMAGINARY,
               NEP_SMALLEST_IMAGINARY,
               NEP_TARGET_MAGNITUDE,
               NEP_TARGET_REAL,
               NEP_TARGET_IMAGINARY,
               NEP_ALL,
               NEP_WHICH_USER } NEPWhich;

/*E
    NEPErrorType - The error type used to assess accuracy of computed solutions

    Level: intermediate

.seealso: NEPComputeError()
E*/
typedef enum { NEP_ERROR_ABSOLUTE,
               NEP_ERROR_RELATIVE,
               NEP_ERROR_BACKWARD } NEPErrorType;
PETSC_EXTERN const char *NEPErrorTypes[];

/*E
    NEPRefine - The refinement type

    Level: intermediate

.seealso: NEPSetRefine()
E*/
typedef enum { NEP_REFINE_NONE,
               NEP_REFINE_SIMPLE,
               NEP_REFINE_MULTIPLE } NEPRefine;
PETSC_EXTERN const char *NEPRefineTypes[];

/*E
    NEPRefineScheme - The scheme used for solving linear systems during iterative refinement

    Level: intermediate

.seealso: NEPSetRefine()
E*/
typedef enum { NEP_REFINE_SCHEME_SCHUR=1,
               NEP_REFINE_SCHEME_MBE,
               NEP_REFINE_SCHEME_EXPLICIT } NEPRefineScheme;
PETSC_EXTERN const char *NEPRefineSchemes[];

/*E
    NEPConv - Determines the convergence test

    Level: intermediate

.seealso: NEPSetConvergenceTest(), NEPSetConvergenceTestFunction()
E*/
typedef enum { NEP_CONV_ABS,
               NEP_CONV_REL,
               NEP_CONV_NORM,
               NEP_CONV_USER } NEPConv;

/*E
    NEPStop - Determines the stopping test

    Level: advanced

.seealso: NEPSetStoppingTest(), NEPSetStoppingTestFunction()
E*/
typedef enum { NEP_STOP_BASIC,
               NEP_STOP_USER } NEPStop;

/*E
    NEPConvergedReason - Reason a nonlinear eigensolver was said to
         have converged or diverged

    Level: intermediate

.seealso: NEPSolve(), NEPGetConvergedReason(), NEPSetTolerances()
E*/
typedef enum {/* converged */
              NEP_CONVERGED_TOL                =  1,
              NEP_CONVERGED_USER               =  2,
              /* diverged */
              NEP_DIVERGED_ITS                 = -1,
              NEP_DIVERGED_BREAKDOWN           = -2,
                    /* unused                  = -3 */
              NEP_DIVERGED_LINEAR_SOLVE        = -4,
              NEP_CONVERGED_ITERATING          =  0} NEPConvergedReason;
PETSC_EXTERN const char *const*NEPConvergedReasons;

PETSC_EXTERN PetscErrorCode NEPCreate(MPI_Comm,NEP*);
PETSC_EXTERN PetscErrorCode NEPDestroy(NEP*);
PETSC_EXTERN PetscErrorCode NEPReset(NEP);
PETSC_EXTERN PetscErrorCode NEPSetType(NEP,NEPType);
PETSC_EXTERN PetscErrorCode NEPGetType(NEP,NEPType*);
PETSC_EXTERN PetscErrorCode NEPSetTarget(NEP,PetscScalar);
PETSC_EXTERN PetscErrorCode NEPGetTarget(NEP,PetscScalar*);
PETSC_EXTERN PetscErrorCode NEPSetFromOptions(NEP);
PETSC_EXTERN PetscErrorCode NEPSetUp(NEP);
PETSC_EXTERN PetscErrorCode NEPSolve(NEP);
PETSC_EXTERN PetscErrorCode NEPView(NEP,PetscViewer);
PETSC_STATIC_INLINE PetscErrorCode NEPViewFromOptions(NEP nep,PetscObject obj,const char name[]) {return PetscObjectViewFromOptions((PetscObject)nep,obj,name);}
PETSC_EXTERN PetscErrorCode NEPErrorView(NEP,NEPErrorType,PetscViewer);
PETSC_EXTERN PetscErrorCode NEPErrorViewFromOptions(NEP);
PETSC_EXTERN PetscErrorCode NEPReasonView(NEP,PetscViewer);
PETSC_EXTERN PetscErrorCode NEPReasonViewFromOptions(NEP);
PETSC_EXTERN PetscErrorCode NEPValuesView(NEP,PetscViewer);
PETSC_EXTERN PetscErrorCode NEPValuesViewFromOptions(NEP);
PETSC_EXTERN PetscErrorCode NEPVectorsView(NEP,PetscViewer);
PETSC_EXTERN PetscErrorCode NEPVectorsViewFromOptions(NEP);

PETSC_EXTERN PetscErrorCode NEPSetFunction(NEP,Mat,Mat,PetscErrorCode (*)(NEP,PetscScalar,Mat,Mat,void*),void*);
PETSC_EXTERN PetscErrorCode NEPGetFunction(NEP,Mat*,Mat*,PetscErrorCode (**)(NEP,PetscScalar,Mat,Mat,void*),void**);
PETSC_EXTERN PetscErrorCode NEPSetJacobian(NEP,Mat,PetscErrorCode (*)(NEP,PetscScalar,Mat,void*),void*);
PETSC_EXTERN PetscErrorCode NEPGetJacobian(NEP,Mat*,PetscErrorCode (**)(NEP,PetscScalar,Mat,void*),void**);
PETSC_EXTERN PetscErrorCode NEPSetDerivatives(NEP,Mat,PetscErrorCode (*)(NEP,PetscScalar,PetscInt,Mat,void*),void*);
PETSC_EXTERN PetscErrorCode NEPGetDerivatives(NEP,Mat*,PetscErrorCode (**)(NEP,PetscScalar,PetscInt,Mat,void*),void**);
PETSC_EXTERN PetscErrorCode NEPSetSplitOperator(NEP,PetscInt,Mat*,FN*,MatStructure);
PETSC_EXTERN PetscErrorCode NEPGetSplitOperatorTerm(NEP,PetscInt,Mat*,FN*);
PETSC_EXTERN PetscErrorCode NEPGetSplitOperatorInfo(NEP,PetscInt*,MatStructure*);

PETSC_EXTERN PetscErrorCode NEPSetBV(NEP,BV);
PETSC_EXTERN PetscErrorCode NEPGetBV(NEP,BV*);
PETSC_EXTERN PetscErrorCode NEPSetRG(NEP,RG);
PETSC_EXTERN PetscErrorCode NEPGetRG(NEP,RG*);
PETSC_EXTERN PetscErrorCode NEPSetDS(NEP,DS);
PETSC_EXTERN PetscErrorCode NEPGetDS(NEP,DS*);
PETSC_EXTERN PetscErrorCode NEPRefineGetKSP(NEP,KSP*);
PETSC_EXTERN PetscErrorCode NEPSetTolerances(NEP,PetscReal,PetscInt);
PETSC_EXTERN PetscErrorCode NEPGetTolerances(NEP,PetscReal*,PetscInt*);
PETSC_EXTERN PetscErrorCode NEPSetConvergenceTestFunction(NEP,PetscErrorCode (*)(NEP,PetscScalar,PetscScalar,PetscReal,PetscReal*,void*),void*,PetscErrorCode (*)(void*));
PETSC_EXTERN PetscErrorCode NEPSetConvergenceTest(NEP,NEPConv);
PETSC_EXTERN PetscErrorCode NEPGetConvergenceTest(NEP,NEPConv*);
PETSC_EXTERN PetscErrorCode NEPConvergedAbsolute(NEP,PetscScalar,PetscScalar,PetscReal,PetscReal*,void*);
PETSC_EXTERN PetscErrorCode NEPConvergedRelative(NEP,PetscScalar,PetscScalar,PetscReal,PetscReal*,void*);
PETSC_EXTERN PetscErrorCode NEPConvergedNorm(NEP,PetscScalar,PetscScalar,PetscReal,PetscReal*,void*);
PETSC_EXTERN PetscErrorCode NEPSetStoppingTestFunction(NEP,PetscErrorCode (*)(NEP,PetscInt,PetscInt,PetscInt,PetscInt,NEPConvergedReason*,void*),void*,PetscErrorCode (*)(void*));
PETSC_EXTERN PetscErrorCode NEPSetStoppingTest(NEP,NEPStop);
PETSC_EXTERN PetscErrorCode NEPGetStoppingTest(NEP,NEPStop*);
PETSC_EXTERN PetscErrorCode NEPStoppingBasic(NEP,PetscInt,PetscInt,PetscInt,PetscInt,NEPConvergedReason*,void*);
PETSC_EXTERN PetscErrorCode NEPSetDimensions(NEP,PetscInt,PetscInt,PetscInt);
PETSC_EXTERN PetscErrorCode NEPGetDimensions(NEP,PetscInt*,PetscInt*,PetscInt*);
PETSC_EXTERN PetscErrorCode NEPSetRefine(NEP,NEPRefine,PetscInt,PetscReal,PetscInt,NEPRefineScheme);
PETSC_EXTERN PetscErrorCode NEPGetRefine(NEP,NEPRefine*,PetscInt*,PetscReal*,PetscInt*,NEPRefineScheme*);

PETSC_EXTERN PetscErrorCode NEPGetConverged(NEP,PetscInt*);
PETSC_EXTERN PetscErrorCode NEPGetEigenpair(NEP,PetscInt,PetscScalar*,PetscScalar*,Vec,Vec);

PETSC_EXTERN PetscErrorCode NEPComputeError(NEP,PetscInt,NEPErrorType,PetscReal*);
PETSC_DEPRECATED("Use NEPComputeError()") PETSC_STATIC_INLINE PetscErrorCode NEPComputeRelativeError(NEP nep,PetscInt i,PetscReal *r) {return NEPComputeError(nep,i,NEP_ERROR_RELATIVE,r);}
PETSC_DEPRECATED("Use NEPComputeError() with NEP_ERROR_ABSOLUTE") PETSC_STATIC_INLINE PetscErrorCode NEPComputeResidualNorm(NEP nep,PetscInt i,PetscReal *r) {return NEPComputeError(nep,i,NEP_ERROR_ABSOLUTE,r);}
PETSC_EXTERN PetscErrorCode NEPGetErrorEstimate(NEP,PetscInt,PetscReal*);

PETSC_EXTERN PetscErrorCode NEPComputeFunction(NEP,PetscScalar,Mat,Mat);
PETSC_EXTERN PetscErrorCode NEPComputeJacobian(NEP,PetscScalar,Mat);
PETSC_EXTERN PetscErrorCode NEPApplyFunction(NEP,PetscScalar,Vec,Vec,Vec,Mat,Mat);
PETSC_EXTERN PetscErrorCode NEPApplyJacobian(NEP,PetscScalar,Vec,Vec,Vec,Mat);
PETSC_EXTERN PetscErrorCode NEPProjectOperator(NEP,PetscInt,PetscInt);

PETSC_EXTERN PetscErrorCode NEPMonitor(NEP,PetscInt,PetscInt,PetscScalar*,PetscScalar*,PetscReal*,PetscInt);
PETSC_EXTERN PetscErrorCode NEPMonitorSet(NEP,PetscErrorCode (*)(NEP,PetscInt,PetscInt,PetscScalar*,PetscScalar*,PetscReal*,PetscInt,void*),void*,PetscErrorCode (*)(void**));
PETSC_EXTERN PetscErrorCode NEPMonitorSetFromOptions(NEP,const char*,const char*,const char*,PetscErrorCode (*)(NEP,PetscInt,PetscInt,PetscScalar*,PetscScalar*,PetscReal*,PetscInt,PetscViewerAndFormat*),PetscBool);
PETSC_EXTERN PetscErrorCode NEPConvMonitorSetFromOptions(NEP,const char*,const char*,const char*,PetscErrorCode (*)(NEP,PetscInt,PetscInt,PetscScalar*,PetscScalar*,PetscReal*,PetscInt,SlepcConvMonitor));
PETSC_EXTERN PetscErrorCode NEPMonitorCancel(NEP);
PETSC_EXTERN PetscErrorCode NEPGetMonitorContext(NEP,void **);
PETSC_EXTERN PetscErrorCode NEPGetIterationNumber(NEP,PetscInt*);

PETSC_EXTERN PetscErrorCode NEPSetInitialSpace(NEP,PetscInt,Vec*);
PETSC_EXTERN PetscErrorCode NEPSetWhichEigenpairs(NEP,NEPWhich);
PETSC_EXTERN PetscErrorCode NEPGetWhichEigenpairs(NEP,NEPWhich*);
PETSC_EXTERN PetscErrorCode NEPSetEigenvalueComparison(NEP,PetscErrorCode (*func)(PetscScalar,PetscScalar,PetscScalar,PetscScalar,PetscInt*,void*),void*);

PETSC_EXTERN PetscErrorCode NEPMonitorAll(NEP,PetscInt,PetscInt,PetscScalar*,PetscScalar*,PetscReal*,PetscInt,PetscViewerAndFormat*);
PETSC_EXTERN PetscErrorCode NEPMonitorFirst(NEP,PetscInt,PetscInt,PetscScalar*,PetscScalar*,PetscReal*,PetscInt,PetscViewerAndFormat*);
PETSC_EXTERN PetscErrorCode NEPMonitorConverged(NEP,PetscInt,PetscInt,PetscScalar*,PetscScalar*,PetscReal*,PetscInt,SlepcConvMonitor);
PETSC_EXTERN PetscErrorCode NEPMonitorLGCreate(MPI_Comm,const char[],const char[],int,int,int,int,PetscDrawLG*);
PETSC_EXTERN PetscErrorCode NEPMonitorLG(NEP,PetscInt,PetscInt,PetscScalar*,PetscScalar*,PetscReal*,PetscInt,void*);
PETSC_EXTERN PetscErrorCode NEPMonitorLGAll(NEP,PetscInt,PetscInt,PetscScalar*,PetscScalar*,PetscReal*,PetscInt,void*);

PETSC_EXTERN PetscErrorCode NEPSetTrackAll(NEP,PetscBool);
PETSC_EXTERN PetscErrorCode NEPGetTrackAll(NEP,PetscBool*);

PETSC_EXTERN PetscErrorCode NEPSetOptionsPrefix(NEP,const char*);
PETSC_EXTERN PetscErrorCode NEPAppendOptionsPrefix(NEP,const char*);
PETSC_EXTERN PetscErrorCode NEPGetOptionsPrefix(NEP,const char*[]);

PETSC_EXTERN PetscErrorCode NEPGetConvergedReason(NEP,NEPConvergedReason *);

PETSC_EXTERN PetscFunctionList NEPList;
PETSC_EXTERN PetscErrorCode NEPRegister(const char[],PetscErrorCode(*)(NEP));

PETSC_EXTERN PetscErrorCode NEPSetWorkVecs(NEP,PetscInt);
PETSC_EXTERN PetscErrorCode NEPAllocateSolution(NEP,PetscInt);

/* --------- options specific to particular eigensolvers -------- */

PETSC_EXTERN PetscErrorCode NEPRIISetMaximumIterations(NEP,PetscInt);
PETSC_EXTERN PetscErrorCode NEPRIIGetMaximumIterations(NEP,PetscInt*);
PETSC_EXTERN PetscErrorCode NEPRIISetLagPreconditioner(NEP,PetscInt);
PETSC_EXTERN PetscErrorCode NEPRIIGetLagPreconditioner(NEP,PetscInt*);
PETSC_EXTERN PetscErrorCode NEPRIISetConstCorrectionTol(NEP,PetscBool);
PETSC_EXTERN PetscErrorCode NEPRIIGetConstCorrectionTol(NEP,PetscBool*);
PETSC_EXTERN PetscErrorCode NEPRIISetKSP(NEP,KSP);
PETSC_EXTERN PetscErrorCode NEPRIIGetKSP(NEP,KSP*);

PETSC_EXTERN PetscErrorCode NEPSLPSetEPS(NEP,EPS);
PETSC_EXTERN PetscErrorCode NEPSLPGetEPS(NEP,EPS*);

PETSC_EXTERN PetscErrorCode NEPNArnoldiSetKSP(NEP,KSP);
PETSC_EXTERN PetscErrorCode NEPNArnoldiGetKSP(NEP,KSP*);

PETSC_EXTERN PetscErrorCode NEPCISSSetSizes(NEP,PetscInt,PetscInt,PetscInt,PetscInt,PetscInt,PetscBool);
PETSC_EXTERN PetscErrorCode NEPCISSGetSizes(NEP,PetscInt*,PetscInt*,PetscInt*,PetscInt*,PetscInt*,PetscBool*);
PETSC_EXTERN PetscErrorCode NEPCISSSetThreshold(NEP,PetscReal,PetscReal);
PETSC_EXTERN PetscErrorCode NEPCISSGetThreshold(NEP,PetscReal*,PetscReal*);
PETSC_EXTERN PetscErrorCode NEPCISSSetRefinement(NEP,PetscInt,PetscInt);
PETSC_EXTERN PetscErrorCode NEPCISSGetRefinement(NEP,PetscInt*,PetscInt*);

PETSC_EXTERN PetscErrorCode NEPInterpolSetPEP(NEP,PEP);
PETSC_EXTERN PetscErrorCode NEPInterpolGetPEP(NEP,PEP*);
PETSC_EXTERN PetscErrorCode NEPInterpolSetDegree(NEP,PetscInt);
PETSC_EXTERN PetscErrorCode NEPInterpolGetDegree(NEP,PetscInt*);

PETSC_EXTERN PetscErrorCode NEPNLEIGSSetSingularitiesFunction(NEP,PetscErrorCode (*)(NEP,PetscInt*,PetscScalar*,void*),void*);
PETSC_EXTERN PetscErrorCode NEPNLEIGSGetSingularitiesFunction(NEP,PetscErrorCode (**)(NEP,PetscInt*,PetscScalar*,void*),void **);
PETSC_EXTERN PetscErrorCode NEPNLEIGSSetRestart(NEP,PetscReal);
PETSC_EXTERN PetscErrorCode NEPNLEIGSGetRestart(NEP,PetscReal*);
PETSC_EXTERN PetscErrorCode NEPNLEIGSSetLocking(NEP,PetscBool);
PETSC_EXTERN PetscErrorCode NEPNLEIGSGetLocking(NEP,PetscBool*);
PETSC_EXTERN PetscErrorCode NEPNLEIGSSetInterpolation(NEP,PetscReal,PetscInt);
PETSC_EXTERN PetscErrorCode NEPNLEIGSGetInterpolation(NEP,PetscReal*,PetscInt*);
PETSC_EXTERN PetscErrorCode NEPNLEIGSSetTrueResidual(NEP,PetscBool);
PETSC_EXTERN PetscErrorCode NEPNLEIGSGetTrueResidual(NEP,PetscBool*);
PETSC_EXTERN PetscErrorCode NEPNLEIGSSetRKShifts(NEP,PetscInt,PetscScalar*);
PETSC_EXTERN PetscErrorCode NEPNLEIGSGetRKShifts(NEP,PetscInt*,PetscScalar**);
PETSC_EXTERN PetscErrorCode NEPNLEIGSGetKSPs(NEP,KSP**);

#endif

