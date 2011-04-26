/*
   User interface for the SLEPC eigenproblem solvers. 

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

#if !defined(__SLEPCEPS_H)
#define __SLEPCEPS_H
#include "slepcsys.h"
#include "slepcst.h"
#include "slepcip.h"
PETSC_EXTERN_CXX_BEGIN

extern PetscClassId EPS_CLASSID;

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
#define EPSGD        "gd"
#define EPSJD        "jd"
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
               EPS_HARMONIC_RELATIVE,
               EPS_HARMONIC_RIGHT,
               EPS_HARMONIC_LARGEST,
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

/*E
    EPSConv - determines the convergence test

    Level: intermediate

.seealso: EPSSetConvergenceTest(), EPSSetConvergenceTestFunction()
E*/
typedef enum { EPS_CONV_ABS=1,
               EPS_CONV_EIG,
               EPS_CONV_NORM,
               EPS_CONV_USER } EPSConv;

extern PetscErrorCode EPSCreate(MPI_Comm,EPS *);
extern PetscErrorCode EPSDestroy(EPS*);
extern PetscErrorCode EPSSetType(EPS,const EPSType);
extern PetscErrorCode EPSGetType(EPS,const EPSType*);
extern PetscErrorCode EPSSetProblemType(EPS,EPSProblemType);
extern PetscErrorCode EPSGetProblemType(EPS,EPSProblemType*);
extern PetscErrorCode EPSSetExtraction(EPS,EPSExtraction);
extern PetscErrorCode EPSGetExtraction(EPS,EPSExtraction*);
extern PetscErrorCode EPSSetBalance(EPS,EPSBalance,PetscInt,PetscReal);
extern PetscErrorCode EPSGetBalance(EPS,EPSBalance*,PetscInt*,PetscReal*);
extern PetscErrorCode EPSSetOperators(EPS,Mat,Mat);
extern PetscErrorCode EPSGetOperators(EPS,Mat*,Mat*);
extern PetscErrorCode EPSSetFromOptions(EPS);
extern PetscErrorCode EPSSetUp(EPS);
extern PetscErrorCode EPSSolve(EPS);
extern PetscErrorCode EPSView(EPS,PetscViewer);

extern PetscErrorCode EPSSetTarget(EPS,PetscScalar);
extern PetscErrorCode EPSGetTarget(EPS,PetscScalar*);
extern PetscErrorCode EPSSetST(EPS,ST);
extern PetscErrorCode EPSGetST(EPS,ST*);
extern PetscErrorCode EPSSetIP(EPS,IP);
extern PetscErrorCode EPSGetIP(EPS,IP*);
extern PetscErrorCode EPSSetTolerances(EPS,PetscReal,PetscInt);
extern PetscErrorCode EPSGetTolerances(EPS,PetscReal*,PetscInt*);
extern PetscErrorCode EPSSetConvergenceTestFunction(EPS,PetscErrorCode (*)(EPS,PetscScalar,PetscScalar,PetscReal,PetscReal*,void*),void*);
extern PetscErrorCode EPSSetConvergenceTest(EPS eps,EPSConv conv);
extern PetscErrorCode EPSGetConvergenceTest(EPS eps,EPSConv *conv);
extern PetscErrorCode EPSEigRelativeConverged(EPS,PetscScalar,PetscScalar,PetscReal,PetscReal*,void*);
extern PetscErrorCode EPSAbsoluteConverged(EPS,PetscScalar,PetscScalar,PetscReal,PetscReal*,void*);
extern PetscErrorCode EPSNormRelativeConverged(EPS,PetscScalar,PetscScalar,PetscReal,PetscReal*,void*);
extern PetscErrorCode EPSSetDimensions(EPS,PetscInt,PetscInt,PetscInt);
extern PetscErrorCode EPSGetDimensions(EPS,PetscInt*,PetscInt*,PetscInt*);

extern PetscErrorCode EPSGetConverged(EPS,PetscInt*);
extern PetscErrorCode EPSGetEigenpair(EPS,PetscInt,PetscScalar*,PetscScalar*,Vec,Vec);
extern PetscErrorCode EPSGetEigenvalue(EPS,PetscInt,PetscScalar*,PetscScalar*);
extern PetscErrorCode EPSGetEigenvector(EPS,PetscInt,Vec,Vec);
extern PetscErrorCode EPSGetEigenvectorLeft(EPS,PetscInt,Vec,Vec);
extern PetscErrorCode EPSComputeRelativeError(EPS,PetscInt,PetscReal*);
extern PetscErrorCode EPSComputeRelativeErrorLeft(EPS,PetscInt,PetscReal*);
extern PetscErrorCode EPSComputeResidualNorm(EPS,PetscInt,PetscReal*);
extern PetscErrorCode EPSComputeResidualNormLeft(EPS,PetscInt,PetscReal*);
extern PetscErrorCode EPSGetInvariantSubspace(EPS,Vec*);
extern PetscErrorCode EPSGetInvariantSubspaceLeft(EPS,Vec*);
extern PetscErrorCode EPSGetErrorEstimate(EPS,PetscInt,PetscReal*);
extern PetscErrorCode EPSGetErrorEstimateLeft(EPS,PetscInt,PetscReal*);

extern PetscErrorCode EPSMonitorSet(EPS,PetscErrorCode (*)(EPS,PetscInt,PetscInt,PetscScalar*,PetscScalar*,PetscReal*,PetscInt,void*),
                                    void*,PetscErrorCode (*monitordestroy)(void*));
extern PetscErrorCode EPSMonitorCancel(EPS);
extern PetscErrorCode EPSGetMonitorContext(EPS,void **);
extern PetscErrorCode EPSGetIterationNumber(EPS,PetscInt*);
extern PetscErrorCode EPSGetOperationCounters(EPS,PetscInt*,PetscInt*,PetscInt*);

extern PetscErrorCode EPSSetWhichEigenpairs(EPS,EPSWhich);
extern PetscErrorCode EPSGetWhichEigenpairs(EPS,EPSWhich*);
extern PetscErrorCode EPSSetLeftVectorsWanted(EPS,PetscBool);
extern PetscErrorCode EPSGetLeftVectorsWanted(EPS,PetscBool*);
extern PetscErrorCode EPSSetMatrixNorms(EPS,PetscReal,PetscReal,PetscBool);
extern PetscErrorCode EPSGetMatrixNorms(EPS,PetscReal*,PetscReal*,PetscBool*);
extern PetscErrorCode EPSSetTrueResidual(EPS,PetscBool);
extern PetscErrorCode EPSGetTrueResidual(EPS,PetscBool*);
extern PetscErrorCode EPSSetEigenvalueComparison(EPS,PetscErrorCode (*func)(EPS,PetscScalar,PetscScalar,PetscScalar,PetscScalar,PetscInt*,void*),void*);
extern PetscErrorCode EPSIsGeneralized(EPS,PetscBool*);
extern PetscErrorCode EPSIsHermitian(EPS,PetscBool*);

extern PetscErrorCode EPSMonitorFirst(EPS,PetscInt,PetscInt,PetscScalar*,PetscScalar*,PetscReal*,PetscInt,void*);
extern PetscErrorCode EPSMonitorAll(EPS,PetscInt,PetscInt,PetscScalar*,PetscScalar*,PetscReal*,PetscInt,void*);
extern PetscErrorCode EPSMonitorConverged(EPS,PetscInt,PetscInt,PetscScalar*,PetscScalar*,PetscReal*,PetscInt,void*);
extern PetscErrorCode EPSMonitorLG(EPS,PetscInt,PetscInt,PetscScalar*,PetscScalar*,PetscReal*,PetscInt,void*);
extern PetscErrorCode EPSMonitorLGAll(EPS,PetscInt,PetscInt,PetscScalar*,PetscScalar*,PetscReal*,PetscInt,void*);

extern PetscErrorCode EPSSetTrackAll(EPS,PetscBool);
extern PetscErrorCode EPSGetTrackAll(EPS,PetscBool*);

extern PetscErrorCode EPSSetDeflationSpace(EPS,PetscInt,Vec*);
extern PetscErrorCode EPSRemoveDeflationSpace(EPS);
extern PetscErrorCode EPSSetInitialSpace(EPS,PetscInt,Vec*);
extern PetscErrorCode EPSSetInitialSpaceLeft(EPS,PetscInt,Vec*);

extern PetscErrorCode EPSSetOptionsPrefix(EPS,const char*);
extern PetscErrorCode EPSAppendOptionsPrefix(EPS,const char*);
extern PetscErrorCode EPSGetOptionsPrefix(EPS,const char*[]);

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

extern PetscErrorCode EPSGetConvergedReason(EPS,EPSConvergedReason *);

extern PetscErrorCode EPSSortEigenvalues(EPS,PetscInt,PetscScalar*,PetscScalar*,PetscInt*);
extern PetscErrorCode EPSSortEigenvaluesReal(EPS,PetscInt,PetscReal*,PetscInt*);
extern PetscErrorCode EPSCompareEigenvalues(EPS,PetscScalar,PetscScalar,PetscScalar,PetscScalar,PetscInt*);
extern PetscErrorCode EPSDenseNHEP(PetscInt,PetscScalar*,PetscScalar*,PetscScalar*,PetscScalar*,PetscScalar*);
extern PetscErrorCode EPSDenseGNHEP(PetscInt,PetscScalar*,PetscScalar*,PetscScalar*,PetscScalar*,PetscScalar*,PetscScalar*);
extern PetscErrorCode EPSDenseHEP(PetscInt,PetscScalar*,PetscInt,PetscReal*,PetscScalar*);
extern PetscErrorCode EPSDenseGHEP(PetscInt,PetscScalar*,PetscScalar*,PetscReal*,PetscScalar*);
extern PetscErrorCode EPSDenseHessenberg(PetscInt,PetscInt,PetscScalar*,PetscInt,PetscScalar*);
extern PetscErrorCode EPSDenseSchur(PetscInt,PetscInt,PetscScalar*,PetscInt,PetscScalar*,PetscScalar*,PetscScalar*);
extern PetscErrorCode EPSSortDenseSchur(EPS,PetscInt,PetscInt,PetscScalar*,PetscInt,PetscScalar*,PetscScalar*,PetscScalar*);
extern PetscErrorCode EPSSortDenseSchurGeneralized(EPS,PetscInt,PetscInt,PetscInt,PetscScalar*,PetscScalar*,PetscInt,PetscScalar*,PetscScalar*,PetscScalar*,PetscScalar*);
extern PetscErrorCode EPSDenseTridiagonal(PetscInt,PetscReal*,PetscReal*,PetscReal*,PetscScalar*);

extern PetscErrorCode EPSGetStartVector(EPS,PetscInt,Vec,PetscBool*);
extern PetscErrorCode EPSGetStartVectorLeft(EPS,PetscInt,Vec,PetscBool*);

extern PetscErrorCode EPSRegister(const char*,const char*,const char*,PetscErrorCode(*)(EPS));
#if defined(PETSC_USE_DYNAMIC_LIBRARIES)
#define EPSRegisterDynamic(a,b,c,d) EPSRegister(a,b,c,0)
#else
#define EPSRegisterDynamic(a,b,c,d) EPSRegister(a,b,c,d)
#endif
extern PetscErrorCode EPSRegisterDestroy(void);

/* --------- options specific to particular eigensolvers -------- */

/*E
    EPSPowerShiftType - determines the type of shift used in the Power iteration

    Level: advanced

.seealso: EPSPowerSetShiftType(), EPSPowerGetShiftType()
E*/
typedef enum { EPS_POWER_SHIFT_CONSTANT,
               EPS_POWER_SHIFT_RAYLEIGH,
               EPS_POWER_SHIFT_WILKINSON } EPSPowerShiftType;

extern PetscErrorCode EPSPowerSetShiftType(EPS,EPSPowerShiftType);
extern PetscErrorCode EPSPowerGetShiftType(EPS,EPSPowerShiftType*);

extern PetscErrorCode EPSArnoldiSetDelayed(EPS,PetscBool);
extern PetscErrorCode EPSArnoldiGetDelayed(EPS,PetscBool*);

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

extern PetscErrorCode EPSLanczosSetReorthog(EPS,EPSLanczosReorthogType);
extern PetscErrorCode EPSLanczosGetReorthog(EPS,EPSLanczosReorthogType*);

extern PetscErrorCode EPSBlzpackSetBlockSize(EPS,PetscInt);
extern PetscErrorCode EPSBlzpackSetInterval(EPS,PetscReal,PetscReal);
extern PetscErrorCode EPSBlzpackSetNSteps(EPS,PetscInt);

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

extern PetscErrorCode EPSPRIMMESetBlockSize(EPS eps,PetscInt bs);
extern PetscErrorCode EPSPRIMMESetMethod(EPS eps, EPSPRIMMEMethod method);
extern PetscErrorCode EPSPRIMMEGetBlockSize(EPS eps,PetscInt *bs);
extern PetscErrorCode EPSPRIMMEGetMethod(EPS eps, EPSPRIMMEMethod *method);

extern PetscErrorCode EPSGDSetKrylovStart(EPS eps,PetscBool krylovstart);
extern PetscErrorCode EPSGDGetKrylovStart(EPS eps,PetscBool *krylovstart);
extern PetscErrorCode EPSGDSetBlockSize(EPS eps,PetscInt blocksize);
extern PetscErrorCode EPSGDGetBlockSize(EPS eps,PetscInt *blocksize);
extern PetscErrorCode EPSGDSetRestart(EPS eps,PetscInt minv,PetscInt plusk);
extern PetscErrorCode EPSGDGetRestart(EPS eps,PetscInt *minv,PetscInt *plusk);
extern PetscErrorCode EPSGDSetInitialSize(EPS eps,PetscInt initialsize);
extern PetscErrorCode EPSGDGetInitialSize(EPS eps,PetscInt *initialsize);

extern PetscErrorCode EPSJDSetKrylovStart(EPS eps,PetscBool krylovstart);
extern PetscErrorCode EPSJDGetKrylovStart(EPS eps,PetscBool *krylovstart);
extern PetscErrorCode EPSJDSetBlockSize(EPS eps,PetscInt blocksize);
extern PetscErrorCode EPSJDGetBlockSize(EPS eps,PetscInt *blocksize);
extern PetscErrorCode EPSJDSetRestart(EPS eps,PetscInt minv,PetscInt plusk);
extern PetscErrorCode EPSJDGetRestart(EPS eps,PetscInt *minv,PetscInt *plusk);
extern PetscErrorCode EPSJDSetInitialSize(EPS eps,PetscInt initialsize);
extern PetscErrorCode EPSJDGetInitialSize(EPS eps,PetscInt *initialsize);
extern PetscErrorCode EPSJDSetFix(EPS eps,PetscReal fix);
extern PetscErrorCode EPSJDGetFix(EPS eps,PetscReal *fix);

PETSC_EXTERN_CXX_END
#endif

