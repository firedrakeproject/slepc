/*
   User interface for the SLEPC eigenproblem solvers. 

   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
      SLEPc - Scalable Library for Eigenvalue Problem Computations
      Copyright (c) 2002-2007, Universidad Politecnica de Valencia, Spain

      This file is part of SLEPc. See the README file for conditions of use
      and additional information.
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
*/

#if !defined(__SLEPCEPS_H)
#define __SLEPCEPS_H
#include "slepc.h"
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
typedef enum { EPS_HEP=1,  EPS_GHEP,
               EPS_NHEP,   EPS_GNHEP, EPS_PGNHEP } EPSProblemType;

/*E
    EPSProjection - determines the type of projection employed
    by the eigensolver

    Level: beginner

.seealso: EPSSetProjection(), EPSGetProjection()
E*/
typedef enum { EPS_RITZ=1,   EPS_HARMONIC,
               EPS_REFINED,  EPS_REFINED_HARMONIC } EPSProjection;

/*E
    EPSClass - determines if the eigensolver is one- or two-sided

    Level: intermediate

.seealso: EPSSetClass(), EPSGetClass()
E*/
typedef enum { EPS_ONE_SIDE, EPS_TWO_SIDE } EPSClass;

/*E
    EPSWhich - determines which part of the spectrum is requested

    Level: intermediate

.seealso: EPSSetWhichEigenpairs(), EPSGetWhichEigenpairs()
E*/
typedef enum { EPS_LARGEST_MAGNITUDE, EPS_SMALLEST_MAGNITUDE,
               EPS_LARGEST_REAL,      EPS_SMALLEST_REAL,
               EPS_LARGEST_IMAGINARY, EPS_SMALLEST_IMAGINARY } EPSWhich;

EXTERN PetscErrorCode EPSCreate(MPI_Comm,EPS *);
EXTERN PetscErrorCode EPSDestroy(EPS);
EXTERN PetscErrorCode EPSSetType(EPS,const EPSType);
EXTERN PetscErrorCode EPSGetType(EPS,const EPSType*);
EXTERN PetscErrorCode EPSSetProblemType(EPS,EPSProblemType);
EXTERN PetscErrorCode EPSGetProblemType(EPS,EPSProblemType*);
EXTERN PetscErrorCode EPSSetProjection(EPS,EPSProjection);
EXTERN PetscErrorCode EPSGetProjection(EPS,EPSProjection*);
EXTERN PetscErrorCode EPSSetClass(EPS,EPSClass);
EXTERN PetscErrorCode EPSGetClass(EPS,EPSClass*);
EXTERN PetscErrorCode EPSSetOperators(EPS,Mat,Mat);
EXTERN PetscErrorCode EPSGetOperators(EPS,Mat*,Mat*);
EXTERN PetscErrorCode EPSSetFromOptions(EPS);
EXTERN PetscErrorCode EPSSetUp(EPS);
EXTERN PetscErrorCode EPSSolve(EPS);
EXTERN PetscErrorCode EPSView(EPS,PetscViewer);

EXTERN PetscErrorCode EPSInitializePackage(char *);

EXTERN PetscErrorCode EPSSetTarget(EPS,PetscScalar);
EXTERN PetscErrorCode EPSGetTarget(EPS,PetscScalar*);
EXTERN PetscErrorCode EPSSetST(EPS,ST);
EXTERN PetscErrorCode EPSGetST(EPS,ST*);
EXTERN PetscErrorCode EPSSetIP(EPS,IP);
EXTERN PetscErrorCode EPSGetIP(EPS,IP*);
EXTERN PetscErrorCode EPSSetTolerances(EPS,PetscReal,PetscInt);
EXTERN PetscErrorCode EPSGetTolerances(EPS,PetscReal*,PetscInt*);
EXTERN PetscErrorCode EPSSetDimensions(EPS,PetscInt,PetscInt);
EXTERN PetscErrorCode EPSGetDimensions(EPS,PetscInt*,PetscInt*);

EXTERN PetscErrorCode EPSGetConverged(EPS,PetscInt*);
EXTERN PetscErrorCode EPSGetEigenpair(EPS,PetscInt,PetscScalar*,PetscScalar*,Vec,Vec);
EXTERN PetscErrorCode EPSGetValue(EPS,PetscInt,PetscScalar*,PetscScalar*);
EXTERN PetscErrorCode EPSGetRightVector(EPS,PetscInt,Vec,Vec);
EXTERN PetscErrorCode EPSGetLeftVector(EPS,PetscInt,Vec,Vec);
EXTERN PetscErrorCode EPSComputeRelativeError(EPS,PetscInt,PetscReal*);
EXTERN PetscErrorCode EPSComputeRelativeErrorLeft(EPS,PetscInt,PetscReal*);
EXTERN PetscErrorCode EPSComputeResidualNorm(EPS,PetscInt,PetscReal*);
EXTERN PetscErrorCode EPSComputeResidualNormLeft(EPS,PetscInt,PetscReal*);
EXTERN PetscErrorCode EPSGetInvariantSubspace(EPS,Vec*);
EXTERN PetscErrorCode EPSGetLeftInvariantSubspace(EPS,Vec*);
EXTERN PetscErrorCode EPSGetErrorEstimate(EPS,PetscInt,PetscReal*);
EXTERN PetscErrorCode EPSGetErrorEstimateLeft(EPS,PetscInt,PetscReal*);

EXTERN PetscErrorCode EPSMonitorSet(EPS,PetscErrorCode (*)(EPS,PetscInt,PetscInt,PetscScalar*,PetscScalar*,PetscReal*,PetscInt,void*),
                                    void*,PetscErrorCode (*monitordestroy)(void*));
EXTERN PetscErrorCode EPSMonitorCancel(EPS);
EXTERN PetscErrorCode EPSGetMonitorContext(EPS,void **);
EXTERN PetscErrorCode EPSGetIterationNumber(EPS,PetscInt*);
EXTERN PetscErrorCode EPSGetOperationCounters(EPS,PetscInt*,PetscInt*,PetscInt*);

EXTERN PetscErrorCode EPSSetInitialVector(EPS,Vec);
EXTERN PetscErrorCode EPSGetInitialVector(EPS,Vec*);
EXTERN PetscErrorCode EPSSetLeftInitialVector(EPS,Vec);
EXTERN PetscErrorCode EPSGetLeftInitialVector(EPS,Vec*);
EXTERN PetscErrorCode EPSSetWhichEigenpairs(EPS,EPSWhich);
EXTERN PetscErrorCode EPSGetWhichEigenpairs(EPS,EPSWhich*);
EXTERN PetscErrorCode EPSIsGeneralized(EPS,PetscTruth*);
EXTERN PetscErrorCode EPSIsHermitian(EPS,PetscTruth*);

EXTERN PetscErrorCode EPSMonitorDefault(EPS,PetscInt,PetscInt,PetscScalar*,PetscScalar*,PetscReal*,PetscInt,void*);
EXTERN PetscErrorCode EPSMonitorLG(EPS,PetscInt,PetscInt,PetscScalar*,PetscScalar*,PetscReal*,PetscInt,void*);

EXTERN PetscErrorCode EPSAttachDeflationSpace(EPS,PetscInt,Vec*,PetscTruth);
EXTERN PetscErrorCode EPSRemoveDeflationSpace(EPS);

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

EXTERN PetscErrorCode EPSSortEigenvalues(PetscInt,PetscScalar*,PetscScalar*,EPSWhich,PetscInt,PetscInt*);
EXTERN PetscErrorCode EPSSortEigenvaluesReal(PetscInt,PetscReal*,EPSWhich,PetscInt,PetscInt*,PetscReal*);
EXTERN PetscErrorCode EPSDenseNHEP(PetscInt,PetscScalar*,PetscScalar*,PetscScalar*,PetscScalar*,PetscScalar*);
EXTERN PetscErrorCode EPSDenseGNHEP(PetscInt,PetscScalar*,PetscScalar*,PetscScalar*,PetscScalar*,PetscScalar*,PetscScalar*);
EXTERN PetscErrorCode EPSDenseHEP(PetscInt,PetscScalar*,PetscInt,PetscReal*,PetscScalar*);
EXTERN PetscErrorCode EPSDenseGHEP(PetscInt,PetscScalar*,PetscScalar*,PetscReal*,PetscScalar*);
EXTERN PetscErrorCode EPSDenseHessenberg(PetscInt,PetscInt,PetscScalar*,PetscInt,PetscScalar*);
EXTERN PetscErrorCode EPSDenseSchur(PetscInt,PetscInt,PetscScalar*,PetscInt,PetscScalar*,PetscScalar*,PetscScalar*);
EXTERN PetscErrorCode EPSSortDenseSchur(PetscInt,PetscInt,PetscScalar*,PetscInt,PetscScalar*,PetscScalar*,PetscScalar*,EPSWhich);
EXTERN PetscErrorCode EPSSortDenseSchurTarget(PetscInt,PetscInt,PetscScalar*,PetscInt,PetscScalar*,PetscScalar*,PetscScalar*,PetscScalar,EPSWhich);
EXTERN PetscErrorCode EPSDenseTridiagonal(PetscInt,PetscScalar*,PetscInt,PetscReal*,PetscScalar*);

EXTERN PetscErrorCode EPSGetStartVector(EPS,PetscInt,Vec,PetscTruth*);
EXTERN PetscErrorCode EPSGetLeftStartVector(EPS,PetscInt,Vec);

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
typedef enum { EPSPOWER_SHIFT_CONSTANT, EPSPOWER_SHIFT_RAYLEIGH,
               EPSPOWER_SHIFT_WILKINSON } EPSPowerShiftType;

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
typedef enum { EPSLANCZOS_REORTHOG_LOCAL, 
               EPSLANCZOS_REORTHOG_FULL,
               EPSLANCZOS_REORTHOG_SELECTIVE,
               EPSLANCZOS_REORTHOG_PERIODIC,
               EPSLANCZOS_REORTHOG_PARTIAL, 
	       EPSLANCZOS_REORTHOG_DELAYED } EPSLanczosReorthogType;

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
typedef enum {
  EPSPRIMME_DYNAMIC,
  EPSPRIMME_DEFAULT_MIN_TIME,
  EPSPRIMME_DEFAULT_MIN_MATVECS,
  EPSPRIMME_ARNOLDI,
  EPSPRIMME_GD,
  EPSPRIMME_GD_PLUSK,
  EPSPRIMME_GD_OLSEN_PLUSK,
  EPSPRIMME_JD_OLSEN_PLUSK,
  EPSPRIMME_RQI,
  EPSPRIMME_JDQR,
  EPSPRIMME_JDQMR,
  EPSPRIMME_JDQMR_ETOL,
  EPSPRIMME_SUBSPACE_ITERATION,
  EPSPRIMME_LOBPCG_ORTHOBASIS,
  EPSPRIMME_LOBPCG_ORTHOBASISW
} EPSPRIMMEMethod;

/*E
    EPSPRIMMEPrecond - determines the type of preconditioning
    used in the PRIMME library

    Level: advanced

.seealso: EPSPRIMMESetPrecond(), EPSPRIMMEGetPrecond()
E*/
typedef enum {
  EPSPRIMME_NONE,
  EPSPRIMME_DIAGONAL
} EPSPRIMMEPrecond;

EXTERN PetscErrorCode EPSPRIMMESetBlockSize(EPS eps,PetscInt bs);
EXTERN PetscErrorCode EPSPRIMMESetMethod(EPS eps, EPSPRIMMEMethod method);
EXTERN PetscErrorCode EPSPRIMMESetPrecond(EPS eps, EPSPRIMMEPrecond precond);
EXTERN PetscErrorCode EPSPRIMMEGetBlockSize(EPS eps,PetscInt *bs);
EXTERN PetscErrorCode EPSPRIMMEGetMethod(EPS eps, EPSPRIMMEMethod *method);
EXTERN PetscErrorCode EPSPRIMMEGetPrecond(EPS eps, EPSPRIMMEPrecond *precond);

PETSC_EXTERN_CXX_END
#endif

