/*
   User interface for the SLEPC eigenproblem solvers. 
*/
#if !defined(__SLEPCEPS_H)
#define __SLEPCEPS_H
#include "slepc.h"
#include "slepcst.h"
PETSC_EXTERN_CXX_BEGIN

extern PetscCookie EPS_COOKIE;

/*S
     EPS - Abstract SLEPc object that manages all the eigenvalue 
     problem solvers.

   Level: beginner

  Concepts: eigen solvers

.seealso:  EPSCreate(), ST
S*/
typedef struct _p_EPS* EPS;

#define EPSPOWER     "power"
#define EPSSUBSPACE  "subspace"
#define EPSARNOLDI   "arnoldi"
#define EPSLAPACK    "lapack"
/* the next ones are interfaces to external libraries */
#define EPSARPACK    "arpack"
#define EPSBLZPACK   "blzpack"
#define EPSPLANSO    "planso"
#define EPSTRLAN     "trlan"

typedef char * EPSType;

typedef enum { EPS_HEP=1,  EPS_GHEP,
               EPS_NHEP,   EPS_GNHEP } EPSProblemType;

typedef enum { EPS_LARGEST_MAGNITUDE, EPS_SMALLEST_MAGNITUDE,
               EPS_LARGEST_REAL,      EPS_SMALLEST_REAL,
               EPS_LARGEST_IMAGINARY, EPS_SMALLEST_IMAGINARY } EPSWhich;

typedef enum { EPS_MGS_ORTH,  EPS_CGS_ORTH } EPSOrthogonalizationType;
typedef enum { EPS_ORTH_REFINE_NEVER, EPS_ORTH_REFINE_IFNEEDED,
               EPS_ORTH_REFINE_ALWAYS } EPSOrthogonalizationRefinementType;

EXTERN PetscErrorCode EPSCreate(MPI_Comm,EPS *);
EXTERN PetscErrorCode EPSDestroy(EPS);
EXTERN PetscErrorCode EPSSetType(EPS,EPSType);
EXTERN PetscErrorCode EPSGetType(EPS,EPSType*);
EXTERN PetscErrorCode EPSSetProblemType(EPS,EPSProblemType);
EXTERN PetscErrorCode EPSGetProblemType(EPS,EPSProblemType*);
EXTERN PetscErrorCode EPSSetOperators(EPS,Mat,Mat);
EXTERN PetscErrorCode EPSSetFromOptions(EPS);
EXTERN PetscErrorCode EPSSetUp(EPS);
EXTERN PetscErrorCode EPSSolve(EPS);
EXTERN PetscErrorCode EPSView(EPS,PetscViewer);

extern PetscFList EPSList;
EXTERN PetscErrorCode EPSRegisterAll(char *);
EXTERN PetscErrorCode EPSRegisterDestroy(void);
EXTERN PetscErrorCode EPSRegister(char*,char*,char*,int(*)(EPS));
#if defined(PETSC_USE_DYNAMIC_LIBRARIES)
#define EPSRegisterDynamic(a,b,c,d) EPSRegister(a,b,c,0)
#else
#define EPSRegisterDynamic(a,b,c,d) EPSRegister(a,b,c,d)
#endif

EXTERN PetscErrorCode EPSSetST(EPS,ST);
EXTERN PetscErrorCode EPSGetST(EPS,ST*);
EXTERN PetscErrorCode EPSSetTolerances(EPS,PetscReal,int);
EXTERN PetscErrorCode EPSGetTolerances(EPS,PetscReal*,int*);
EXTERN PetscErrorCode EPSSetDimensions(EPS,int,int);
EXTERN PetscErrorCode EPSGetDimensions(EPS,int*,int*);

EXTERN PetscErrorCode EPSGetConverged(EPS,int*);
EXTERN PetscErrorCode EPSGetEigenpair(EPS,int,PetscScalar*,PetscScalar*,Vec,Vec);
EXTERN PetscErrorCode EPSComputeRelativeError(EPS,int,PetscReal*);
EXTERN PetscErrorCode EPSComputeResidualNorm(EPS,int,PetscReal*);
EXTERN PetscErrorCode EPSGetInvariantSubspace(EPS,Vec*);
EXTERN PetscErrorCode EPSGetErrorEstimate(EPS,int,PetscReal*);

EXTERN PetscErrorCode EPSSetMonitor(EPS,int (*)(EPS,int,int,PetscScalar*,PetscScalar*,PetscReal*,int,void*),void*);
EXTERN PetscErrorCode EPSClearMonitor(EPS);
EXTERN PetscErrorCode EPSGetMonitorContext(EPS,void **);
EXTERN PetscErrorCode EPSGetIterationNumber(EPS,int*);
EXTERN PetscErrorCode EPSGetNumberLinearIterations(EPS eps,int*);

EXTERN PetscErrorCode EPSSetInitialVector(EPS,Vec);
EXTERN PetscErrorCode EPSGetInitialVector(EPS,Vec*);
EXTERN PetscErrorCode EPSSetWhichEigenpairs(EPS,EPSWhich);
EXTERN PetscErrorCode EPSGetWhichEigenpairs(EPS,EPSWhich*);
EXTERN PetscErrorCode EPSSetOrthogonalization(EPS,EPSOrthogonalizationType,EPSOrthogonalizationRefinementType,PetscReal);
EXTERN PetscErrorCode EPSGetOrthogonalization(EPS,EPSOrthogonalizationType*,EPSOrthogonalizationRefinementType*,PetscReal*);
EXTERN PetscErrorCode EPSPurge(EPS,Vec);
EXTERN PetscErrorCode EPSOrthogonalize(EPS,int,Vec*,Vec,PetscScalar*,PetscReal*,PetscTruth*);

EXTERN PetscErrorCode EPSIsGeneralized(EPS,PetscTruth*);
EXTERN PetscErrorCode EPSIsHermitian(EPS,PetscTruth*);

EXTERN PetscErrorCode EPSDefaultMonitor(EPS,int,int,PetscScalar*,PetscScalar*,PetscReal*,int,void*);

EXTERN PetscErrorCode EPSSetOptionsPrefix(EPS,char*);
EXTERN PetscErrorCode EPSAppendOptionsPrefix(EPS,char*);
EXTERN PetscErrorCode EPSGetOptionsPrefix(EPS,char**);

typedef enum {/* converged */
              EPS_CONVERGED_TOL                =  2,
              /* diverged */
              EPS_DIVERGED_ITS                 = -3,
              EPS_DIVERGED_BREAKDOWN           = -4,
              EPS_DIVERGED_NONSYMMETRIC        = -5,
              EPS_CONVERGED_ITERATING          =  0} EPSConvergedReason;

EXTERN PetscErrorCode EPSGetConvergedReason(EPS,EPSConvergedReason *);

EXTERN PetscErrorCode EPSComputeExplicitOperator(EPS,Mat*);
EXTERN PetscErrorCode EPSSortEigenvalues(int,PetscScalar*,PetscScalar*,EPSWhich,int,int*);
EXTERN PetscErrorCode EPSDenseNHEP(int,PetscScalar*,PetscScalar*,PetscScalar*,PetscScalar*);
EXTERN PetscErrorCode EPSDenseNHEPSorted(int,PetscScalar*,PetscScalar*,PetscScalar*,PetscScalar*,int,EPSWhich);
EXTERN PetscErrorCode EPSDenseSchur(PetscScalar*,PetscScalar*,PetscScalar*,PetscScalar*,int,int);
EXTERN PetscErrorCode EPSQRDecomposition(EPS,Vec*,int,int,PetscScalar*,int);
EXTERN PetscErrorCode EPSReverseProjection(EPS,Vec*,PetscScalar*,int,int,Vec*);
EXTERN PetscErrorCode EPSSwapEigenpairs(EPS,int,int);

EXTERN PetscErrorCode STPreSolve(ST,EPS);
EXTERN PetscErrorCode STPostSolve(ST,EPS);

EXTERN PetscErrorCode EPSAttachDeflationSpace(EPS,int,Vec*,PetscTruth);
EXTERN PetscErrorCode EPSRemoveDeflationSpace(EPS);

/* --------- options specific to particular eigensolvers -------- */

typedef enum { EPSPOWER_SHIFT_CONSTANT, EPSPOWER_SHIFT_RAYLEIGH,
               EPSPOWER_SHIFT_WILKINSON } EPSPowerShiftType;

EXTERN PetscErrorCode EPSPowerSetShiftType(EPS,EPSPowerShiftType);
EXTERN PetscErrorCode EPSPowerGetShiftType(EPS,EPSPowerShiftType*);

EXTERN PetscErrorCode EPSBlzpackSetBlockSize(EPS,int);
EXTERN PetscErrorCode EPSBlzpackSetInterval(EPS,PetscReal,PetscReal);
EXTERN PetscErrorCode EPSBlzpackSetNSteps(EPS,int);

PETSC_EXTERN_CXX_END
#endif

