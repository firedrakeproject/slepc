/*
   User interface for the SLEPC eigenproblem solvers. 
*/
#if !defined(__SLEPCEPS_H)
#define __SLEPCEPS_H
#include "slepc.h"
#include "slepcst.h"

extern int EPS_COOKIE;

/*S
     EPS - Abstract SLEPc object that manages all the eigenvalue 
     problem solvers.

   Level: beginner

  Concepts: eigen solvers

.seealso:  EPSCreate(), ST
S*/
typedef struct _p_EPS* EPS;

#define EPSPOWER     "power"
#define EPSRQI       "rqi"
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

typedef enum { EPS_MGS_ORTH,  EPS_CGS_ORTH,
               EPS_IR_ORTH } EPSOrthogonalizationType;

extern int EPSCreate(MPI_Comm,EPS *);
extern int EPSDestroy(EPS);
extern int EPSSetType(EPS,EPSType);
extern int EPSGetType(EPS,EPSType*);
extern int EPSSetProblemType(EPS,EPSProblemType);
extern int EPSGetProblemType(EPS,EPSProblemType*);
extern int EPSSetOperators(EPS,Mat,Mat);
extern int EPSSetFromOptions(EPS);
extern int EPSSetUp(EPS);
extern int EPSSolve(EPS);
extern int EPSView(EPS,PetscViewer);

extern PetscFList EPSList;
extern int EPSRegisterAll(char *);
extern int EPSRegisterDestroy(void);
extern int EPSRegister(char*,char*,char*,int(*)(EPS));
#if defined(PETSC_USE_DYNAMIC_LIBRARIES)
#define EPSRegisterDynamic(a,b,c,d) EPSRegister(a,b,c,0)
#else
#define EPSRegisterDynamic(a,b,c,d) EPSRegister(a,b,c,d)
#endif

extern int EPSSetST(EPS,ST);
extern int EPSGetST(EPS,ST*);
extern int EPSSetTolerances(EPS,PetscReal,int);
extern int EPSGetTolerances(EPS,PetscReal*,int*);
extern int EPSSetDimensions(EPS,int,int);
extern int EPSGetDimensions(EPS,int*,int*);

extern int EPSGetConverged(EPS,int*);
extern int EPSGetEigenpair(EPS,int,PetscScalar*,PetscScalar*,Vec,Vec);
extern int EPSComputeRelativeError(EPS,int,PetscReal*);
extern int EPSComputeResidualNorm(EPS,int,PetscReal*);
extern int EPSGetErrorEstimates(EPS,PetscReal**);

extern int EPSSetMonitor(EPS,int (*)(EPS,int,int,PetscReal*,int,void*),void*);
extern int EPSSetValuesMonitor(EPS,int (*)(EPS,int,int,PetscScalar*,PetscScalar*,int,void*),void*);
extern int EPSClearMonitor(EPS);
extern int EPSGetMonitorContext(EPS,void **);
extern int EPSGetIterationNumber(EPS,int*);

extern int EPSSetInitialVector(EPS,Vec);
extern int EPSGetInitialVector(EPS,Vec*);
extern int EPSSetDropEigenvectors(EPS);
extern int EPSSetWhichEigenpairs(EPS,EPSWhich);
extern int EPSGetWhichEigenpairs(EPS,EPSWhich*);
extern int EPSSetOrthogonalization(EPS,EPSOrthogonalizationType);
extern int EPSGetOrthogonalization(EPS,EPSOrthogonalizationType*);

extern int EPSIsGeneralized(EPS,PetscTruth*);
extern int EPSIsHermitian(EPS,PetscTruth*);

extern int EPSDefaultEstimatesMonitor(EPS,int,int,PetscReal*,int,void*);
extern int EPSDefaultValuesMonitor(EPS,int,int,PetscScalar*,PetscScalar*,int,void*);

extern int EPSSetOptionsPrefix(EPS,char*);
extern int EPSAppendOptionsPrefix(EPS,char*);
extern int EPSGetOptionsPrefix(EPS,char**);

typedef enum {/* converged */
              EPS_CONVERGED_TOL                =  2,
              /* diverged */
              EPS_DIVERGED_ITS                 = -3,
              EPS_DIVERGED_BREAKDOWN           = -4,
              EPS_DIVERGED_NONSYMMETRIC        = -5,
              EPS_CONVERGED_ITERATING          =  0} EPSConvergedReason;

extern int EPSGetConvergedReason(EPS,EPSConvergedReason *);

extern int EPSBackTransform(EPS);
extern int EPSComputeExplicitOperator(EPS,Mat*);
extern int EPSSortEigenvalues(int,PetscScalar*,PetscScalar*,EPSWhich,int,int*);
extern int EPSDenseNHEP(int,PetscScalar*,PetscScalar*,PetscScalar*,PetscScalar*);
extern int EPSDenseNHEPSorted(int,PetscScalar*,PetscScalar*,PetscScalar*,PetscScalar*,int,EPSWhich);
extern int EPSQRDecomposition(EPS,int,int,PetscScalar*,int);
extern int EPSReverseProjection(EPS,int,int,PetscScalar*);
extern int EPSSwapEigenpairs(EPS,int,int);

extern int STPreSolve(ST,EPS);
extern int STPostSolve(ST,EPS);
extern int EPSSetDefaults(EPS eps);

/* --------- options specific to particular eigensolvers -------- */

extern int EPSSubspaceSetInner(EPS,int);

extern int EPSBlzpackSetBlockSize(EPS,int);
extern int EPSBlzpackSetInterval(EPS,PetscReal,PetscReal);
extern int EPSBlzpackSetMatGetInertia(EPS,int (*f)(Mat,int*,int*,int*));

#endif

