/*
   User interface for the SLEPC singular value solvers. 
*/
#if !defined(__SLEPCSVD_H)
#define __SLEPCSVD_H
#include "slepc.h"
#include "slepceps.h"
PETSC_EXTERN_CXX_BEGIN

extern PetscCookie SVD_COOKIE;

/*S
     SVD - Abstract SLEPc object that manages all the singular value 
     problem solvers.

   Level: beginner

.seealso:  SVDCreate()
S*/
typedef struct _p_SVD* SVD;

#define SVDType const char*
#define SVDEIGENSOLVER "eigensolver"
#define SVDLAPACK      "lapack"
#define SVDLANCZOS     "lanczos"

typedef enum { SVD_TRANSPOSE_EXPLICIT, SVD_TRANSPOSE_IMPLICIT } SVDTransposeMode;

typedef enum { SVD_LARGEST, SVD_SMALLEST } SVDWhich;

typedef enum {/* converged */
              SVD_CONVERGED_TOL                =  2,
              /* diverged */
              SVD_DIVERGED_ITS                 = -3,
              SVD_DIVERGED_BREAKDOWN           = -4,
              SVD_CONVERGED_ITERATING          =  0 } SVDConvergedReason;

EXTERN PetscErrorCode SVDCreate(MPI_Comm,SVD*);
EXTERN PetscErrorCode SVDSetType(SVD,SVDType);
EXTERN PetscErrorCode SVDGetType(SVD,SVDType*);
EXTERN PetscErrorCode SVDSetOperator(SVD,Mat);
EXTERN PetscErrorCode SVDGetOperator(SVD,Mat*);
EXTERN PetscErrorCode SVDSetInitialVector(SVD,Vec);
EXTERN PetscErrorCode SVDGetInitialVector(SVD,Vec*);
EXTERN PetscErrorCode SVDSetTransposeMode(SVD,SVDTransposeMode);
EXTERN PetscErrorCode SVDGetTransposeMode(SVD,SVDTransposeMode*);
EXTERN PetscErrorCode SVDSetDimensions(SVD,int,int);
EXTERN PetscErrorCode SVDGetDimensions(SVD,int*,int*);
EXTERN PetscErrorCode SVDSetTolerances(SVD,PetscReal,int);
EXTERN PetscErrorCode SVDGetTolerances(SVD,PetscReal*,int*);
EXTERN PetscErrorCode SVDSetWhichSingularTriplets(SVD,SVDWhich);
EXTERN PetscErrorCode SVDGetWhichSingularTriplets(SVD,SVDWhich*);
EXTERN PetscErrorCode SVDSetFromOptions(SVD);
EXTERN PetscErrorCode SVDSetUp(SVD);
EXTERN PetscErrorCode SVDSolve(SVD);
EXTERN PetscErrorCode SVDGetIterationNumber(SVD,int*);
EXTERN PetscErrorCode SVDGetConverged(SVD,int*);
EXTERN PetscErrorCode SVDGetSingularTriplet(SVD,int,PetscReal*,Vec,Vec);
EXTERN PetscErrorCode SVDComputeResidualNorms(SVD,int,PetscReal*,PetscReal*);
EXTERN PetscErrorCode SVDView(SVD,PetscViewer);
EXTERN PetscErrorCode SVDDestroy(SVD);
EXTERN PetscErrorCode SVDInitializePackage(char*);

EXTERN PetscErrorCode SVDSetMonitor(SVD,PetscErrorCode (*)(SVD,int,int,PetscReal*,PetscReal*,int,void*),
                                    void*,PetscErrorCode (*monitordestroy)(void*));
EXTERN PetscErrorCode SVDClearMonitor(SVD);
EXTERN PetscErrorCode SVDGetMonitorContext(SVD,void **);
EXTERN PetscErrorCode SVDDefaultMonitor(SVD,int,int,PetscReal*,PetscReal*,int,void*);
EXTERN PetscErrorCode SVDLGMonitor(SVD,int,int,PetscReal*,PetscReal*,int,void*);

typedef enum { SVDEIGENSOLVER_ATA, SVDEIGENSOLVER_AAT,
               SVDEIGENSOLVER_CYCLIC } SVDEigensolverMode;

EXTERN PetscErrorCode SVDEigensolverSetMode(SVD,SVDEigensolverMode);
EXTERN PetscErrorCode SVDEigensolverGetMode(SVD,SVDEigensolverMode*);
EXTERN PetscErrorCode SVDEigensolverSetEPS(SVD,EPS);
EXTERN PetscErrorCode SVDEigensolverGetEPS(SVD,EPS*);

#endif
