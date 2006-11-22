/*
   User interface for the SLEPC singular value solvers. 
*/
#if !defined(__SLEPCSVD_H)
#define __SLEPCSVD_H
#include "slepc.h"
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

EXTERN PetscErrorCode SVDCreate(MPI_Comm,SVD*);
EXTERN PetscErrorCode SVDSetType(SVD,SVDType);
EXTERN PetscErrorCode SVDGetType(SVD,SVDType*);
EXTERN PetscErrorCode SVDSetOperator(SVD,Mat);
EXTERN PetscErrorCode SVDGetOperator(SVD,Mat*);
EXTERN PetscErrorCode SVDSetFromOptions(SVD);
EXTERN PetscErrorCode SVDSetUp(SVD);
EXTERN PetscErrorCode SVDSolve(SVD);
EXTERN PetscErrorCode SVDGetConverged(SVD,int*);
EXTERN PetscErrorCode SVDGetSingularTriplet(SVD,int,PetscReal*,Vec,Vec);
EXTERN PetscErrorCode SVDComputeResidualNorm(SVD,int,PetscReal*);
EXTERN PetscErrorCode SVDView(SVD,PetscViewer);
EXTERN PetscErrorCode SVDDestroy(SVD);
EXTERN PetscErrorCode SVDInitializePackage(char*);

#endif
