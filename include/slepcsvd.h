/*
   User interface for the SLEPC singular value solvers. 

   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
      SLEPc - Scalable Library for Eigenvalue Problem Computations
      Copyright (c) 2002-2007, Universidad Politecnica de Valencia, Spain

      This file is part of SLEPc. See the README file for conditions of use
      and additional information.
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
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

/*E
    SVDType - String with the name of a SLEPc singular value solver

   Level: beginner

.seealso: SVDSetType(), SVD
E*/
#define SVDType const char*
#define SVDCROSS       "cross"
#define SVDCYCLIC      "cyclic"
#define SVDLAPACK      "lapack"
#define SVDLANCZOS     "lanczos"
#define SVDTRLANCZOS   "trlanczos"

/*E
    SVDTransposeMode - determines how to handle the transpose of the matrix

    Level: advanced

.seealso: SVDSetTransposeMode(), SVDGetTransposeMode()
E*/
typedef enum { SVD_TRANSPOSE_EXPLICIT, SVD_TRANSPOSE_IMPLICIT } SVDTransposeMode;

/*E
    SVDWhich - determines whether largest or smallest singular triplets
    are to be computed

    Level: intermediate

.seealso: SVDSetWhichSingularTriplets(), SVDGetWhichSingularTriplets()
E*/
typedef enum { SVD_LARGEST, SVD_SMALLEST } SVDWhich;

/*E
    SVDConvergedReason - reason a singular value solver was said to 
         have converged or diverged

   Level: beginner

.seealso: SVDSolve(), SVDGetConvergedReason(), SVDSetTolerances()
E*/
typedef enum {/* converged */
              SVD_CONVERGED_TOL                =  2,
              /* diverged */
              SVD_DIVERGED_ITS                 = -3,
              SVD_DIVERGED_BREAKDOWN           = -4,
              SVD_CONVERGED_ITERATING          =  0 } SVDConvergedReason;

EXTERN PetscErrorCode SVDCreate(MPI_Comm,SVD*);
EXTERN PetscErrorCode SVDSetIP(SVD,IP);
EXTERN PetscErrorCode SVDGetIP(SVD,IP*);
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
EXTERN PetscErrorCode SVDSetOptionsPrefix(SVD,const char*);
EXTERN PetscErrorCode SVDAppendOptionsPrefix(SVD,const char*);
EXTERN PetscErrorCode SVDGetOptionsPrefix(SVD,const char*[]);
EXTERN PetscErrorCode SVDSetUp(SVD);
EXTERN PetscErrorCode SVDSolve(SVD);
EXTERN PetscErrorCode SVDGetIterationNumber(SVD,int*);
EXTERN PetscErrorCode SVDGetConvergedReason(SVD,SVDConvergedReason*);
EXTERN PetscErrorCode SVDGetConverged(SVD,int*);
EXTERN PetscErrorCode SVDGetSingularTriplet(SVD,int,PetscReal*,Vec,Vec);
EXTERN PetscErrorCode SVDComputeResidualNorms(SVD,int,PetscReal*,PetscReal*);
EXTERN PetscErrorCode SVDComputeRelativeError(SVD,int,PetscReal*);
EXTERN PetscErrorCode SVDGetOperationCounters(SVD,int*,int*);
EXTERN PetscErrorCode SVDView(SVD,PetscViewer);
EXTERN PetscErrorCode SVDDestroy(SVD);
EXTERN PetscErrorCode SVDInitializePackage(char*);

EXTERN PetscErrorCode SVDMonitorSet(SVD,PetscErrorCode (*)(SVD,int,int,PetscReal*,PetscReal*,int,void*),
                                    void*,PetscErrorCode (*monitordestroy)(void*));
EXTERN PetscErrorCode SVDMonitorCancel(SVD);
EXTERN PetscErrorCode SVDGetMonitorContext(SVD,void **);
EXTERN PetscErrorCode SVDMonitorDefault(SVD,int,int,PetscReal*,PetscReal*,int,void*);
EXTERN PetscErrorCode SVDMonitorLG(SVD,int,int,PetscReal*,PetscReal*,int,void*);

EXTERN PetscErrorCode SVDDense(int,int,PetscScalar*,PetscReal*,PetscScalar*,PetscScalar*);

EXTERN PetscErrorCode SVDCrossSetEPS(SVD,EPS);
EXTERN PetscErrorCode SVDCrossGetEPS(SVD,EPS*);

EXTERN PetscErrorCode SVDCyclicSetExplicitMatrix(SVD,PetscTruth);
EXTERN PetscErrorCode SVDCyclicGetExplicitMatrix(SVD,PetscTruth*);
EXTERN PetscErrorCode SVDCyclicSetEPS(SVD,EPS);
EXTERN PetscErrorCode SVDCyclicGetEPS(SVD,EPS*);

EXTERN PetscErrorCode SVDLanczosSetOneSide(SVD,PetscTruth);

EXTERN PetscErrorCode SVDTRLanczosSetOneSide(SVD,PetscTruth);

EXTERN PetscErrorCode SVDDense(int,int,PetscScalar*,PetscReal*,PetscScalar*,PetscScalar*);

#endif
