/*
   User interface for SLEPc's singular value solvers.

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

#if !defined(__SLEPCSVD_H)
#define __SLEPCSVD_H
#include <slepceps.h>
#include <slepcbv.h>
#include <slepcds.h>

PETSC_EXTERN PetscErrorCode SVDInitializePackage(void);

/*S
     SVD - Abstract SLEPc object that manages all the singular value
     problem solvers.

   Level: beginner

.seealso:  SVDCreate()
S*/
typedef struct _p_SVD* SVD;

/*J
    SVDType - String with the name of a SLEPc singular value solver

   Level: beginner

.seealso: SVDSetType(), SVD
J*/
typedef const char* SVDType;
#define SVDCROSS       "cross"
#define SVDCYCLIC      "cyclic"
#define SVDLAPACK      "lapack"
#define SVDLANCZOS     "lanczos"
#define SVDTRLANCZOS   "trlanczos"

/* Logging support */
PETSC_EXTERN PetscClassId SVD_CLASSID;

/*E
    SVDWhich - Determines whether largest or smallest singular triplets
    are to be computed

    Level: intermediate

.seealso: SVDSetWhichSingularTriplets(), SVDGetWhichSingularTriplets()
E*/
typedef enum { SVD_LARGEST,
               SVD_SMALLEST } SVDWhich;

/*E
    SVDErrorType - The error type used to assess accuracy of computed solutions

    Level: intermediate

.seealso: SVDComputeError()
E*/
typedef enum { SVD_ERROR_ABSOLUTE,
               SVD_ERROR_RELATIVE } SVDErrorType;
PETSC_EXTERN const char *SVDErrorTypes[];

/*E
    SVDConv - Determines the convergence test

    Level: intermediate

.seealso: SVDSetConvergenceTest(), SVDSetConvergenceTestFunction()
E*/
typedef enum { SVD_CONV_ABS,
               SVD_CONV_REL,
               SVD_CONV_USER } SVDConv;

/*E
    SVDStop - Determines the stopping test

    Level: advanced

.seealso: SVDSetStoppingTest(), SVDSetStoppingTestFunction()
E*/
typedef enum { SVD_STOP_BASIC,
               SVD_STOP_USER } SVDStop;

/*E
    SVDConvergedReason - Reason a singular value solver was said to
         have converged or diverged

   Level: intermediate

.seealso: SVDSolve(), SVDGetConvergedReason(), SVDSetTolerances()
E*/
typedef enum {/* converged */
              SVD_CONVERGED_TOL                =  1,
              SVD_CONVERGED_USER               =  2,
              /* diverged */
              SVD_DIVERGED_ITS                 = -1,
              SVD_DIVERGED_BREAKDOWN           = -2,
              SVD_CONVERGED_ITERATING          =  0 } SVDConvergedReason;
PETSC_EXTERN const char *const*SVDConvergedReasons;

PETSC_EXTERN PetscErrorCode SVDCreate(MPI_Comm,SVD*);
PETSC_EXTERN PetscErrorCode SVDSetBV(SVD,BV,BV);
PETSC_EXTERN PetscErrorCode SVDGetBV(SVD,BV*,BV*);
PETSC_EXTERN PetscErrorCode SVDSetDS(SVD,DS);
PETSC_EXTERN PetscErrorCode SVDGetDS(SVD,DS*);
PETSC_EXTERN PetscErrorCode SVDSetType(SVD,SVDType);
PETSC_EXTERN PetscErrorCode SVDGetType(SVD,SVDType*);
PETSC_EXTERN PetscErrorCode SVDSetOperator(SVD,Mat);
PETSC_EXTERN PetscErrorCode SVDGetOperator(SVD,Mat*);
PETSC_EXTERN PetscErrorCode SVDSetInitialSpace(SVD,PetscInt,Vec*);
PETSC_EXTERN PetscErrorCode SVDSetInitialSpaceLeft(SVD,PetscInt,Vec*);
PETSC_EXTERN PetscErrorCode SVDSetImplicitTranspose(SVD,PetscBool);
PETSC_EXTERN PetscErrorCode SVDGetImplicitTranspose(SVD,PetscBool*);
PETSC_EXTERN PetscErrorCode SVDSetDimensions(SVD,PetscInt,PetscInt,PetscInt);
PETSC_EXTERN PetscErrorCode SVDGetDimensions(SVD,PetscInt*,PetscInt*,PetscInt*);
PETSC_EXTERN PetscErrorCode SVDSetTolerances(SVD,PetscReal,PetscInt);
PETSC_EXTERN PetscErrorCode SVDGetTolerances(SVD,PetscReal*,PetscInt*);
PETSC_EXTERN PetscErrorCode SVDSetWhichSingularTriplets(SVD,SVDWhich);
PETSC_EXTERN PetscErrorCode SVDGetWhichSingularTriplets(SVD,SVDWhich*);
PETSC_EXTERN PetscErrorCode SVDSetFromOptions(SVD);
PETSC_EXTERN PetscErrorCode SVDSetOptionsPrefix(SVD,const char*);
PETSC_EXTERN PetscErrorCode SVDAppendOptionsPrefix(SVD,const char*);
PETSC_EXTERN PetscErrorCode SVDGetOptionsPrefix(SVD,const char*[]);
PETSC_EXTERN PetscErrorCode SVDSetUp(SVD);
PETSC_EXTERN PetscErrorCode SVDSolve(SVD);
PETSC_EXTERN PetscErrorCode SVDGetIterationNumber(SVD,PetscInt*);
PETSC_EXTERN PetscErrorCode SVDSetConvergenceTestFunction(SVD,PetscErrorCode (*)(SVD,PetscReal,PetscReal,PetscReal*,void*),void*,PetscErrorCode (*)(void*));
PETSC_EXTERN PetscErrorCode SVDSetConvergenceTest(SVD,SVDConv);
PETSC_EXTERN PetscErrorCode SVDGetConvergenceTest(SVD,SVDConv*);
PETSC_EXTERN PetscErrorCode SVDConvergedAbsolute(SVD,PetscReal,PetscReal,PetscReal*,void*);
PETSC_EXTERN PetscErrorCode SVDConvergedRelative(SVD,PetscReal,PetscReal,PetscReal*,void*);
PETSC_EXTERN PetscErrorCode SVDSetStoppingTestFunction(SVD,PetscErrorCode (*)(SVD,PetscInt,PetscInt,PetscInt,PetscInt,SVDConvergedReason*,void*),void*,PetscErrorCode (*)(void*));
PETSC_EXTERN PetscErrorCode SVDSetStoppingTest(SVD,SVDStop);
PETSC_EXTERN PetscErrorCode SVDGetStoppingTest(SVD,SVDStop*);
PETSC_EXTERN PetscErrorCode SVDStoppingBasic(SVD,PetscInt,PetscInt,PetscInt,PetscInt,SVDConvergedReason*,void*);
PETSC_EXTERN PetscErrorCode SVDGetConvergedReason(SVD,SVDConvergedReason*);
PETSC_EXTERN PetscErrorCode SVDGetConverged(SVD,PetscInt*);
PETSC_EXTERN PetscErrorCode SVDGetSingularTriplet(SVD,PetscInt,PetscReal*,Vec,Vec);
PETSC_EXTERN PetscErrorCode SVDComputeError(SVD,PetscInt,SVDErrorType,PetscReal*);
PETSC_DEPRECATED("Use SVDComputeError()") PETSC_STATIC_INLINE PetscErrorCode SVDComputeRelativeError(SVD svd,PetscInt i,PetscReal *r) {return SVDComputeError(svd,i,SVD_ERROR_RELATIVE,r);}
PETSC_DEPRECATED("Use SVDComputeError() with SVD_ERROR_ABSOLUTE") PETSC_STATIC_INLINE PetscErrorCode SVDComputeResidualNorms(SVD svd,PetscInt i,PetscReal *r1,PetscReal *r2) {return SVDComputeError(svd,i,SVD_ERROR_ABSOLUTE,r1);}
PETSC_EXTERN PetscErrorCode SVDView(SVD,PetscViewer);
PETSC_STATIC_INLINE PetscErrorCode SVDViewFromOptions(SVD svd,PetscObject obj,const char name[]) {return PetscObjectViewFromOptions((PetscObject)svd,obj,name);}
PETSC_EXTERN PetscErrorCode SVDErrorView(SVD,SVDErrorType,PetscViewer);
PETSC_DEPRECATED("Use SVDErrorView()") PETSC_STATIC_INLINE PetscErrorCode SVDPrintSolution(SVD svd,PetscViewer v) {return SVDErrorView(svd,SVD_ERROR_RELATIVE,v);}
PETSC_EXTERN PetscErrorCode SVDErrorViewFromOptions(SVD);
PETSC_EXTERN PetscErrorCode SVDReasonView(SVD,PetscViewer);
PETSC_EXTERN PetscErrorCode SVDReasonViewFromOptions(SVD);
PETSC_EXTERN PetscErrorCode SVDValuesView(SVD,PetscViewer);
PETSC_EXTERN PetscErrorCode SVDValuesViewFromOptions(SVD);
PETSC_EXTERN PetscErrorCode SVDVectorsView(SVD,PetscViewer);
PETSC_EXTERN PetscErrorCode SVDVectorsViewFromOptions(SVD);
PETSC_EXTERN PetscErrorCode SVDDestroy(SVD*);
PETSC_EXTERN PetscErrorCode SVDReset(SVD);

PETSC_EXTERN PetscErrorCode SVDMonitor(SVD,PetscInt,PetscInt,PetscReal*,PetscReal*,PetscInt);
PETSC_EXTERN PetscErrorCode SVDMonitorSet(SVD,PetscErrorCode (*)(SVD,PetscInt,PetscInt,PetscReal*,PetscReal*,PetscInt,void*),void*,PetscErrorCode (*)(void**));
PETSC_EXTERN PetscErrorCode SVDMonitorSetFromOptions(SVD,const char*,const char*,const char*,PetscErrorCode (*)(SVD,PetscInt,PetscInt,PetscReal*,PetscReal*,PetscInt,PetscViewerAndFormat*),PetscBool);
PETSC_EXTERN PetscErrorCode SVDConvMonitorSetFromOptions(SVD,const char*,const char*,const char*,PetscErrorCode (*)(SVD,PetscInt,PetscInt,PetscReal*,PetscReal*,PetscInt,SlepcConvMonitor));
PETSC_EXTERN PetscErrorCode SVDMonitorCancel(SVD);
PETSC_EXTERN PetscErrorCode SVDGetMonitorContext(SVD,void **);
PETSC_EXTERN PetscErrorCode SVDMonitorAll(SVD,PetscInt,PetscInt,PetscReal*,PetscReal*,PetscInt,PetscViewerAndFormat*);
PETSC_EXTERN PetscErrorCode SVDMonitorFirst(SVD,PetscInt,PetscInt,PetscReal*,PetscReal*,PetscInt,PetscViewerAndFormat*);
PETSC_EXTERN PetscErrorCode SVDMonitorConverged(SVD,PetscInt,PetscInt,PetscReal*,PetscReal*,PetscInt,SlepcConvMonitor);
PETSC_EXTERN PetscErrorCode SVDMonitorLGCreate(MPI_Comm,const char[],const char[],int,int,int,int,PetscDrawLG*);
PETSC_EXTERN PetscErrorCode SVDMonitorLG(SVD,PetscInt,PetscInt,PetscReal*,PetscReal*,PetscInt,void*);
PETSC_EXTERN PetscErrorCode SVDMonitorLGAll(SVD,PetscInt,PetscInt,PetscReal*,PetscReal*,PetscInt,void*);

PETSC_EXTERN PetscErrorCode SVDSetTrackAll(SVD,PetscBool);
PETSC_EXTERN PetscErrorCode SVDGetTrackAll(SVD,PetscBool*);

PETSC_EXTERN PetscErrorCode SVDCrossSetEPS(SVD,EPS);
PETSC_EXTERN PetscErrorCode SVDCrossGetEPS(SVD,EPS*);

PETSC_EXTERN PetscErrorCode SVDCyclicSetExplicitMatrix(SVD,PetscBool);
PETSC_EXTERN PetscErrorCode SVDCyclicGetExplicitMatrix(SVD,PetscBool*);
PETSC_EXTERN PetscErrorCode SVDCyclicSetEPS(SVD,EPS);
PETSC_EXTERN PetscErrorCode SVDCyclicGetEPS(SVD,EPS*);

PETSC_EXTERN PetscErrorCode SVDLanczosSetOneSide(SVD,PetscBool);
PETSC_EXTERN PetscErrorCode SVDLanczosGetOneSide(SVD,PetscBool*);

PETSC_EXTERN PetscErrorCode SVDTRLanczosSetOneSide(SVD,PetscBool);
PETSC_EXTERN PetscErrorCode SVDTRLanczosGetOneSide(SVD,PetscBool*);

PETSC_EXTERN PetscFunctionList SVDList;
PETSC_EXTERN PetscErrorCode SVDRegister(const char[],PetscErrorCode(*)(SVD));

PETSC_EXTERN PetscErrorCode SVDAllocateSolution(SVD,PetscInt);

#endif
