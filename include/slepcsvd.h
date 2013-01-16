/*
   User interface for the SLEPC singular value solvers. 

   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
   SLEPc - Scalable Library for Eigenvalue Problem Computations
   Copyright (c) 2002-2012, Universitat Politecnica de Valencia, Spain

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
#include "slepcsys.h"
#include "slepceps.h"

PETSC_EXTERN PetscErrorCode SVDInitializePackage(const char[]);

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
    SVDTransposeMode - Determines how to handle the transpose of the matrix

    Level: advanced

.seealso: SVDSetTransposeMode(), SVDGetTransposeMode()
E*/
typedef enum { SVD_TRANSPOSE_EXPLICIT,
               SVD_TRANSPOSE_IMPLICIT } SVDTransposeMode;

/*E
    SVDWhich - Determines whether largest or smallest singular triplets
    are to be computed

    Level: intermediate

.seealso: SVDSetWhichSingularTriplets(), SVDGetWhichSingularTriplets()
E*/
typedef enum { SVD_LARGEST,
               SVD_SMALLEST } SVDWhich;

/*E
    SVDConvergedReason - Reason a singular value solver was said to 
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

PETSC_EXTERN PetscErrorCode SVDCreate(MPI_Comm,SVD*);
PETSC_EXTERN PetscErrorCode SVDSetIP(SVD,IP);
PETSC_EXTERN PetscErrorCode SVDGetIP(SVD,IP*);
PETSC_EXTERN PetscErrorCode SVDSetDS(SVD,DS);
PETSC_EXTERN PetscErrorCode SVDGetDS(SVD,DS*);
PETSC_EXTERN PetscErrorCode SVDSetType(SVD,SVDType);
PETSC_EXTERN PetscErrorCode SVDGetType(SVD,SVDType*);
PETSC_EXTERN PetscErrorCode SVDSetOperator(SVD,Mat);
PETSC_EXTERN PetscErrorCode SVDGetOperator(SVD,Mat*);
PETSC_EXTERN PetscErrorCode SVDSetInitialSpace(SVD,PetscInt,Vec*);
PETSC_EXTERN PetscErrorCode SVDSetTransposeMode(SVD,SVDTransposeMode);
PETSC_EXTERN PetscErrorCode SVDGetTransposeMode(SVD,SVDTransposeMode*);
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
PETSC_EXTERN PetscErrorCode SVDGetConvergedReason(SVD,SVDConvergedReason*);
PETSC_EXTERN PetscErrorCode SVDGetConverged(SVD,PetscInt*);
PETSC_EXTERN PetscErrorCode SVDGetSingularTriplet(SVD,PetscInt,PetscReal*,Vec,Vec);
PETSC_EXTERN PetscErrorCode SVDComputeResidualNorms(SVD,PetscInt,PetscReal*,PetscReal*);
PETSC_EXTERN PetscErrorCode SVDComputeRelativeError(SVD,PetscInt,PetscReal*);
PETSC_EXTERN PetscErrorCode SVDGetOperationCounters(SVD,PetscInt*,PetscInt*);
PETSC_EXTERN PetscErrorCode SVDView(SVD,PetscViewer);
PETSC_EXTERN PetscErrorCode SVDPrintSolution(SVD,PetscViewer);
PETSC_EXTERN PetscErrorCode SVDDestroy(SVD*);
PETSC_EXTERN PetscErrorCode SVDReset(SVD);

PETSC_EXTERN PetscErrorCode SVDMonitorSet(SVD,PetscErrorCode (*)(SVD,PetscInt,PetscInt,PetscReal*,PetscReal*,PetscInt,void*),void*,PetscErrorCode (*)(void**));
PETSC_EXTERN PetscErrorCode SVDMonitorCancel(SVD);
PETSC_EXTERN PetscErrorCode SVDGetMonitorContext(SVD,void **);
PETSC_EXTERN PetscErrorCode SVDMonitorAll(SVD,PetscInt,PetscInt,PetscReal*,PetscReal*,PetscInt,void*);
PETSC_EXTERN PetscErrorCode SVDMonitorFirst(SVD,PetscInt,PetscInt,PetscReal*,PetscReal*,PetscInt,void*);
PETSC_EXTERN PetscErrorCode SVDMonitorConverged(SVD,PetscInt,PetscInt,PetscReal*,PetscReal*,PetscInt,void*);
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
PETSC_EXTERN PetscBool         SVDRegisterAllCalled;
PETSC_EXTERN PetscErrorCode SVDRegisterAll(const char[]);
PETSC_EXTERN PetscErrorCode SVDRegisterDestroy(void);
PETSC_EXTERN PetscErrorCode SVDRegister(const char[],const char[],const char[],PetscErrorCode(*)(SVD));

/*MC
   SVDRegisterDynamic - Adds a method to the singular value solver package.

   Synopsis:
   PetscErrorCode SVDRegisterDynamic(const char *name_solver,const char *path,const char *name_create,PetscErrorCode (*routine_create)(SVD))

   Not Collective

   Input Parameters:
+  name_solver - name of a new user-defined solver
.  path - path (either absolute or relative) the library containing this solver
.  name_create - name of routine to create the solver context
-  routine_create - routine to create the solver context

   Notes:
   SVDRegisterDynamic() may be called multiple times to add several user-defined solvers.

   If dynamic libraries are used, then the fourth input argument (routine_create)
   is ignored.

   Sample usage:
.vb
   SVDRegisterDynamic("my_solver",/home/username/my_lib/lib/libO/solaris/mylib.a,
               "MySolverCreate",MySolverCreate);
.ve

   Then, your solver can be chosen with the procedural interface via
$     SVDSetType(svd,"my_solver")
   or at runtime via the option
$     -svd_type my_solver

   Level: advanced

.seealso: SVDRegisterDestroy(), SVDRegisterAll()
M*/
#if defined(PETSC_USE_DYNAMIC_LIBRARIES)
#define SVDRegisterDynamic(a,b,c,d) SVDRegister(a,b,c,0)
#else
#define SVDRegisterDynamic(a,b,c,d) SVDRegister(a,b,c,d)
#endif

#endif
