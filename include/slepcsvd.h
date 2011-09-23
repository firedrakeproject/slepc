/*
   User interface for the SLEPC singular value solvers. 

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

#if !defined(__SLEPCSVD_H)
#define __SLEPCSVD_H
#include "slepcsys.h"
#include "slepceps.h"
PETSC_EXTERN_CXX_BEGIN

extern PetscErrorCode SVDInitializePackage(const char[]);

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
#define SVDType        char*
#define SVDCROSS       "cross"
#define SVDCYCLIC      "cyclic"
#define SVDLAPACK      "lapack"
#define SVDLANCZOS     "lanczos"
#define SVDTRLANCZOS   "trlanczos"

/* Logging support */
extern PetscClassId SVD_CLASSID;

/*E
    SVDTransposeMode - determines how to handle the transpose of the matrix

    Level: advanced

.seealso: SVDSetTransposeMode(), SVDGetTransposeMode()
E*/
typedef enum { SVD_TRANSPOSE_EXPLICIT,
               SVD_TRANSPOSE_IMPLICIT } SVDTransposeMode;

/*E
    SVDWhich - determines whether largest or smallest singular triplets
    are to be computed

    Level: intermediate

.seealso: SVDSetWhichSingularTriplets(), SVDGetWhichSingularTriplets()
E*/
typedef enum { SVD_LARGEST,
               SVD_SMALLEST } SVDWhich;

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

extern PetscErrorCode SVDCreate(MPI_Comm,SVD*);
extern PetscErrorCode SVDSetIP(SVD,IP);
extern PetscErrorCode SVDGetIP(SVD,IP*);
extern PetscErrorCode SVDSetType(SVD,const SVDType);
extern PetscErrorCode SVDGetType(SVD,const SVDType*);
extern PetscErrorCode SVDSetOperator(SVD,Mat);
extern PetscErrorCode SVDGetOperator(SVD,Mat*);
extern PetscErrorCode SVDSetInitialSpace(SVD,PetscInt,Vec*);
extern PetscErrorCode SVDSetTransposeMode(SVD,SVDTransposeMode);
extern PetscErrorCode SVDGetTransposeMode(SVD,SVDTransposeMode*);
extern PetscErrorCode SVDSetDimensions(SVD,PetscInt,PetscInt,PetscInt);
extern PetscErrorCode SVDGetDimensions(SVD,PetscInt*,PetscInt*,PetscInt*);
extern PetscErrorCode SVDSetTolerances(SVD,PetscReal,PetscInt);
extern PetscErrorCode SVDGetTolerances(SVD,PetscReal*,PetscInt*);
extern PetscErrorCode SVDSetWhichSingularTriplets(SVD,SVDWhich);
extern PetscErrorCode SVDGetWhichSingularTriplets(SVD,SVDWhich*);
extern PetscErrorCode SVDSetFromOptions(SVD);
extern PetscErrorCode SVDSetOptionsPrefix(SVD,const char*);
extern PetscErrorCode SVDAppendOptionsPrefix(SVD,const char*);
extern PetscErrorCode SVDGetOptionsPrefix(SVD,const char*[]);
extern PetscErrorCode SVDSetUp(SVD);
extern PetscErrorCode SVDSolve(SVD);
extern PetscErrorCode SVDGetIterationNumber(SVD,PetscInt*);
extern PetscErrorCode SVDGetConvergedReason(SVD,SVDConvergedReason*);
extern PetscErrorCode SVDGetConverged(SVD,PetscInt*);
extern PetscErrorCode SVDGetSingularTriplet(SVD,PetscInt,PetscReal*,Vec,Vec);
extern PetscErrorCode SVDComputeResidualNorms(SVD,PetscInt,PetscReal*,PetscReal*);
extern PetscErrorCode SVDComputeRelativeError(SVD,PetscInt,PetscReal*);
extern PetscErrorCode SVDGetOperationCounters(SVD,PetscInt*,PetscInt*);
extern PetscErrorCode SVDView(SVD,PetscViewer);
extern PetscErrorCode SVDPrintSolution(SVD,PetscViewer);
extern PetscErrorCode SVDDestroy(SVD*);
extern PetscErrorCode SVDReset(SVD);

extern PetscErrorCode SVDMonitorSet(SVD,PetscErrorCode (*)(SVD,PetscInt,PetscInt,PetscReal*,PetscReal*,PetscInt,void*),void*,PetscErrorCode (*)(void**));
extern PetscErrorCode SVDMonitorCancel(SVD);
extern PetscErrorCode SVDGetMonitorContext(SVD,void **);
extern PetscErrorCode SVDMonitorAll(SVD,PetscInt,PetscInt,PetscReal*,PetscReal*,PetscInt,void*);
extern PetscErrorCode SVDMonitorFirst(SVD,PetscInt,PetscInt,PetscReal*,PetscReal*,PetscInt,void*);
extern PetscErrorCode SVDMonitorConverged(SVD,PetscInt,PetscInt,PetscReal*,PetscReal*,PetscInt,void*);
extern PetscErrorCode SVDMonitorLG(SVD,PetscInt,PetscInt,PetscReal*,PetscReal*,PetscInt,void*);
extern PetscErrorCode SVDMonitorLGAll(SVD,PetscInt,PetscInt,PetscReal*,PetscReal*,PetscInt,void*);

extern PetscErrorCode SVDSetTrackAll(SVD,PetscBool);
extern PetscErrorCode SVDGetTrackAll(SVD,PetscBool*);

extern PetscErrorCode SVDDense(PetscInt,PetscInt,PetscScalar*,PetscReal*,PetscScalar*,PetscScalar*);

extern PetscErrorCode SVDCrossSetEPS(SVD,EPS);
extern PetscErrorCode SVDCrossGetEPS(SVD,EPS*);

extern PetscErrorCode SVDCyclicSetExplicitMatrix(SVD,PetscBool);
extern PetscErrorCode SVDCyclicGetExplicitMatrix(SVD,PetscBool*);
extern PetscErrorCode SVDCyclicSetEPS(SVD,EPS);
extern PetscErrorCode SVDCyclicGetEPS(SVD,EPS*);

extern PetscErrorCode SVDLanczosSetOneSide(SVD,PetscBool);
extern PetscErrorCode SVDLanczosGetOneSide(SVD,PetscBool*);

extern PetscErrorCode SVDTRLanczosSetOneSide(SVD,PetscBool);
extern PetscErrorCode SVDTRLanczosGetOneSide(SVD,PetscBool*);

extern PetscFList SVDList;
extern PetscBool  SVDRegisterAllCalled;
extern PetscErrorCode SVDRegisterAll(const char[]);
extern PetscErrorCode SVDRegisterDestroy(void);
extern PetscErrorCode SVDRegister(const char[],const char[],const char[],PetscErrorCode(*)(SVD));

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

PETSC_EXTERN_CXX_END
#endif
