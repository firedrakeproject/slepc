/*
   User interface for SLEPc's nonlinear eigenvalue solvers. 

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

#if !defined(__SLEPCNEP_H)
#define __SLEPCNEP_H
#include "slepcsys.h"
#include "slepceps.h"

PETSC_EXTERN PetscErrorCode NEPInitializePackage(const char[]);

/*S
     NEP - Abstract SLEPc object that manages all solvers for
     nonlinear eigenvalue problems.

   Level: beginner

.seealso:  NEPCreate()
S*/
typedef struct _p_NEP* NEP;

/*J
    NEPType - String with the name of a nonlinear eigensolver

   Level: beginner

.seealso: NEPSetType(), NEP
J*/
typedef const char* NEPType;
#define NEPRII       "rii"
#define NEPNARNOLDI  "narnoldi"

/* Logging support */
PETSC_EXTERN PetscClassId NEP_CLASSID;

/*E
    NEPWhich - Determines which part of the spectrum is requested

    Level: intermediate

.seealso: NEPSetWhichEigenpairs(), NEPGetWhichEigenpairs()
E*/
typedef enum { NEP_LARGEST_MAGNITUDE=1,
               NEP_SMALLEST_MAGNITUDE,
               NEP_LARGEST_REAL,
               NEP_SMALLEST_REAL,
               NEP_LARGEST_IMAGINARY,
               NEP_SMALLEST_IMAGINARY,
               NEP_TARGET_MAGNITUDE,
               NEP_TARGET_REAL,
               NEP_TARGET_IMAGINARY} NEPWhich;

/*E
    NEPConvergedReason - Reason a nonlinear eigensolver was said to 
         have converged or diverged

    Level: beginner

.seealso: NEPSolve(), NEPGetConvergedReason(), NEPSetTolerances()
E*/
typedef enum {/* converged */
              NEP_CONVERGED_FNORM_ABS          =  2,
              NEP_CONVERGED_FNORM_RELATIVE     =  3,
              NEP_CONVERGED_SNORM_RELATIVE     =  4,
              /* diverged */
              NEP_DIVERGED_LINEAR_SOLVE        = -1,
              NEP_DIVERGED_FUNCTION_COUNT      = -2,
              NEP_DIVERGED_MAX_IT              = -3,
              NEP_DIVERGED_BREAKDOWN           = -4,
              NEP_DIVERGED_FNORM_NAN           = -5,
              NEP_CONVERGED_ITERATING          =  0} NEPConvergedReason;

PETSC_EXTERN PetscErrorCode NEPCreate(MPI_Comm,NEP*);
PETSC_EXTERN PetscErrorCode NEPDestroy(NEP*);
PETSC_EXTERN PetscErrorCode NEPReset(NEP);
PETSC_EXTERN PetscErrorCode NEPSetType(NEP,NEPType);
PETSC_EXTERN PetscErrorCode NEPGetType(NEP,NEPType*);
PETSC_EXTERN PetscErrorCode NEPSetTarget(NEP,PetscScalar);
PETSC_EXTERN PetscErrorCode NEPGetTarget(NEP,PetscScalar*);
PETSC_EXTERN PetscErrorCode NEPSetKSP(NEP,KSP);
PETSC_EXTERN PetscErrorCode NEPGetKSP(NEP,KSP*);
PETSC_EXTERN PetscErrorCode NEPSetFromOptions(NEP);
PETSC_EXTERN PetscErrorCode NEPSetUp(NEP);
PETSC_EXTERN PetscErrorCode NEPSolve(NEP);
PETSC_EXTERN PetscErrorCode NEPView(NEP,PetscViewer);

PETSC_EXTERN PetscErrorCode NEPSetFunction(NEP,Vec,PetscErrorCode (*)(NEP,Vec,Vec,void*),void*);
PETSC_EXTERN PetscErrorCode NEPGetFunction(NEP,Vec*,PetscErrorCode (**)(NEP,Vec,Vec,void*),void**);
PETSC_EXTERN PetscErrorCode NEPSetJacobian(NEP,Mat,Mat,PetscErrorCode (*)(NEP,Vec,Mat*,Mat*,MatStructure*,void*),void*);
PETSC_EXTERN PetscErrorCode NEPGetJacobian(NEP,Mat*,Mat*,PetscErrorCode (**)(NEP,Vec,Mat*,Mat*,MatStructure*,void*),void**);


PETSC_EXTERN PetscErrorCode NEPSetIP(NEP,IP);
PETSC_EXTERN PetscErrorCode NEPGetIP(NEP,IP*);
PETSC_EXTERN PetscErrorCode NEPSetDS(NEP,DS);
PETSC_EXTERN PetscErrorCode NEPGetDS(NEP,DS*);
PETSC_EXTERN PetscErrorCode NEPSetTolerances(NEP,PetscReal,PetscReal,PetscReal,PetscInt,PetscInt);
PETSC_EXTERN PetscErrorCode NEPGetTolerances(NEP,PetscReal*,PetscReal*,PetscReal*,PetscInt*,PetscInt*);
PETSC_EXTERN PetscErrorCode NEPSetConvergenceTest(NEP,PetscErrorCode (*)(NEP,PetscInt,PetscReal,PetscReal,PetscReal,NEPConvergedReason*,void*),void*,PetscErrorCode (*)(void*));
PETSC_EXTERN PetscErrorCode NEPConvergedDefault(NEP,PetscInt,PetscReal,PetscReal,PetscReal,NEPConvergedReason*,void*);
PETSC_EXTERN PetscErrorCode NEPSetDimensions(NEP,PetscInt,PetscInt,PetscInt);
PETSC_EXTERN PetscErrorCode NEPGetDimensions(NEP,PetscInt*,PetscInt*,PetscInt*);

PETSC_EXTERN PetscErrorCode NEPGetConverged(NEP,PetscInt*);
PETSC_EXTERN PetscErrorCode NEPGetEigenpair(NEP,PetscInt,PetscScalar*,PetscScalar*,Vec,Vec);

PETSC_EXTERN PetscErrorCode NEPMonitorSet(NEP,PetscErrorCode (*)(NEP,PetscInt,PetscInt,PetscScalar*,PetscScalar*,PetscReal*,PetscInt,void*),void*,PetscErrorCode (*)(void**));
PETSC_EXTERN PetscErrorCode NEPMonitorCancel(NEP);
PETSC_EXTERN PetscErrorCode NEPGetMonitorContext(NEP,void **);
PETSC_EXTERN PetscErrorCode NEPGetIterationNumber(NEP,PetscInt*);
PETSC_EXTERN PetscErrorCode NEPGetOperationCounters(NEP,PetscInt*,PetscInt*,PetscInt*);

PETSC_EXTERN PetscErrorCode NEPSetInitialSpace(NEP,PetscInt,Vec*);
PETSC_EXTERN PetscErrorCode NEPSetWhichEigenpairs(NEP,NEPWhich);
PETSC_EXTERN PetscErrorCode NEPGetWhichEigenpairs(NEP,NEPWhich*);
PETSC_EXTERN PetscErrorCode NEPSetEigenvalueComparison(NEP,PetscErrorCode (*func)(NEP,PetscScalar,PetscScalar,PetscScalar,PetscScalar,PetscInt*,void*),void*);

PETSC_EXTERN PetscErrorCode NEPMonitorAll(NEP,PetscInt,PetscInt,PetscScalar*,PetscScalar*,PetscReal*,PetscInt,void*);
PETSC_EXTERN PetscErrorCode NEPMonitorFirst(NEP,PetscInt,PetscInt,PetscScalar*,PetscScalar*,PetscReal*,PetscInt,void*);
PETSC_EXTERN PetscErrorCode NEPMonitorConverged(NEP,PetscInt,PetscInt,PetscScalar*,PetscScalar*,PetscReal*,PetscInt,void*);
PETSC_EXTERN PetscErrorCode NEPMonitorLG(NEP,PetscInt,PetscInt,PetscScalar*,PetscScalar*,PetscReal*,PetscInt,void*);
PETSC_EXTERN PetscErrorCode NEPMonitorLGAll(NEP,PetscInt,PetscInt,PetscScalar*,PetscScalar*,PetscReal*,PetscInt,void*);

PETSC_EXTERN PetscErrorCode NEPSetTrackAll(NEP,PetscBool);
PETSC_EXTERN PetscErrorCode NEPGetTrackAll(NEP,PetscBool*);

PETSC_EXTERN PetscErrorCode NEPSetOptionsPrefix(NEP,const char*);
PETSC_EXTERN PetscErrorCode NEPAppendOptionsPrefix(NEP,const char*);
PETSC_EXTERN PetscErrorCode NEPGetOptionsPrefix(NEP,const char*[]);

PETSC_EXTERN PetscErrorCode NEPGetConvergedReason(NEP,NEPConvergedReason *);

PETSC_EXTERN PetscErrorCode NEPSortEigenvalues(NEP,PetscInt,PetscScalar*,PetscScalar*,PetscInt*);
PETSC_EXTERN PetscErrorCode NEPCompareEigenvalues(NEP,PetscScalar,PetscScalar,PetscScalar,PetscScalar,PetscInt*);

PETSC_EXTERN PetscFunctionList NEPList;
PETSC_EXTERN PetscBool         NEPRegisterAllCalled;
PETSC_EXTERN PetscErrorCode NEPRegisterAll(const char[]);
PETSC_EXTERN PetscErrorCode NEPRegisterDestroy(void);
PETSC_EXTERN PetscErrorCode NEPRegister(const char[],const char[],const char[],PetscErrorCode(*)(NEP));

/*MC
   NEPRegisterDynamic - Adds a method to the quadratic eigenproblem solver package.

   Synopsis:
   PetscErrorCode NEPRegisterDynamic(const char *name_solver,const char *path,const char *name_create,PetscErrorCode (*routine_create)(NEP))

   Not Collective

   Input Parameters:
+  name_solver - name of a new user-defined solver
.  path - path (either absolute or relative) the library containing this solver
.  name_create - name of routine to create the solver context
-  routine_create - routine to create the solver context

   Notes:
   NEPRegisterDynamic() may be called multiple times to add several user-defined solvers.

   If dynamic libraries are used, then the fourth input argument (routine_create)
   is ignored.

   Sample usage:
.vb
   NEPRegisterDynamic("my_solver",/home/username/my_lib/lib/libO/solaris/mylib.a,
               "MySolverCreate",MySolverCreate);
.ve

   Then, your solver can be chosen with the procedural interface via
$     NEPSetType(qep,"my_solver")
   or at runtime via the option
$     -qep_type my_solver

   Level: advanced

.seealso: NEPRegisterDestroy(), NEPRegisterAll()
M*/
#if defined(PETSC_USE_DYNAMIC_LIBRARIES)
#define NEPRegisterDynamic(a,b,c,d) NEPRegister(a,b,c,0)
#else
#define NEPRegisterDynamic(a,b,c,d) NEPRegister(a,b,c,d)
#endif

/* --------- options specific to particular eigensolvers -------- */


#endif

