/*
      Spectral transformation module for eigenvalue problems.  

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

#if !defined(__SLEPCST_H)
#define __SLEPCST_H
#include "slepcsys.h"
#include "petscksp.h"

PETSC_EXTERN PetscErrorCode STInitializePackage(const char[]);

/*S
    ST - Abstract SLEPc object that manages spectral transformations.
    This object is accessed only in advanced applications.

    Level: beginner

.seealso:  STCreate(), EPS
S*/
typedef struct _p_ST* ST;

/*E
    STType - String with the name of a SLEPc spectral transformation

    Level: beginner

.seealso: STSetType(), ST
E*/
#define STType      char*
#define STSHELL     "shell"
#define STSHIFT     "shift"
#define STSINVERT   "sinvert"
#define STCAYLEY    "cayley"
#define STFOLD      "fold"
#define STPRECOND   "precond"

/* Logging support */
PETSC_EXTERN PetscClassId ST_CLASSID;

PETSC_EXTERN PetscErrorCode STCreate(MPI_Comm,ST*);
PETSC_EXTERN PetscErrorCode STDestroy(ST*);
PETSC_EXTERN PetscErrorCode STReset(ST);
PETSC_EXTERN PetscErrorCode STSetType(ST,const STType);
PETSC_EXTERN PetscErrorCode STGetType(ST,const STType*);
PETSC_EXTERN PetscErrorCode STSetOperators(ST,PetscInt,Mat*);
PETSC_EXTERN PetscErrorCode STGetOperators(ST,PetscInt,Mat*);
PETSC_EXTERN PetscErrorCode STGetNumMatrices(ST,PetscInt*);
PETSC_EXTERN PetscErrorCode STSetUp(ST);
PETSC_EXTERN PetscErrorCode STSetFromOptions(ST);
PETSC_EXTERN PetscErrorCode STView(ST,PetscViewer);

PETSC_EXTERN PetscErrorCode STApply(ST,Vec,Vec);
PETSC_EXTERN PetscErrorCode STMatMult(ST,PetscInt,Vec,Vec);
PETSC_EXTERN PetscErrorCode STMatMultTranspose(ST,PetscInt,Vec,Vec);
PETSC_EXTERN PetscErrorCode STMatSolve(ST,PetscInt,Vec,Vec);
PETSC_EXTERN PetscErrorCode STMatSolveTranspose(ST,PetscInt,Vec,Vec);
PETSC_EXTERN PetscErrorCode STGetBilinearForm(ST,Mat*);
PETSC_EXTERN PetscErrorCode STApplyTranspose(ST,Vec,Vec);
PETSC_EXTERN PetscErrorCode STComputeExplicitOperator(ST,Mat*);
PETSC_EXTERN PetscErrorCode STPostSolve(ST);

PETSC_EXTERN PetscErrorCode STSetKSP(ST,KSP);
PETSC_EXTERN PetscErrorCode STGetKSP(ST,KSP*);
PETSC_EXTERN PetscErrorCode STSetShift(ST,PetscScalar);
PETSC_EXTERN PetscErrorCode STGetShift(ST,PetscScalar*);
PETSC_EXTERN PetscErrorCode STSetDefaultShift(ST,PetscScalar);
PETSC_EXTERN PetscErrorCode STSetBalanceMatrix(ST,Vec);
PETSC_EXTERN PetscErrorCode STGetBalanceMatrix(ST,Vec*);

PETSC_EXTERN PetscErrorCode STSetOptionsPrefix(ST,const char*);
PETSC_EXTERN PetscErrorCode STAppendOptionsPrefix(ST,const char*);
PETSC_EXTERN PetscErrorCode STGetOptionsPrefix(ST,const char*[]);

PETSC_EXTERN PetscErrorCode STBackTransform(ST,PetscInt,PetscScalar*,PetscScalar*);

PETSC_EXTERN PetscErrorCode STCheckNullSpace(ST,PetscInt,const Vec[]);

PETSC_EXTERN PetscErrorCode STGetOperationCounters(ST,PetscInt*,PetscInt*);
PETSC_EXTERN PetscErrorCode STResetOperationCounters(ST);

/*E
    STMatMode - determines how to handle the coefficient matrix associated
    to the spectral transformation

    Level: intermediate

.seealso: STSetMatMode(), STGetMatMode()
E*/
typedef enum { ST_MATMODE_COPY,
               ST_MATMODE_INPLACE, 
               ST_MATMODE_SHELL } STMatMode;
PETSC_EXTERN PetscErrorCode STSetMatMode(ST,STMatMode);
PETSC_EXTERN PetscErrorCode STGetMatMode(ST,STMatMode*);
PETSC_EXTERN PetscErrorCode STSetMatStructure(ST,MatStructure);
PETSC_EXTERN PetscErrorCode STGetMatStructure(ST,MatStructure*);

PETSC_EXTERN PetscFList STList;
PETSC_EXTERN PetscBool  STRegisterAllCalled;
PETSC_EXTERN PetscErrorCode STRegisterAll(const char[]);
PETSC_EXTERN PetscErrorCode STRegisterDestroy(void);
PETSC_EXTERN PetscErrorCode STRegister(const char[],const char[],const char[],PetscErrorCode(*)(ST));

/*MC
   STRegisterDynamic - Adds a method to the spectral transformation package.

   Synopsis:
   PetscErrorCode STRegisterDynamic(const char *name,const char *path,const char *name_create,PetscErrorCode (*routine_create)(ST))

   Not collective

   Input Parameters:
+  name - name of a new user-defined transformation
.  path - path (either absolute or relative) the library containing this solver
.  name_create - name of routine to create method context
-  routine_create - routine to create method context

   Notes:
   STRegisterDynamic() may be called multiple times to add several user-defined
   spectral transformations.

   If dynamic libraries are used, then the fourth input argument (routine_create)
   is ignored.

   Sample usage:
.vb
   STRegisterDynamic("my_solver","/home/username/my_lib/lib/libO/solaris/mylib.a",
              "MySolverCreate",MySolverCreate);
.ve

   Then, your solver can be chosen with the procedural interface via
$     STSetType(st,"my_solver")
   or at runtime via the option
$     -st_type my_solver

   Level: advanced

.seealso: STRegisterDestroy(), STRegisterAll()
M*/
#if defined(PETSC_USE_DYNAMIC_LIBRARIES)
#define STRegisterDynamic(a,b,c,d) STRegister(a,b,c,0)
#else
#define STRegisterDynamic(a,b,c,d) STRegister(a,b,c,d)
#endif

/* --------- options specific to particular spectral transformations-------- */

PETSC_EXTERN PetscErrorCode STShellGetContext(ST st,void **ctx);
PETSC_EXTERN PetscErrorCode STShellSetContext(ST st,void *ctx);
PETSC_EXTERN PetscErrorCode STShellSetApply(ST st,PetscErrorCode (*apply)(ST,Vec,Vec));
PETSC_EXTERN PetscErrorCode STShellSetApplyTranspose(ST st,PetscErrorCode (*applytrans)(ST,Vec,Vec));
PETSC_EXTERN PetscErrorCode STShellSetBackTransform(ST st,PetscErrorCode (*backtr)(ST,PetscInt,PetscScalar*,PetscScalar*));

PETSC_EXTERN PetscErrorCode STCayleyGetAntishift(ST,PetscScalar*);
PETSC_EXTERN PetscErrorCode STCayleySetAntishift(ST,PetscScalar);

PETSC_EXTERN PetscErrorCode STPrecondGetMatForPC(ST st,Mat *mat);
PETSC_EXTERN PetscErrorCode STPrecondSetMatForPC(ST st,Mat mat);
PETSC_EXTERN PetscErrorCode STPrecondGetKSPHasMat(ST st,PetscBool *setmat);
PETSC_EXTERN PetscErrorCode STPrecondSetKSPHasMat(ST st,PetscBool setmat);

#endif

