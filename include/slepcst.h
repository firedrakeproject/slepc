/*
      Spectral transformation module for eigenvalue problems.  

   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
   SLEPc - Scalable Library for Eigenvalue Problem Computations
   Copyright (c) 2002-2011, Universitat Politecnica de Valencia, Spain

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
#include "petscksp.h"
PETSC_EXTERN_CXX_BEGIN

extern PetscErrorCode STInitializePackage(const char[]);

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
extern PetscClassId ST_CLASSID;

extern PetscErrorCode STCreate(MPI_Comm,ST*);
extern PetscErrorCode STDestroy(ST*);
extern PetscErrorCode STReset(ST);
extern PetscErrorCode STSetType(ST,const STType);
extern PetscErrorCode STGetType(ST,const STType*);
extern PetscErrorCode STSetOperators(ST,Mat,Mat);
extern PetscErrorCode STGetOperators(ST,Mat*,Mat*);
extern PetscErrorCode STSetUp(ST);
extern PetscErrorCode STSetFromOptions(ST);
extern PetscErrorCode STView(ST,PetscViewer);

extern PetscErrorCode STApply(ST,Vec,Vec);
extern PetscErrorCode STGetBilinearForm(ST,Mat*);
extern PetscErrorCode STApplyTranspose(ST,Vec,Vec);
extern PetscErrorCode STComputeExplicitOperator(ST,Mat*);
extern PetscErrorCode STPostSolve(ST);

extern PetscErrorCode STSetKSP(ST,KSP);
extern PetscErrorCode STGetKSP(ST,KSP*);
extern PetscErrorCode STSetShift(ST,PetscScalar);
extern PetscErrorCode STGetShift(ST,PetscScalar*);
extern PetscErrorCode STSetDefaultShift(ST,PetscScalar);
extern PetscErrorCode STSetBalanceMatrix(ST,Vec);
extern PetscErrorCode STGetBalanceMatrix(ST,Vec*);

extern PetscErrorCode STSetOptionsPrefix(ST,const char*);
extern PetscErrorCode STAppendOptionsPrefix(ST,const char*);
extern PetscErrorCode STGetOptionsPrefix(ST,const char*[]);

extern PetscErrorCode STBackTransform(ST,PetscInt,PetscScalar*,PetscScalar*);

extern PetscErrorCode STCheckNullSpace(ST,PetscInt,const Vec[]);

extern PetscErrorCode STGetOperationCounters(ST,PetscInt*,PetscInt*);
extern PetscErrorCode STResetOperationCounters(ST);

/*E
    STMatMode - determines how to handle the coefficient matrix associated
    to the spectral transformation

    Level: intermediate

.seealso: STSetMatMode(), STGetMatMode()
E*/
typedef enum { ST_MATMODE_COPY,
               ST_MATMODE_INPLACE, 
               ST_MATMODE_SHELL } STMatMode;
extern PetscErrorCode STSetMatMode(ST,STMatMode);
extern PetscErrorCode STGetMatMode(ST,STMatMode*);
extern PetscErrorCode STSetMatStructure(ST,MatStructure);
extern PetscErrorCode STGetMatStructure(ST,MatStructure*);

extern PetscFList STList;
extern PetscBool  STRegisterAllCalled;
extern PetscErrorCode STRegisterAll(const char[]);
extern PetscErrorCode STRegisterDestroy(void);
extern PetscErrorCode STRegister(const char[],const char[],const char[],PetscErrorCode(*)(ST));

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

extern PetscErrorCode STShellGetContext(ST st,void **ctx);
extern PetscErrorCode STShellSetContext(ST st,void *ctx);
extern PetscErrorCode STShellSetApply(ST st,PetscErrorCode (*apply)(ST,Vec,Vec));
extern PetscErrorCode STShellSetApplyTranspose(ST st,PetscErrorCode (*applytrans)(ST,Vec,Vec));
extern PetscErrorCode STShellSetBackTransform(ST st,PetscErrorCode (*backtr)(ST,PetscInt,PetscScalar*,PetscScalar*));

extern PetscErrorCode STCayleyGetAntishift(ST,PetscScalar*);
extern PetscErrorCode STCayleySetAntishift(ST,PetscScalar);

extern PetscErrorCode STPrecondGetMatForPC(ST st,Mat *mat);
extern PetscErrorCode STPrecondSetMatForPC(ST st,Mat mat);
extern PetscErrorCode STPrecondGetKSPHasMat(ST st,PetscBool *setmat);
extern PetscErrorCode STPrecondSetKSPHasMat(ST st,PetscBool setmat);

PETSC_EXTERN_CXX_END
#endif

