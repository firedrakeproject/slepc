/*
   Spectral transformation module for eigenvalue problems.

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

#if !defined(__SLEPCST_H)
#define __SLEPCST_H
#include <slepcsys.h>
#include <slepcbv.h>
#include <petscksp.h>

PETSC_EXTERN PetscErrorCode STInitializePackage(void);

/*S
    ST - Abstract SLEPc object that manages spectral transformations.
    This object is accessed only in advanced applications.

    Level: beginner

.seealso:  STCreate(), EPS
S*/
typedef struct _p_ST* ST;

/*J
    STType - String with the name of a SLEPc spectral transformation

    Level: beginner

.seealso: STSetType(), ST
J*/
typedef const char* STType;
#define STSHELL     "shell"
#define STSHIFT     "shift"
#define STSINVERT   "sinvert"
#define STCAYLEY    "cayley"
#define STPRECOND   "precond"

/* Logging support */
PETSC_EXTERN PetscClassId ST_CLASSID;

PETSC_EXTERN PetscErrorCode STCreate(MPI_Comm,ST*);
PETSC_EXTERN PetscErrorCode STDestroy(ST*);
PETSC_EXTERN PetscErrorCode STReset(ST);
PETSC_EXTERN PetscErrorCode STSetType(ST,STType);
PETSC_EXTERN PetscErrorCode STGetType(ST,STType*);
PETSC_EXTERN PetscErrorCode STSetOperators(ST,PetscInt,Mat*);
PETSC_EXTERN PetscErrorCode STGetOperators(ST,PetscInt,Mat*);
PETSC_EXTERN PetscErrorCode STGetTOperators(ST,PetscInt,Mat*);
PETSC_EXTERN PetscErrorCode STGetNumMatrices(ST,PetscInt*);
PETSC_EXTERN PetscErrorCode STSetUp(ST);
PETSC_EXTERN PetscErrorCode STSetFromOptions(ST);
PETSC_EXTERN PetscErrorCode STView(ST,PetscViewer);

PETSC_EXTERN PetscErrorCode STApply(ST,Vec,Vec);
PETSC_EXTERN PetscErrorCode STMatMult(ST,PetscInt,Vec,Vec);
PETSC_EXTERN PetscErrorCode STMatMultTranspose(ST,PetscInt,Vec,Vec);
PETSC_EXTERN PetscErrorCode STMatSolve(ST,Vec,Vec);
PETSC_EXTERN PetscErrorCode STMatSolveTranspose(ST,Vec,Vec);
PETSC_EXTERN PetscErrorCode STGetBilinearForm(ST,Mat*);
PETSC_EXTERN PetscErrorCode STApplyTranspose(ST,Vec,Vec);
PETSC_EXTERN PetscErrorCode STComputeExplicitOperator(ST,Mat*);
PETSC_EXTERN PetscErrorCode STMatSetUp(ST,PetscScalar,PetscScalar*);
PETSC_EXTERN PetscErrorCode STPostSolve(ST);

PETSC_EXTERN PetscErrorCode STSetKSP(ST,KSP);
PETSC_EXTERN PetscErrorCode STGetKSP(ST,KSP*);
PETSC_EXTERN PetscErrorCode STSetShift(ST,PetscScalar);
PETSC_EXTERN PetscErrorCode STGetShift(ST,PetscScalar*);
PETSC_EXTERN PetscErrorCode STSetDefaultShift(ST,PetscScalar);
PETSC_EXTERN PetscErrorCode STScaleShift(ST,PetscScalar);
PETSC_EXTERN PetscErrorCode STSetBalanceMatrix(ST,Vec);
PETSC_EXTERN PetscErrorCode STGetBalanceMatrix(ST,Vec*);
PETSC_EXTERN PetscErrorCode STSetTransform(ST,PetscBool);
PETSC_EXTERN PetscErrorCode STGetTransform(ST,PetscBool*);

PETSC_EXTERN PetscErrorCode STSetOptionsPrefix(ST,const char*);
PETSC_EXTERN PetscErrorCode STAppendOptionsPrefix(ST,const char*);
PETSC_EXTERN PetscErrorCode STGetOptionsPrefix(ST,const char*[]);

PETSC_EXTERN PetscErrorCode STBackTransform(ST,PetscInt,PetscScalar*,PetscScalar*);

PETSC_EXTERN PetscErrorCode STCheckNullSpace(ST,BV);

PETSC_EXTERN PetscErrorCode STMatCreateVecs(ST,Vec*,Vec*);
PETSC_EXTERN PetscErrorCode STMatGetSize(ST,PetscInt*,PetscInt*);
PETSC_EXTERN PetscErrorCode STMatGetLocalSize(ST,PetscInt*,PetscInt*);

/*E
    STMatMode - Determines how to handle the coefficient matrix associated
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

PETSC_EXTERN PetscFunctionList STList;
PETSC_EXTERN PetscErrorCode STRegister(const char[],PetscErrorCode(*)(ST));

/* --------- options specific to particular spectral transformations-------- */

PETSC_EXTERN PetscErrorCode STShellGetContext(ST st,void **ctx);
PETSC_EXTERN PetscErrorCode STShellSetContext(ST st,void *ctx);
PETSC_EXTERN PetscErrorCode STShellSetApply(ST st,PetscErrorCode (*apply)(ST,Vec,Vec));
PETSC_EXTERN PetscErrorCode STShellSetApplyTranspose(ST st,PetscErrorCode (*applytrans)(ST,Vec,Vec));
PETSC_EXTERN PetscErrorCode STShellSetBackTransform(ST st,PetscErrorCode (*backtr)(ST,PetscInt,PetscScalar*,PetscScalar*));

PETSC_EXTERN PetscErrorCode STCayleyGetAntishift(ST,PetscScalar*);
PETSC_EXTERN PetscErrorCode STCayleySetAntishift(ST,PetscScalar);

PETSC_EXTERN PetscErrorCode STPrecondGetMatForPC(ST,Mat*);
PETSC_EXTERN PetscErrorCode STPrecondSetMatForPC(ST,Mat);
PETSC_EXTERN PetscErrorCode STPrecondGetKSPHasMat(ST,PetscBool*);
PETSC_EXTERN PetscErrorCode STPrecondSetKSPHasMat(ST,PetscBool);

#endif

