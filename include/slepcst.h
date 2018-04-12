/*
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
   SLEPc - Scalable Library for Eigenvalue Problem Computations
   Copyright (c) 2002-2018, Universitat Politecnica de Valencia, Spain

   This file is part of SLEPc.
   SLEPc is distributed under a 2-clause BSD license (see LICENSE).
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
*/
/*
   Spectral transformation module for eigenvalue problems
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
#define STFILTER    "filter"

/* Logging support */
PETSC_EXTERN PetscClassId ST_CLASSID;

PETSC_EXTERN PetscErrorCode STCreate(MPI_Comm,ST*);
PETSC_EXTERN PetscErrorCode STDestroy(ST*);
PETSC_EXTERN PetscErrorCode STReset(ST);
PETSC_EXTERN PetscErrorCode STSetType(ST,STType);
PETSC_EXTERN PetscErrorCode STGetType(ST,STType*);
PETSC_EXTERN PetscErrorCode STSetMatrices(ST,PetscInt,Mat*);
PETSC_EXTERN PetscErrorCode STGetMatrix(ST,PetscInt,Mat*);
PETSC_EXTERN PetscErrorCode STGetMatrixTransformed(ST,PetscInt,Mat*);
PETSC_EXTERN PetscErrorCode STGetNumMatrices(ST,PetscInt*);
PETSC_EXTERN PetscErrorCode STGetOperator(ST,Mat*);
PETSC_EXTERN PetscErrorCode STSetUp(ST);
PETSC_EXTERN PetscErrorCode STSetFromOptions(ST);
PETSC_EXTERN PetscErrorCode STView(ST,PetscViewer);

PETSC_DEPRECATED("Use STSetMatrices()") PETSC_STATIC_INLINE PetscErrorCode STSetOperators(ST st,PetscInt n,Mat *A) {return STSetMatrices(st,n,A);}
PETSC_DEPRECATED("Use STGetMatrix()") PETSC_STATIC_INLINE PetscErrorCode STGetOperators(ST st,PetscInt k,Mat *A) {return STGetMatrix(st,k,A);}
PETSC_DEPRECATED("Use STGetMatrixTransformed()") PETSC_STATIC_INLINE PetscErrorCode STGetTOperators(ST st,PetscInt k,Mat *A) {return STGetMatrixTransformed(st,k,A);}
PETSC_DEPRECATED("Use STGetOperator() followed by MatComputeExplicitOperator()") PETSC_STATIC_INLINE PetscErrorCode STComputeExplicitOperator(ST st,Mat *A) {
  PetscErrorCode ierr; Mat Op; 
  ierr = STGetOperator(st,&Op);CHKERRQ(ierr);
  ierr = MatComputeExplicitOperator(Op,A);CHKERRQ(ierr);
  ierr = MatDestroy(&Op);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PETSC_EXTERN PetscErrorCode STApply(ST,Vec,Vec);
PETSC_EXTERN PetscErrorCode STMatMult(ST,PetscInt,Vec,Vec);
PETSC_EXTERN PetscErrorCode STMatMultTranspose(ST,PetscInt,Vec,Vec);
PETSC_EXTERN PetscErrorCode STMatSolve(ST,Vec,Vec);
PETSC_EXTERN PetscErrorCode STMatSolveTranspose(ST,Vec,Vec);
PETSC_EXTERN PetscErrorCode STGetBilinearForm(ST,Mat*);
PETSC_EXTERN PetscErrorCode STApplyTranspose(ST,Vec,Vec);
PETSC_EXTERN PetscErrorCode STMatSetUp(ST,PetscScalar,PetscScalar*);
PETSC_EXTERN PetscErrorCode STPostSolve(ST);
PETSC_EXTERN PetscErrorCode STResetMatrixState(ST);
PETSC_EXTERN PetscErrorCode STSetWorkVecs(ST,PetscInt);

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
PETSC_EXTERN PetscErrorCode STIsInjective(ST,PetscBool*);

PETSC_EXTERN PetscErrorCode STCheckNullSpace(ST,BV);

PETSC_EXTERN PetscErrorCode STMatCreateVecs(ST,Vec*,Vec*);
PETSC_EXTERN PetscErrorCode STMatCreateVecsEmpty(ST,Vec*,Vec*);
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
PETSC_EXTERN const char *STMatModes[];

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

PETSC_EXTERN PetscErrorCode STFilterSetInterval(ST,PetscReal,PetscReal);
PETSC_EXTERN PetscErrorCode STFilterGetInterval(ST,PetscReal*,PetscReal*);
PETSC_EXTERN PetscErrorCode STFilterSetRange(ST,PetscReal,PetscReal);
PETSC_EXTERN PetscErrorCode STFilterGetRange(ST,PetscReal*,PetscReal*);
PETSC_EXTERN PetscErrorCode STFilterSetDegree(ST,PetscInt);
PETSC_EXTERN PetscErrorCode STFilterGetDegree(ST,PetscInt*);
PETSC_EXTERN PetscErrorCode STFilterGetThreshold(ST,PetscReal*);

#endif

