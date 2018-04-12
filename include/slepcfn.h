/*
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
   SLEPc - Scalable Library for Eigenvalue Problem Computations
   Copyright (c) 2002-2018, Universitat Politecnica de Valencia, Spain

   This file is part of SLEPc.
   SLEPc is distributed under a 2-clause BSD license (see LICENSE).
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
*/
/*
   User interface for the mathematical function object in SLEPc
*/

#if !defined(__SLEPCFN_H)
#define __SLEPCFN_H
#include <slepcsys.h>

#define FN_MAX_SOLVE 6

PETSC_EXTERN PetscErrorCode FNInitializePackage(void);
/*S
   FN - Abstraction of a mathematical function.

   Level: beginner

.seealso: FNCreate()
S*/
typedef struct _p_FN* FN;

/*J
   FNType - String with the name of the mathematical function.

   Level: beginner

.seealso: FNSetType(), FN
J*/
typedef const char* FNType;
#define FNCOMBINE  "combine"
#define FNRATIONAL "rational"
#define FNEXP      "exp"
#define FNLOG      "log"
#define FNPHI      "phi"
#define FNSQRT     "sqrt"
#define FNINVSQRT  "invsqrt"

/* Logging support */
PETSC_EXTERN PetscClassId FN_CLASSID;

/*E
    FNCombineType - Determines how two functions are combined

    Level: advanced

.seealso: FNCombineSetChildren()
E*/
typedef enum { FN_COMBINE_ADD,
               FN_COMBINE_MULTIPLY,
               FN_COMBINE_DIVIDE,
               FN_COMBINE_COMPOSE } FNCombineType;

/*E
    FNParallelType - Indicates the parallel mode that will be used for matrix evaluation

    Level: advanced

.seealso: FNSetParallel()
E*/
typedef enum { FN_PARALLEL_REDUNDANT,
               FN_PARALLEL_SYNCHRONIZED } FNParallelType;
PETSC_EXTERN const char *FNParallelTypes[];

PETSC_EXTERN PetscErrorCode FNCreate(MPI_Comm,FN*);
PETSC_EXTERN PetscErrorCode FNSetType(FN,FNType);
PETSC_EXTERN PetscErrorCode FNGetType(FN,FNType*);
PETSC_EXTERN PetscErrorCode FNSetOptionsPrefix(FN,const char *);
PETSC_EXTERN PetscErrorCode FNAppendOptionsPrefix(FN,const char *);
PETSC_EXTERN PetscErrorCode FNGetOptionsPrefix(FN,const char *[]);
PETSC_EXTERN PetscErrorCode FNSetFromOptions(FN);
PETSC_EXTERN PetscErrorCode FNView(FN,PetscViewer);
PETSC_EXTERN PetscErrorCode FNDestroy(FN*);
PETSC_EXTERN PetscErrorCode FNDuplicate(FN,MPI_Comm,FN*);

PETSC_EXTERN PetscErrorCode FNSetScale(FN,PetscScalar,PetscScalar);
PETSC_EXTERN PetscErrorCode FNGetScale(FN,PetscScalar*,PetscScalar*);
PETSC_EXTERN PetscErrorCode FNSetMethod(FN,PetscInt);
PETSC_EXTERN PetscErrorCode FNGetMethod(FN,PetscInt*);
PETSC_EXTERN PetscErrorCode FNSetParallel(FN,FNParallelType);
PETSC_EXTERN PetscErrorCode FNGetParallel(FN,FNParallelType*);

PETSC_EXTERN PetscErrorCode FNEvaluateFunction(FN,PetscScalar,PetscScalar*);
PETSC_EXTERN PetscErrorCode FNEvaluateDerivative(FN,PetscScalar,PetscScalar*);
PETSC_EXTERN PetscErrorCode FNEvaluateFunctionMat(FN,Mat,Mat);
PETSC_EXTERN PetscErrorCode FNEvaluateFunctionMatVec(FN,Mat,Vec);

PETSC_EXTERN PetscFunctionList FNList;
PETSC_EXTERN PetscErrorCode FNRegister(const char[],PetscErrorCode(*)(FN));

/* --------- options specific to particular functions -------- */

PETSC_EXTERN PetscErrorCode FNRationalSetNumerator(FN,PetscInt,PetscScalar*);
PETSC_EXTERN PetscErrorCode FNRationalGetNumerator(FN,PetscInt*,PetscScalar**);
PETSC_EXTERN PetscErrorCode FNRationalSetDenominator(FN,PetscInt,PetscScalar*);
PETSC_EXTERN PetscErrorCode FNRationalGetDenominator(FN,PetscInt*,PetscScalar**);

PETSC_EXTERN PetscErrorCode FNCombineSetChildren(FN,FNCombineType,FN,FN);
PETSC_EXTERN PetscErrorCode FNCombineGetChildren(FN,FNCombineType*,FN*,FN*);

PETSC_EXTERN PetscErrorCode FNPhiSetIndex(FN,PetscInt);
PETSC_EXTERN PetscErrorCode FNPhiGetIndex(FN,PetscInt*);

#endif
