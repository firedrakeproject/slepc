/*
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
   SLEPc - Scalable Library for Eigenvalue Problem Computations
   Copyright (c) 2002-2014, Universitat Politecnica de Valencia, Spain

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

#if !defined(__SLEPCFN_H)
#define __SLEPCFN_H
#include <slepcsys.h>

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
#define FNRATIONAL "rational"
#define FNEXP      "exp"
#define FNLOG      "log"
#define FNPHI      "phi"

/* Logging support */
PETSC_EXTERN PetscClassId FN_CLASSID;

PETSC_EXTERN PetscErrorCode FNCreate(MPI_Comm,FN*);
PETSC_EXTERN PetscErrorCode FNSetType(FN,FNType);
PETSC_EXTERN PetscErrorCode FNGetType(FN,FNType*);
PETSC_EXTERN PetscErrorCode FNSetOptionsPrefix(FN,const char *);
PETSC_EXTERN PetscErrorCode FNAppendOptionsPrefix(FN,const char *);
PETSC_EXTERN PetscErrorCode FNGetOptionsPrefix(FN,const char *[]);
PETSC_EXTERN PetscErrorCode FNSetFromOptions(FN);
PETSC_EXTERN PetscErrorCode FNView(FN,PetscViewer);
PETSC_EXTERN PetscErrorCode FNDestroy(FN*);

PETSC_EXTERN PetscErrorCode FNSetParameters(FN,PetscInt,PetscScalar*,PetscInt,PetscScalar*);
PETSC_EXTERN PetscErrorCode FNGetParameters(FN,PetscInt*,PetscScalar**,PetscInt*,PetscScalar**);

PETSC_EXTERN PetscErrorCode FNEvaluateFunction(FN,PetscScalar,PetscScalar*);
PETSC_EXTERN PetscErrorCode FNEvaluateDerivative(FN,PetscScalar,PetscScalar*);

PETSC_EXTERN PetscFunctionList FNList;
PETSC_EXTERN PetscBool         FNRegisterAllCalled;
PETSC_EXTERN PetscErrorCode FNRegisterAll(void);
PETSC_EXTERN PetscErrorCode FNRegister(const char[],PetscErrorCode(*)(FN));

#endif
