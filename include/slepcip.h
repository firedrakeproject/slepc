/*
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

#if !defined(__SLEPCIP_H)
#define __SLEPCIP_H
#include "slepcsys.h"

PETSC_EXTERN PetscErrorCode IPInitializePackage(const char[]);
/*S
    IP - Abstraction of a vector inner product, that can be defined
    in different ways. Using this object is not required for application
    programmers.

    Level: beginner

.seealso:  IPCreate()
S*/
typedef struct _p_IP* IP;

/*E
    IPType - String with the name of the inner product. For complex scalars,
    it is possible to choose between a sesquilinear form (x,y)=x^H*M*y (the default)
    or a bilinear form (x,y)=x^T*M*y (without complex conjugation). In the case
    of real scalars, only the bilinear form (x,y)=x^T*M*y is available.
    Apart form these, there is also an indefinite inner product, defined by
    and indefinite matrix M.

    Level: advanced

.seealso: IPSetType(), IP
E*/
#define IPType         char*
#define IPBILINEAR     "bilinear"
#define IPSESQUILINEAR "sesquilinear"
#define IPINDEFINITE   "indefinite"

/* Logging support */
PETSC_EXTERN PetscClassId IP_CLASSID;

/*E
    IPOrthogType - determines what type of orthogonalization to use

    Level: advanced

.seealso: IPSetOrthogonalization(), IPGetOrthogonalization(), IPOrthogonalize()
E*/
typedef enum { IP_ORTHOG_MGS,
               IP_ORTHOG_CGS } IPOrthogType;

/*E
    IPOrthogRefineType - determines what type of refinement
    to use during orthogonalization

    Level: advanced

.seealso: IPSetOrthogonalization(), IPGetOrthogonalization(), IPOrthogonalize()
E*/
typedef enum { IP_ORTHOG_REFINE_NEVER,
               IP_ORTHOG_REFINE_IFNEEDED,
               IP_ORTHOG_REFINE_ALWAYS } IPOrthogRefineType;

PETSC_EXTERN PetscErrorCode IPCreate(MPI_Comm,IP*);
PETSC_EXTERN PetscErrorCode IPSetType(IP,const IPType);
PETSC_EXTERN PetscErrorCode IPGetType(IP,const IPType*);
PETSC_EXTERN PetscErrorCode IPSetOptionsPrefix(IP,const char *);
PETSC_EXTERN PetscErrorCode IPAppendOptionsPrefix(IP,const char *);
PETSC_EXTERN PetscErrorCode IPGetOptionsPrefix(IP,const char *[]);
PETSC_EXTERN PetscErrorCode IPSetFromOptions(IP);
PETSC_EXTERN PetscErrorCode IPSetOrthogonalization(IP,IPOrthogType,IPOrthogRefineType,PetscReal);
PETSC_EXTERN PetscErrorCode IPGetOrthogonalization(IP,IPOrthogType*,IPOrthogRefineType*,PetscReal*);
PETSC_EXTERN PetscErrorCode IPView(IP,PetscViewer);
PETSC_EXTERN PetscErrorCode IPDestroy(IP*);
PETSC_EXTERN PetscErrorCode IPReset(IP);

PETSC_EXTERN PetscErrorCode IPOrthogonalize(IP,PetscInt,Vec*,PetscInt,PetscBool*,Vec*,Vec,PetscScalar*,PetscReal*,PetscBool*);
PETSC_EXTERN PetscErrorCode IPBOrthogonalize(IP,PetscInt,Vec*,Vec*,PetscReal*,PetscInt,PetscBool*,Vec*,Vec*,PetscReal*,Vec,Vec,PetscScalar*,PetscReal*,PetscBool*);
PETSC_EXTERN PetscErrorCode IPBiOrthogonalize(IP,PetscInt,Vec*,Vec*,Vec,PetscScalar*,PetscReal*);
PETSC_EXTERN PetscErrorCode IPPseudoOrthogonalize(IP,PetscInt,Vec*,PetscReal*,Vec,PetscScalar*,PetscReal*,PetscBool*);
PETSC_EXTERN PetscErrorCode IPQRDecomposition(IP,Vec*,PetscInt,PetscInt,PetscScalar*,PetscInt);

PETSC_EXTERN PetscErrorCode IPSetMatrix(IP,Mat);
PETSC_EXTERN PetscErrorCode IPGetMatrix(IP,Mat*);
PETSC_EXTERN PetscErrorCode IPApplyMatrix(IP,Vec,Vec);

PETSC_EXTERN PetscErrorCode IPInnerProduct(IP ip,Vec,Vec,PetscScalar*);
PETSC_EXTERN PetscErrorCode IPInnerProductBegin(IP ip,Vec,Vec,PetscScalar*);
PETSC_EXTERN PetscErrorCode IPInnerProductEnd(IP ip,Vec,Vec,PetscScalar*);
PETSC_EXTERN PetscErrorCode IPMInnerProduct(IP ip,Vec,PetscInt,const Vec[],PetscScalar*);
PETSC_EXTERN PetscErrorCode IPMInnerProductBegin(IP ip,Vec,PetscInt,const Vec[],PetscScalar*);
PETSC_EXTERN PetscErrorCode IPMInnerProductEnd(IP ip,Vec,PetscInt,const Vec[],PetscScalar*);
PETSC_EXTERN PetscErrorCode IPNorm(IP ip,Vec,PetscReal*);
PETSC_EXTERN PetscErrorCode IPNormBegin(IP ip,Vec,PetscReal*);
PETSC_EXTERN PetscErrorCode IPNormEnd(IP ip,Vec,PetscReal*);

PETSC_EXTERN PetscFList IPList;
PETSC_EXTERN PetscBool  IPRegisterAllCalled;
PETSC_EXTERN PetscErrorCode IPRegisterAll(const char[]);
PETSC_EXTERN PetscErrorCode IPRegister(const char[],const char[],const char[],PetscErrorCode(*)(IP));
PETSC_EXTERN PetscErrorCode IPRegisterDestroy(void);

/*MC
   IPRegisterDynamic - Adds an inner product to the IP package.

   Synopsis:
   PetscErrorCode IPRegisterDynamic(const char *name,const char *path,const char *name_create,PetscErrorCode (*routine_create)(IP))

   Not collective

   Input Parameters:
+  name - name of a new user-defined IP
.  path - path (either absolute or relative) the library containing this solver
.  name_create - name of routine to create context
-  routine_create - routine to create context

   Notes:
   IPRegisterDynamic() may be called multiple times to add several user-defined inner products.

   If dynamic libraries are used, then the fourth input argument (routine_create)
   is ignored.

   Level: advanced

.seealso: IPRegisterDestroy(), IPRegisterAll()
M*/
#if defined(PETSC_USE_DYNAMIC_LIBRARIES)
#define IPRegisterDynamic(a,b,c,d) IPRegister(a,b,c,0)
#else
#define IPRegisterDynamic(a,b,c,d) IPRegister(a,b,c,d)
#endif

PETSC_EXTERN PetscErrorCode IPGetOperationCounters(IP,PetscInt*);
PETSC_EXTERN PetscErrorCode IPResetOperationCounters(IP);

#endif
