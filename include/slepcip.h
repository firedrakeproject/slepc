/*
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

#if !defined(__SLEPCIP_H)
#define __SLEPCIP_H
#include "slepcsys.h"
PETSC_EXTERN_CXX_BEGIN

extern PetscErrorCode IPInitializePackage(const char[]);
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

    Level: advanced

.seealso: IPSetType(), IP
E*/
#define IPType         char*
#define IPBILINEAR     "bilinear"
#define IPSESQUILINEAR "sesquilinear"

/* Logging support */
extern PetscClassId IP_CLASSID;

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

extern PetscErrorCode IPCreate(MPI_Comm,IP*);
extern PetscErrorCode IPSetType(IP,const IPType);
extern PetscErrorCode IPGetType(IP,const IPType*);
extern PetscErrorCode IPSetOptionsPrefix(IP,const char *);
extern PetscErrorCode IPAppendOptionsPrefix(IP,const char *);
extern PetscErrorCode IPGetOptionsPrefix(IP,const char *[]);
extern PetscErrorCode IPSetFromOptions(IP);
extern PetscErrorCode IPSetOrthogonalization(IP,IPOrthogType,IPOrthogRefineType,PetscReal);
extern PetscErrorCode IPGetOrthogonalization(IP,IPOrthogType*,IPOrthogRefineType*,PetscReal*);
extern PetscErrorCode IPView(IP,PetscViewer);
extern PetscErrorCode IPDestroy(IP*);
extern PetscErrorCode IPReset(IP);

extern PetscErrorCode IPOrthogonalize(IP,PetscInt,Vec*,PetscInt,PetscBool*,Vec*,Vec,PetscScalar*,PetscReal*,PetscBool*);
extern PetscErrorCode IPBiOrthogonalize(IP,PetscInt,Vec*,Vec*,Vec,PetscScalar*,PetscReal*);
extern PetscErrorCode IPQRDecomposition(IP,Vec*,PetscInt,PetscInt,PetscScalar*,PetscInt);

extern PetscErrorCode IPSetMatrix(IP,Mat);
extern PetscErrorCode IPGetMatrix(IP,Mat*);
extern PetscErrorCode IPApplyMatrix(IP,Vec,Vec);

extern PetscErrorCode IPInnerProduct(IP ip,Vec,Vec,PetscScalar*);
extern PetscErrorCode IPInnerProductBegin(IP ip,Vec,Vec,PetscScalar*);
extern PetscErrorCode IPInnerProductEnd(IP ip,Vec,Vec,PetscScalar*);
extern PetscErrorCode IPMInnerProduct(IP ip,Vec,PetscInt,const Vec[],PetscScalar*);
extern PetscErrorCode IPMInnerProductBegin(IP ip,Vec,PetscInt,const Vec[],PetscScalar*);
extern PetscErrorCode IPMInnerProductEnd(IP ip,Vec,PetscInt,const Vec[],PetscScalar*);
extern PetscErrorCode IPNorm(IP ip,Vec,PetscReal*);
extern PetscErrorCode IPNormBegin(IP ip,Vec,PetscReal*);
extern PetscErrorCode IPNormEnd(IP ip,Vec,PetscReal*);

extern PetscFList IPList;
extern PetscBool  IPRegisterAllCalled;
extern PetscErrorCode IPRegisterAll(const char[]);
extern PetscErrorCode IPRegister(const char[],const char[],const char[],PetscErrorCode(*)(IP));
extern PetscErrorCode IPRegisterDestroy(void);

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

extern PetscErrorCode IPGetOperationCounters(IP,PetscInt*);
extern PetscErrorCode IPResetOperationCounters(IP);

PETSC_EXTERN_CXX_END
#endif
