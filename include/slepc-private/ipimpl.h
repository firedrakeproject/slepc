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

#if !defined(_IPIMPL)
#define _IPIMPL

#include <slepcip.h>
#include <slepc-private/slepcimpl.h>

PETSC_EXTERN PetscLogEvent IP_InnerProduct,IP_Orthogonalize,IP_ApplyMatrix;

typedef struct _IPOps *IPOps;

struct _IPOps {
  PetscErrorCode (*normbegin)(IP,Vec,PetscReal*);
  PetscErrorCode (*normend)(IP,Vec,PetscReal*);
  PetscErrorCode (*innerproductbegin)(IP,Vec,Vec,PetscScalar*);
  PetscErrorCode (*innerproductend)(IP,Vec,Vec,PetscScalar*);
  PetscErrorCode (*minnerproductbegin)(IP,Vec,PetscInt,const Vec[],PetscScalar*);
  PetscErrorCode (*minnerproductend)(IP,Vec,PetscInt,const Vec[],PetscScalar*);
};

struct _p_IP {
  PETSCHEADER(struct _IPOps);
  IPOrthogType       orthog_type;    /* which orthogonalization to use */
  IPOrthogRefineType orthog_ref;     /* refinement method */
  PetscReal          orthog_eta;     /* refinement threshold */
  Mat                matrix;
  PetscInt           innerproducts;

  /*------------------------- Cache Bx product -------------------*/
  PetscInt           xid;
  PetscInt           xstate;
  Vec                Bx;
};

PETSC_EXTERN PetscErrorCode IPSetDefaultType_Private(IP);
PETSC_EXTERN PetscErrorCode IPApplyMatrix_Private(IP,Vec);
PETSC_EXTERN PetscErrorCode IPOrthogonalizeCGS1(IP,PetscInt,Vec*,PetscInt,PetscBool*,Vec*,Vec,PetscScalar*,PetscReal*,PetscReal*);

#endif
