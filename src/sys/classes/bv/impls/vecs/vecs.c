/*
    BV implemented as an array of independent Vecs

   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
   SLEPc - Scalable Library for Eigenvalue Problem Computations
   Copyright (c) 2002-2013, Universitat Politecnica de Valencia, Spain

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

#include <slepc-private/bvimpl.h>          /*I "slepcbv.h" I*/

typedef struct {
  Vec *V;
} BV_VECS;

#undef __FUNCT__
#define __FUNCT__ "BVGetColumn_Vecs"
PetscErrorCode BVGetColumn_Vecs(BV bv,PetscInt j,Vec *v)
{
  BV_VECS        *ctx = (BV_VECS*)bv->data;
  PetscInt       l;

  PetscFunctionBegin;
  l = BVAvailableVec;
  bv->cv[l] = ctx->V[j];
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "BVDestroy_Vecs"
PetscErrorCode BVDestroy_Vecs(BV bv)
{
  PetscErrorCode ierr;
  BV_VECS        *ctx = (BV_VECS*)bv->data;

  PetscFunctionBegin;
  ierr = VecDestroyVecs(bv->k,&ctx->V);CHKERRQ(ierr);
  ierr = PetscFree(bv->data);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "BVCreate_Vecs"
PETSC_EXTERN PetscErrorCode BVCreate_Vecs(BV bv)
{
  PetscErrorCode ierr;
  BV_VECS        *ctx;

  PetscFunctionBegin;
  ierr = PetscNewLog(bv,&ctx);CHKERRQ(ierr);
  bv->data = (void*)ctx;

  ierr = VecDuplicateVecs(bv->t,bv->k,&ctx->V);CHKERRQ(ierr);
  ierr = PetscLogObjectParents(bv,bv->k,ctx->V);CHKERRQ(ierr);

  bv->ops->getcolumn      = BVGetColumn_Vecs;
  bv->ops->destroy        = BVDestroy_Vecs;
  PetscFunctionReturn(0);
}