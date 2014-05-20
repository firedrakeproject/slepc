/*
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
   SLEPc - Scalable Library for Eigenvalue Problem Computations
   Copyright (c) , Universitat Politecnica de Valencia, Spain

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

#if !defined(_BVIMPL)
#define _BVIMPL

#include <slepcbv.h>
#include <slepc-private/slepcimpl.h>

PETSC_EXTERN PetscLogEvent BV_Create,BV_Copy,BV_Mult,BV_Dot,BV_Orthogonalize,BV_Scale,BV_Norm,BV_SetRandom;

typedef struct _BVOps *BVOps;

struct _BVOps {
  PetscErrorCode (*mult)(BV,PetscScalar,PetscScalar,BV,Mat);
  PetscErrorCode (*multvec)(BV,PetscScalar,PetscScalar,Vec,PetscScalar*);
  PetscErrorCode (*multinplace)(BV,Mat,PetscInt,PetscInt);
  PetscErrorCode (*multinplacetrans)(BV,Mat,PetscInt,PetscInt);
  PetscErrorCode (*dot)(BV,BV,Mat);
  PetscErrorCode (*dotvec)(BV,Vec,PetscScalar*);
  PetscErrorCode (*scale)(BV,PetscInt,PetscScalar);
  PetscErrorCode (*norm)(BV,PetscInt,NormType,PetscReal*);
  PetscErrorCode (*orthogonalize)(BV,Mat);
  PetscErrorCode (*copy)(BV,BV);
  PetscErrorCode (*resize)(BV,PetscInt,PetscBool);
  PetscErrorCode (*getcolumn)(BV,PetscInt,Vec*);
  PetscErrorCode (*restorecolumn)(BV,PetscInt,Vec*);
  PetscErrorCode (*setfromoptions)(BV);
  PetscErrorCode (*create)(BV);
  PetscErrorCode (*view)(BV,PetscViewer);
  PetscErrorCode (*destroy)(BV);
};

struct _p_BV {
  PETSCHEADER(struct _BVOps);
  /*------------------------- User parameters --------------------------*/
  Vec                t;            /* template vector */
  PetscInt           n,N;          /* dimensions of vectors (local, global) */
  PetscInt           m;            /* number of vectors */
  PetscInt           l;            /* number of leading columns */
  PetscInt           k;            /* number of active columns */
  PetscInt           nc;           /* number of constraints */
  BVOrthogType       orthog_type;  /* which orthogonalization to use */
  BVOrthogRefineType orthog_ref;   /* refinement method */
  PetscReal          orthog_eta;   /* refinement threshold */
  Mat                matrix;       /* inner product matrix */
  PetscBool          indef;        /* matrix is indefinite */

  /*---------------------- Cached data and workspace -------------------*/
  Vec                Bx;           /* result of matrix times a vector x */
  PetscInt           xid;          /* object id of vector x */
  PetscInt           xstate;       /* state of vector x */
  Vec                cv[2];        /* column vectors obtained with BVGetColumn() */
  PetscInt           ci[2];        /* column indices of obtained vectors */
  PetscObjectState   st[2];        /* state of obtained vectors */
  PetscObjectId      id[2];        /* object id of obtained vectors */
  PetscScalar        *h,*c;        /* orthogonalization coefficients */
  PetscReal          *omega;       /* signature matrix values for indefinite case */
  PetscScalar        *work;
  PetscInt           lwork;
  void               *data;
};

#undef __FUNCT__
#define __FUNCT__ "BV_MatMult"
PETSC_STATIC_INLINE PetscErrorCode BV_MatMult(BV bv,Vec x)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (((PetscObject)x)->id != bv->xid || ((PetscObject)x)->state != bv->xstate) {
    ierr = MatMult(bv->matrix,x,bv->Bx);CHKERRQ(ierr);
    bv->xid = ((PetscObject)x)->id;
    bv->xstate = ((PetscObject)x)->state;
  }
  PetscFunctionReturn(0);
}

/*
  BVAvailableVec: First (0) or second (1) vector available for
  getcolumn operation (or -1 if both vectors already fetched).
*/
#define BVAvailableVec (((bv->ci[0]==-bv->nc-1)? 0: (bv->ci[1]==-bv->nc-1)? 1: -1))

/*
    Macros to test valid BV arguments
*/
#if !defined(PETSC_USE_DEBUG)

#define BVCheckSizes(h,arg) do {} while (0)

#else

#define BVCheckSizes(h,arg) \
  do { \
    if (!h->m) SETERRQ1(PetscObjectComm((PetscObject)h),PETSC_ERR_ARG_WRONGSTATE,"BV sizes have not been defined: Parameter #%d",arg); \
  } while (0)

#endif

PETSC_INTERN PetscErrorCode BVView_Vecs(BV,PetscViewer);

PETSC_INTERN PetscErrorCode BVAllocateWork_Private(BV,PetscInt);

PETSC_INTERN PetscErrorCode BVMult_BLAS_Private(BV,PetscInt,PetscInt,PetscInt,PetscInt,PetscScalar,PetscScalar*,PetscScalar*,PetscScalar,PetscScalar*);
PETSC_INTERN PetscErrorCode BVMultVec_BLAS_Private(BV,PetscInt,PetscInt,PetscScalar,PetscScalar*,PetscScalar*,PetscScalar,PetscScalar*);
PETSC_INTERN PetscErrorCode BVMultInPlace_BLAS_Private(BV,PetscInt,PetscInt,PetscInt,PetscInt,PetscInt,PetscScalar*,PetscScalar*,PetscBool);
PETSC_INTERN PetscErrorCode BVMultInPlace_Vecs_Private(BV,PetscInt,PetscInt,PetscInt,Vec*,PetscScalar*,PetscBool);
PETSC_INTERN PetscErrorCode BVDot_BLAS_Private(BV,PetscInt,PetscInt,PetscInt,PetscInt,PetscScalar*,PetscScalar*,PetscScalar*,PetscBool);
PETSC_INTERN PetscErrorCode BVDotVec_BLAS_Private(BV,PetscInt,PetscInt,PetscScalar*,PetscScalar*,PetscScalar*,PetscBool);
PETSC_INTERN PetscErrorCode BVScale_BLAS_Private(BV,PetscInt,PetscScalar*,PetscScalar);
PETSC_INTERN PetscErrorCode BVNorm_LAPACK_Private(BV,PetscInt,PetscInt,PetscScalar*,NormType,PetscReal*,PetscBool);
PETSC_INTERN PetscErrorCode BVOrthogonalize_LAPACK_Private(BV,PetscInt,PetscInt,PetscScalar*,PetscScalar*,PetscBool);

#endif
