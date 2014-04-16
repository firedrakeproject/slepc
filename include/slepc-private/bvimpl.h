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

PETSC_EXTERN PetscLogEvent BV_Mult,BV_MultVec,BV_Dot;

typedef struct _BVOps *BVOps;

struct _BVOps {
  PetscErrorCode (*mult)(BV,PetscScalar,PetscScalar,BV,Mat);
  PetscErrorCode (*multvec)(BV,PetscScalar,PetscScalar,Vec,PetscScalar*);
  PetscErrorCode (*dot)(BV,BV,Mat);
  PetscErrorCode (*getcolumn)(BV,PetscInt,Vec*);
  PetscErrorCode (*restorecolumn)(BV,PetscInt,Vec*);
  PetscErrorCode (*setfromoptions)(BV);
  PetscErrorCode (*create)(BV);
  PetscErrorCode (*destroy)(BV);
  PetscErrorCode (*view)(BV,PetscViewer);
};

struct _p_BV {
  PETSCHEADER(struct _BVOps);
  /*------------------------- User parameters --------------------------*/
  Vec              t;            /* Template vector */
  PetscInt         n,N;          /* Dimensions of vectors */
  PetscInt         k;            /* Number of vectors */

  /*------------------------- Misc data --------------------------*/
  Vec              cv[2];        /* Column vectors obtained with BVGetColumn() */
  PetscInt         ci[2];        /* Column indices of obtained vectors */
  PetscObjectState st[2];        /* State of obtained vectors */
  PetscObjectId    id[2];        /* Object id of obtained vectors */
  void             *data;
};

/*
  BVAvailableVec: First (0) or second (1) vector available for
  getcolumn operation (or -1 if both vectors already fetched).
*/
#define BVAvailableVec (((bv->ci[0]==-1)? 0: (bv->ci[1]==-1)? 1: -1))

/*
    Macros to test valid BV arguments
*/
#if !defined(PETSC_USE_DEBUG)

#define BVCheckSizes(h,arg) do {} while (0)

#else

#define BVCheckSizes(h,arg) \
  do { \
    if (!h->k) SETERRQ1(PetscObjectComm((PetscObject)h),PETSC_ERR_ARG_WRONGSTATE,"BV sizes have not been defined: Parameter #%d",arg); \
  } while (0)

#endif

#endif
