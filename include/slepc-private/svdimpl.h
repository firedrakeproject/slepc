/*
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

#if !defined(_SVDIMPL)
#define _SVDIMPL

#include <slepcsvd.h>
#include <slepc-private/slepcimpl.h>

PETSC_EXTERN PetscLogEvent SVD_SetUp,SVD_Solve;

typedef struct _SVDOps *SVDOps;

struct _SVDOps {
  PetscErrorCode (*solve)(SVD);
  PetscErrorCode (*setup)(SVD);
  PetscErrorCode (*setfromoptions)(SVD);
  PetscErrorCode (*publishoptions)(SVD);
  PetscErrorCode (*destroy)(SVD);
  PetscErrorCode (*reset)(SVD);
  PetscErrorCode (*view)(SVD,PetscViewer);
};

/*
     Maximum number of monitors you can run with a single SVD
*/
#define MAXSVDMONITORS 5

/*
   Defines the SVD data structure.
*/
struct _p_SVD {
  PETSCHEADER(struct _SVDOps);
  Mat              OP;          /* problem matrix */
  Mat              A;           /* problem matrix (m>n) */
  Mat              AT;          /* transposed matrix */
  SVDTransposeMode transmode;   /* transpose mode */
  PetscReal        *sigma;      /* singular values */
  PetscInt         *perm;       /* permutation for singular value ordering */
  BV               U,V;         /* left and right singular vectors */
  Vec              *IS,*ISL;    /* placeholder for references to user-provided initial space */
  SVDWhich         which;       /* which singular values are computed */
  PetscInt         nconv;       /* number of converged values */
  PetscInt         nsv;         /* number of requested values */
  PetscInt         ncv;         /* basis size */
  PetscInt         mpd;         /* maximum dimension of projected problem */
  PetscInt         nini,ninil;  /* number of initial vectors (negative means not copied yet) */
  PetscInt         its;         /* iteration counter */
  PetscInt         max_it;      /* max iterations */
  PetscReal        tol;         /* tolerance */
  PetscReal        *errest;     /* error estimates */
  PetscRandom      rand;        /* random number generator */
  void             *data;       /* placeholder for misc stuff associated
                                   with a particular solver */
  PetscInt         setupcalled;
  SVDConvergedReason reason;
  DS               ds;          /* direct solver object */
  PetscBool        trackall;
  PetscInt         matvecs;
  PetscBool        leftbasis;   /* if U is filled by the solver */
  PetscBool        lvecsavail;  /* if U contains left singular vectors */

  PetscErrorCode   (*monitor[MAXSVDMONITORS])(SVD,PetscInt,PetscInt,PetscReal*,PetscReal*,PetscInt,void*);
  PetscErrorCode   (*monitordestroy[MAXSVDMONITORS])(void**);
  void             *monitorcontext[MAXSVDMONITORS];
  PetscInt         numbermonitors;
};

#undef __FUNCT__
#define __FUNCT__ "SVDMatMult"
PETSC_STATIC_INLINE PetscErrorCode SVDMatMult(SVD svd,PetscBool trans,Vec x,Vec y)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  svd->matvecs++;
  if (trans) {
    if (svd->AT) {
      ierr = MatMult(svd->AT,x,y);CHKERRQ(ierr);
    } else {
#if defined(PETSC_USE_COMPLEX)
      ierr = MatMultHermitianTranspose(svd->A,x,y);CHKERRQ(ierr);
#else
      ierr = MatMultTranspose(svd->A,x,y);CHKERRQ(ierr);
#endif
    }
  } else {
    if (svd->A) {
      ierr = MatMult(svd->A,x,y);CHKERRQ(ierr);
    } else {
#if defined(PETSC_USE_COMPLEX)
      ierr = MatMultHermitianTranspose(svd->AT,x,y);CHKERRQ(ierr);
#else
      ierr = MatMultTranspose(svd->AT,x,y);CHKERRQ(ierr);
#endif
    }
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "SVDMatGetVecs"
PETSC_STATIC_INLINE PetscErrorCode SVDMatGetVecs(SVD svd,Vec *x,Vec *y)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (svd->A) {
    ierr = MatGetVecs(svd->A,x,y);CHKERRQ(ierr);
  } else {
    ierr = MatGetVecs(svd->AT,y,x);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "SVDMatGetSize"
PETSC_STATIC_INLINE PetscErrorCode SVDMatGetSize(SVD svd,PetscInt *m,PetscInt *n)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (svd->A) {
    ierr = MatGetSize(svd->A,m,n);CHKERRQ(ierr);
  } else {
    ierr = MatGetSize(svd->AT,n,m);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "SVDMatGetLocalSize"
PETSC_STATIC_INLINE PetscErrorCode SVDMatGetLocalSize(SVD svd,PetscInt *m,PetscInt *n)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (svd->A) {
    ierr = MatGetLocalSize(svd->A,m,n);CHKERRQ(ierr);
  } else {
    ierr = MatGetLocalSize(svd->AT,n,m);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
} 

PETSC_INTERN PetscErrorCode SVDTwoSideLanczos(SVD,PetscReal*,PetscReal*,BV,BV,PetscInt,PetscInt);

#endif
