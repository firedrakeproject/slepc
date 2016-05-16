/*
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
   SLEPc - Scalable Library for Eigenvalue Problem Computations
   Copyright (c) 2002-2016, Universitat Politecnica de Valencia, Spain

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

#if !defined(_MFNIMPL)
#define _MFNIMPL

#include <slepcmfn.h>
#include <slepc/private/slepcimpl.h>

PETSC_EXTERN PetscBool MFNRegisterAllCalled;
PETSC_EXTERN PetscErrorCode MFNRegisterAll(void);
PETSC_EXTERN PetscLogEvent MFN_SetUp, MFN_Solve;

typedef struct _MFNOps *MFNOps;

struct _MFNOps {
  PetscErrorCode (*solve)(MFN,Vec,Vec);
  PetscErrorCode (*setup)(MFN);
  PetscErrorCode (*setfromoptions)(PetscOptionItems*,MFN);
  PetscErrorCode (*publishoptions)(MFN);
  PetscErrorCode (*destroy)(MFN);
  PetscErrorCode (*reset)(MFN);
  PetscErrorCode (*view)(MFN,PetscViewer);
};

/*
     Maximum number of monitors you can run with a single MFN
*/
#define MAXMFNMONITORS 5

/*
   Defines the MFN data structure.
*/
struct _p_MFN {
  PETSCHEADER(struct _MFNOps);
  /*------------------------- User parameters ---------------------------*/
  Mat            A;              /* the problem matrix */
  FN             fn;             /* which function to compute */
  PetscInt       max_it;         /* maximum number of iterations */
  PetscInt       ncv;            /* number of basis vectors */
  PetscReal      tol;            /* tolerance */
  PetscBool      errorifnotconverged;    /* error out if MFNSolve() does not converge */

  /*-------------- User-provided functions and contexts -----------------*/
  PetscErrorCode (*monitor[MAXMFNMONITORS])(MFN,PetscInt,PetscReal,void*);
  PetscErrorCode (*monitordestroy[MAXMFNMONITORS])(void**);
  void           *monitorcontext[MAXMFNMONITORS];
  PetscInt       numbermonitors;

  /*----------------- Child objects and working data -------------------*/
  BV             V;              /* set of basis vectors */
  PetscInt       nwork;          /* number of work vectors */
  Vec            *work;          /* work vectors */
  void           *data;          /* placeholder for solver-specific stuff */

  /* ----------------------- Status variables -------------------------- */
  PetscInt       its;            /* number of iterations so far computed */
  PetscInt       nv;             /* size of current Schur decomposition */
  PetscReal      errest;         /* error estimate */
  PetscReal      bnorm;          /* computed norm of right-hand side in current solve */
  PetscInt       setupcalled;
  MFNConvergedReason reason;
};

#undef __FUNCT__
#define __FUNCT__ "MFN_CreateDenseMat"
/*
   MFN_CreateDenseMat - Creates a dense Mat of size k unless it already has that size
*/
PETSC_STATIC_INLINE PetscErrorCode MFN_CreateDenseMat(PetscInt k,Mat *A)
{
  PetscErrorCode ierr;
  PetscBool      create=PETSC_FALSE;
  PetscInt       m,n;

  PetscFunctionBegin;
  if (!*A) create=PETSC_TRUE;
  else {
    ierr = MatGetSize(*A,&m,&n);CHKERRQ(ierr);
    if (m!=k || n!=k) {
      ierr = MatDestroy(A);CHKERRQ(ierr);
      create=PETSC_TRUE;
    }
  }
  if (create) {
    ierr = MatCreateSeqDense(PETSC_COMM_SELF,k,k,NULL,A);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MFN_CreateVec"
/*
   MFN_CreateVec - Creates a Vec of size k unless it already has that size
*/
PETSC_STATIC_INLINE PetscErrorCode MFN_CreateVec(PetscInt k,Vec *v)
{
  PetscErrorCode ierr;
  PetscBool      create=PETSC_FALSE;
  PetscInt       n;

  PetscFunctionBegin;
  if (!*v) create=PETSC_TRUE;
  else {
    ierr = VecGetSize(*v,&n);CHKERRQ(ierr);
    if (n!=k) {
      ierr = VecDestroy(v);CHKERRQ(ierr);
      create=PETSC_TRUE;
    }
  }
  if (create) {
    ierr = VecCreateSeq(PETSC_COMM_SELF,k,v);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

PETSC_INTERN PetscErrorCode MFNBasicArnoldi(MFN,PetscScalar*,PetscInt,PetscInt,PetscInt*,PetscReal*,PetscBool*);

#endif
