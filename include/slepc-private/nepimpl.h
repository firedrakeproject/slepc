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

#if !defined(_NEPIMPL)
#define _NEPIMPL

#include <slepcnep.h>
#include <slepc-private/slepcimpl.h>

PETSC_EXTERN PetscLogEvent NEP_SetUp,NEP_Solve,NEP_FunctionEval,NEP_JacobianEval;

typedef struct _NEPOps *NEPOps;

struct _NEPOps {
  PetscErrorCode (*solve)(NEP);
  PetscErrorCode (*setup)(NEP);
  PetscErrorCode (*setfromoptions)(NEP);
  PetscErrorCode (*publishoptions)(NEP);
  PetscErrorCode (*destroy)(NEP);
  PetscErrorCode (*reset)(NEP);
  PetscErrorCode (*view)(NEP,PetscViewer);
};

/*
     Maximum number of monitors you can run with a single NEP
*/
#define MAXNEPMONITORS 5

/*
   Defines the NEP data structure.
*/
struct _p_NEP {
  PETSCHEADER(struct _NEPOps);
  /*------------------------- User parameters ---------------------------*/
  PetscInt       max_it;           /* maximum number of iterations */
  PetscInt       max_funcs;        /* maximum number of function evaluations */
  PetscInt       nev;              /* number of eigenvalues to compute */
  PetscInt       ncv;              /* number of basis vectors */
  PetscInt       mpd;              /* maximum dimension of projected problem */
  PetscInt       lag;              /* interval to rebuild preconditioner */
  PetscInt       nini;             /* number of initial vectors (negative means not copied yet) */
  PetscScalar    target;           /* target value */
  PetscReal      abstol,rtol,stol; /* user tolerances */
  PetscReal      ktol;             /* tolerance for linear solver */
  PetscBool      cctol;            /* constant correction tolerance */
  PetscReal      ttol;             /* tolerance used in the convergence criterion */
  NEPWhich       which;            /* which part of the spectrum to be sought */
  PetscBool      trackall;         /* whether all the residuals must be computed */

  /*-------------- User-provided functions and contexts -----------------*/
  PetscErrorCode (*computefunction)(NEP,PetscScalar,Mat,Mat,void*);
  PetscErrorCode (*computejacobian)(NEP,PetscScalar,Mat,void*);
  void           *functionctx;
  void           *jacobianctx;
  PetscErrorCode (*comparison)(PetscScalar,PetscScalar,PetscScalar,PetscScalar,PetscInt*,void*);
  PetscErrorCode (*converged)(NEP,PetscInt,PetscReal,PetscReal,PetscReal,NEPConvergedReason*,void*);
  PetscErrorCode (*convergeddestroy)(void*);
  void           *comparisonctx;
  void           *convergedctx;
  PetscErrorCode (*monitor[MAXNEPMONITORS])(NEP,PetscInt,PetscInt,PetscScalar*,PetscReal*,PetscInt,void*);
  PetscErrorCode (*monitordestroy[MAXNEPMONITORS])(void**);
  void           *monitorcontext[MAXNEPMONITORS];
  PetscInt       numbermonitors;

  /*----------------- Child objects and working data -------------------*/
  DS             ds;               /* direct solver object */
  BV             V;                /* set of basis vectors and computed eigenvectors */
  PetscRandom    rand;             /* random number generator */
  KSP            ksp;              /* linear solver object */
  Mat            function;         /* function matrix */
  Mat            function_pre;     /* function matrix (preconditioner) */
  Mat            jacobian;         /* Jacobian matrix */
  Mat            *A;               /* matrix coefficients of split form */
  FN             *f;               /* matrix functions of split form */
  PetscInt       nt;               /* number of terms in split form */
  MatStructure   mstr;             /* pattern of split matrices */
  Vec            *IS;              /* references to user-provided initial space */
  PetscScalar    *eig;             /* computed eigenvalues */
  PetscReal      *errest;          /* error estimates */
  PetscInt       *perm;            /* permutation for eigenvalue ordering */
  PetscInt       nwork;            /* number of work vectors */
  Vec            *work;            /* work vectors */
  void           *data;            /* placeholder for solver-specific stuff */

  /* ----------------------- Status variables --------------------------*/
  PetscInt       nconv;            /* number of converged eigenvalues */
  PetscInt       its;              /* number of iterations so far computed */
  PetscInt       n,nloc;           /* problem dimensions (global, local) */
  PetscInt       nfuncs;           /* number of function evaluations */
  PetscBool      split;            /* the nonlinear operator has been set in
                                      split form, otherwise user callbacks are used */
  PetscInt       setupcalled;
  NEPConvergedReason reason;
};

#undef __FUNCT__
#define __FUNCT__ "NEP_KSPSolve"
PETSC_STATIC_INLINE PetscErrorCode NEP_KSPSolve(NEP nep,Vec b,Vec x)
{
  PetscErrorCode ierr;
  PetscInt       lits;

  PetscFunctionBegin;
  ierr = KSPSolve(nep->ksp,b,x);CHKERRQ(ierr);
  ierr = KSPGetIterationNumber(nep->ksp,&lits);CHKERRQ(ierr);
  ierr = PetscInfo2(nep,"iter=%D, linear solve iterations=%D\n",nep->its,lits);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PETSC_INTERN PetscErrorCode NEPGetDefaultShift(NEP,PetscScalar*);
PETSC_INTERN PetscErrorCode NEPAllocateSolution(NEP,PetscInt);
PETSC_INTERN PetscErrorCode NEPComputeVectors_Schur(NEP);
PETSC_INTERN PetscErrorCode NEPComputeResidualNorm_Private(NEP,PetscScalar,Vec,PetscReal*);
PETSC_INTERN PetscErrorCode NEPComputeRelativeError_Private(NEP,PetscScalar,Vec,PetscReal*);
PETSC_INTERN PetscErrorCode NEPKrylovConvergence(NEP,PetscBool,PetscInt,PetscInt,PetscInt,PetscReal,PetscInt*);

#endif
