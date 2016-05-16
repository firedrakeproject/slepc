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

#if !defined(_EPSIMPL)
#define _EPSIMPL

#include <slepceps.h>
#include <slepc/private/slepcimpl.h>

PETSC_EXTERN PetscBool EPSRegisterAllCalled;
PETSC_EXTERN PetscErrorCode EPSRegisterAll(void);
PETSC_EXTERN PetscLogEvent EPS_SetUp,EPS_Solve;

typedef struct _EPSOps *EPSOps;

struct _EPSOps {
  PetscErrorCode (*solve)(EPS);
  PetscErrorCode (*setup)(EPS);
  PetscErrorCode (*setfromoptions)(PetscOptionItems*,EPS);
  PetscErrorCode (*publishoptions)(EPS);
  PetscErrorCode (*destroy)(EPS);
  PetscErrorCode (*reset)(EPS);
  PetscErrorCode (*view)(EPS,PetscViewer);
  PetscErrorCode (*backtransform)(EPS);
  PetscErrorCode (*computevectors)(EPS);
};

/*
     Maximum number of monitors you can run with a single EPS
*/
#define MAXEPSMONITORS 5

typedef enum { EPS_STATE_INITIAL,
               EPS_STATE_SETUP,
               EPS_STATE_SOLVED,
               EPS_STATE_EIGENVECTORS } EPSStateType;

/*
   Defines the EPS data structure.
*/
struct _p_EPS {
  PETSCHEADER(struct _EPSOps);
  /*------------------------- User parameters ---------------------------*/
  PetscInt       max_it;           /* maximum number of iterations */
  PetscInt       nev;              /* number of eigenvalues to compute */
  PetscInt       ncv;              /* number of basis vectors */
  PetscInt       mpd;              /* maximum dimension of projected problem */
  PetscInt       nini;             /* number of initial vectors (negative means not copied yet) */
  PetscInt       nds;              /* number of basis vectors of deflation space */
  PetscScalar    target;           /* target value */
  PetscReal      tol;              /* tolerance */
  EPSConv        conv;             /* convergence test */
  EPSStop        stop;             /* stopping test */
  EPSWhich       which;            /* which part of the spectrum to be sought */
  PetscReal      inta,intb;        /* interval [a,b] for spectrum slicing */
  EPSProblemType problem_type;     /* which kind of problem to be solved */
  EPSExtraction  extraction;       /* which kind of extraction to be applied */
  EPSBalance     balance;          /* the balancing method */
  PetscInt       balance_its;      /* number of iterations of the balancing method */
  PetscReal      balance_cutoff;   /* cutoff value for balancing */
  PetscBool      trueres;          /* whether the true residual norm must be computed */
  PetscBool      trackall;         /* whether all the residuals must be computed */
  PetscBool      purify;           /* whether eigenvectors need to be purified */

  /*-------------- User-provided functions and contexts -----------------*/
  PetscErrorCode (*converged)(EPS,PetscScalar,PetscScalar,PetscReal,PetscReal*,void*);
  PetscErrorCode (*convergeddestroy)(void*);
  PetscErrorCode (*stopping)(EPS,PetscInt,PetscInt,PetscInt,PetscInt,EPSConvergedReason*,void*);
  PetscErrorCode (*stoppingdestroy)(void*);
  PetscErrorCode (*arbitrary)(PetscScalar,PetscScalar,Vec,Vec,PetscScalar*,PetscScalar*,void*);
  void           *convergedctx;
  void           *stoppingctx;
  void           *arbitraryctx;
  PetscErrorCode (*monitor[MAXEPSMONITORS])(EPS,PetscInt,PetscInt,PetscScalar*,PetscScalar*,PetscReal*,PetscInt,void*);
  PetscErrorCode (*monitordestroy[MAXEPSMONITORS])(void**);
  void           *monitorcontext[MAXEPSMONITORS];
  PetscInt       numbermonitors;

  /*----------------- Child objects and working data -------------------*/
  ST             st;               /* spectral transformation object */
  DS             ds;               /* direct solver object */
  BV             V;                /* set of basis vectors and computed eigenvectors */
  RG             rg;               /* optional region for filtering */
  SlepcSC        sc;               /* sorting criterion data */
  Vec            D;                /* diagonal matrix for balancing */
  Vec            *IS;              /* references to user-provided initial space */
  Vec            *defl;            /* references to user-provided deflation space */
  PetscScalar    *eigr,*eigi;      /* real and imaginary parts of eigenvalues */
  PetscReal      *errest;          /* error estimates */
  PetscScalar    *rr,*ri;          /* values computed by user's arbitrary selection function */
  PetscInt       *perm;            /* permutation for eigenvalue ordering */
  PetscInt       nwork;            /* number of work vectors */
  Vec            *work;            /* work vectors */
  void           *data;            /* placeholder for solver-specific stuff */

  /* ----------------------- Status variables --------------------------*/
  EPSStateType   state;            /* initial -> setup -> solved -> eigenvectors */
  PetscInt       nconv;            /* number of converged eigenvalues */
  PetscInt       its;              /* number of iterations so far computed */
  PetscInt       n,nloc;           /* problem dimensions (global, local) */
  PetscReal      nrma,nrmb;        /* computed matrix norms */
  PetscBool      isgeneralized;
  PetscBool      ispositive;
  PetscBool      ishermitian;
  EPSConvergedReason reason;
};

/*
    Macros to test valid EPS arguments
*/
#if !defined(PETSC_USE_DEBUG)

#define EPSCheckSolved(h,arg) do {} while (0)

#else

#define EPSCheckSolved(h,arg) \
  do { \
    if (h->state<EPS_STATE_SOLVED) SETERRQ1(PetscObjectComm((PetscObject)h),PETSC_ERR_ARG_WRONGSTATE,"Must call EPSSolve() first: Parameter #%d",arg); \
  } while (0)

#endif

#undef __FUNCT__
#define __FUNCT__ "EPS_SetInnerProduct"
/*
  EPS_SetInnerProduct - set B matrix for inner product if appropriate.
*/
PETSC_STATIC_INLINE PetscErrorCode EPS_SetInnerProduct(EPS eps)
{
  PetscErrorCode ierr;
  Mat            B;

  PetscFunctionBegin;
  if (!eps->V) { ierr = EPSGetBV(eps,&eps->V);CHKERRQ(ierr); }
  if (eps->ispositive || (eps->isgeneralized && eps->ishermitian)) {
    ierr = STGetBilinearForm(eps->st,&B);CHKERRQ(ierr);
    ierr = BVSetMatrix(eps->V,B,PetscNot(eps->ispositive));CHKERRQ(ierr);
    ierr = MatDestroy(&B);CHKERRQ(ierr);
  } else {
    ierr = BVSetMatrix(eps->V,NULL,PETSC_FALSE);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

PETSC_INTERN PetscErrorCode EPSSetWhichEigenpairs_Default(EPS);
PETSC_INTERN PetscErrorCode EPSSetDimensions_Default(EPS,PetscInt,PetscInt*,PetscInt*);
PETSC_INTERN PetscErrorCode EPSBackTransform_Default(EPS);
PETSC_INTERN PetscErrorCode EPSComputeVectors(EPS);
PETSC_INTERN PetscErrorCode EPSComputeVectors_Hermitian(EPS);
PETSC_INTERN PetscErrorCode EPSComputeVectors_Schur(EPS);
PETSC_INTERN PetscErrorCode EPSComputeVectors_Indefinite(EPS);
PETSC_INTERN PetscErrorCode EPSComputeVectors_Slice(EPS);
PETSC_INTERN PetscErrorCode EPSComputeResidualNorm_Private(EPS,PetscScalar,PetscScalar,Vec,Vec,Vec*,PetscReal*);
PETSC_INTERN PetscErrorCode EPSComputeRitzVector(EPS,PetscScalar*,PetscScalar*,BV,Vec,Vec);
PETSC_INTERN PetscErrorCode EPSGetStartVector(EPS,PetscInt,PetscBool*);

/* Private functions of the solver implementations */

PETSC_INTERN PetscErrorCode EPSBasicArnoldi(EPS,PetscBool,PetscScalar*,PetscInt,PetscInt,PetscInt*,PetscReal*,PetscBool*);
PETSC_INTERN PetscErrorCode EPSDelayedArnoldi(EPS,PetscScalar*,PetscInt,PetscInt,PetscInt*,PetscReal*,PetscBool*);
PETSC_INTERN PetscErrorCode EPSDelayedArnoldi1(EPS,PetscScalar*,PetscInt,PetscInt,PetscInt*,PetscReal*,PetscBool*);
PETSC_INTERN PetscErrorCode EPSKrylovConvergence(EPS,PetscBool,PetscInt,PetscInt,PetscReal,PetscReal,PetscInt*);
PETSC_INTERN PetscErrorCode EPSFullLanczos(EPS,PetscReal*,PetscReal*,PetscInt,PetscInt*,PetscBool*);
PETSC_INTERN PetscErrorCode EPSPseudoLanczos(EPS,PetscReal*,PetscReal*,PetscReal*,PetscInt,PetscInt*,PetscBool*,PetscBool*,PetscReal*,Vec);
PETSC_INTERN PetscErrorCode EPSBuildBalance_Krylov(EPS);

#endif
