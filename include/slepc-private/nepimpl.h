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

#if !defined(_NEPIMPL)
#define _NEPIMPL

#include <slepcnep.h>
#include <slepc-private/slepcimpl.h>

PETSC_EXTERN PetscLogEvent NEP_SetUp,NEP_Solve,NEP_Dense,NEP_FunctionEval,NEP_JacobianEval;

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
  /*------------------------- User parameters --------------------------*/
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
  PetscBool      trackall;         /* whether all the residuals must be computed */
  NEPWhich       which;            /* which part of the spectrum to be sought */
  PetscBool      split;            /* the nonlinear operator has been set in
                                      split form, otherwise user callbacks are used */

  /*-------------- User-provided functions and contexts -----------------*/
  PetscErrorCode (*computefunction)(NEP,PetscScalar,PetscScalar,Mat*,Mat*,MatStructure*,void*);
  PetscErrorCode (*computejacobian)(NEP,PetscScalar,PetscScalar,Mat*,MatStructure*,void*);
  PetscErrorCode (*comparison)(PetscScalar,PetscScalar,PetscScalar,PetscScalar,PetscInt*,void*);
  PetscErrorCode (*converged)(NEP,PetscInt,PetscReal,PetscReal,PetscReal,NEPConvergedReason*,void*);
  PetscErrorCode (*convergeddestroy)(void*);
  void           *comparisonctx;
  void           *convergedctx;
  Mat            function,function_pre;
  void           *functionctx;
  Mat            jacobian;
  void           *jacobianctx;
  PetscInt       nt;               /* number of terms in split form */
  MatStructure   mstr;             /* pattern of split matrices */
  Mat            *A;               /* matrix coefficients of split form */
  FN             *f;               /* matrix functions of split form */

  /*------------------------- Working data --------------------------*/
  Vec            *V;               /* set of basis vectors and computed eigenvectors */
  Vec            *IS;              /* placeholder for references to user-provided initial space */
  PetscScalar    *eigr,*eigi;      /* real and imaginary parts of eigenvalues */
  PetscReal      *errest;          /* error estimates */
  IP             ip;               /* innerproduct object */
  DS             ds;               /* direct solver object */
  KSP            ksp;              /* linear solver object */
  void           *data;            /* placeholder for misc stuff associated
                                      with a particular solver */
  PetscInt       nconv;            /* number of converged eigenvalues */
  PetscInt       its;              /* number of iterations so far computed */
  PetscInt       *perm;            /* permutation for eigenvalue ordering */
  PetscInt       nfuncs,linits;    /* operation counters */
  PetscInt       n,nloc;           /* problem dimensions (global, local) */
  PetscRandom    rand;             /* random number generator */
  Vec            t;                /* template vector */
  PetscInt       allocated_ncv;    /* number of basis vectors allocated */

  /* ---------------- Default work-area and status vars -------------------- */
  PetscInt       nwork;
  Vec            *work;

  PetscInt       setupcalled;
  NEPConvergedReason reason;

  PetscErrorCode (*monitor[MAXNEPMONITORS])(NEP,PetscInt,PetscInt,PetscScalar*,PetscScalar*,PetscReal*,PetscInt,void*);
  PetscErrorCode (*monitordestroy[MAXNEPMONITORS])(void**);
  void           *monitorcontext[MAXNEPMONITORS];
  PetscInt       numbermonitors;
};

PETSC_INTERN PetscErrorCode NEPReset_Default(NEP);
PETSC_INTERN PetscErrorCode NEPGetDefaultShift(NEP,PetscScalar*);
PETSC_INTERN PetscErrorCode NEPAllocateSolution(NEP);
PETSC_INTERN PetscErrorCode NEPFreeSolution(NEP);
PETSC_INTERN PetscErrorCode NEP_KSPSolve(NEP,Vec,Vec);
PETSC_INTERN PetscErrorCode NEPComputeResidualNorm_Private(NEP,PetscScalar,PetscScalar,Vec,Vec,PetscReal*);
PETSC_INTERN PetscErrorCode NEPComputeRelativeError_Private(NEP,PetscScalar,PetscScalar,Vec,Vec,PetscReal*);

#endif
