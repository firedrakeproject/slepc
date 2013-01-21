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

#if !defined(_EPSIMPL)
#define _EPSIMPL

#include <slepceps.h>

PETSC_EXTERN PetscFunctionList EPSList;
PETSC_EXTERN PetscLogEvent     EPS_SetUp, EPS_Solve;

typedef struct _EPSOps *EPSOps;

struct _EPSOps {
  PetscErrorCode  (*solve)(EPS);
  PetscErrorCode  (*setup)(EPS);
  PetscErrorCode  (*setfromoptions)(EPS);
  PetscErrorCode  (*publishoptions)(EPS);
  PetscErrorCode  (*destroy)(EPS);
  PetscErrorCode  (*reset)(EPS);
  PetscErrorCode  (*view)(EPS,PetscViewer);
  PetscErrorCode  (*backtransform)(EPS);
  PetscErrorCode  (*computevectors)(EPS);
};

/*
     Maximum number of monitors you can run with a single EPS
*/
#define MAXEPSMONITORS 5 

/*
   Defines the EPS data structure.
*/
struct _p_EPS {
  PETSCHEADER(struct _EPSOps);
  /*------------------------- User parameters --------------------------*/
  PetscInt       max_it,           /* maximum number of iterations */
                 nev,              /* number of eigenvalues to compute */
                 ncv,              /* number of basis vectors */
                 mpd,              /* maximum dimension of projected problem */
                 nini, ninil,      /* number of initial vectors (negative means not copied yet) */
                 nds;              /* number of basis vectors of deflation space */
  PetscScalar    target;           /* target value */
  PetscReal      tol;              /* tolerance */
  EPSConv        conv;             /* convergence test */
  PetscErrorCode (*conv_func)(EPS,PetscScalar,PetscScalar,PetscReal,PetscReal*,void*);
  void           *conv_ctx;
  EPSWhich       which;            /* which part of the spectrum to be sought */
  PetscBool      leftvecs;         /* if left eigenvectors are requested */
  PetscErrorCode (*which_func)(PetscScalar,PetscScalar,PetscScalar,PetscScalar,PetscInt*,void*);
  void           *which_ctx;
  PetscErrorCode (*arbit_func)(PetscScalar,PetscScalar,Vec,Vec,PetscScalar*,PetscScalar*,void*);
  void           *arbit_ctx;
  PetscScalar    *rr,*ri;          /* values computed by user's arbitrary selection function */
  PetscReal      inta, intb;       /* interval [a,b] for spectrum slicing */
  EPSProblemType problem_type;     /* which kind of problem to be solved */
  EPSExtraction  extraction;       /* which kind of extraction to be applied */
  EPSBalance     balance;          /* the balancing method */
  PetscInt       balance_its;      /* number of iterations of the balancing method */
  PetscReal      balance_cutoff;   /* cutoff value for balancing */
  PetscReal      nrma, nrmb;       /* matrix norms */
  PetscBool      adaptive;         /* whether matrix norms are adaptively improved */
  PetscBool      trueres;          /* whether the true residual norm must be computed */
  PetscBool      trackall;         /* whether all the residuals must be computed */

  /*------------------------- Working data --------------------------*/
  Vec         D,                /* diagonal matrix for balancing */
              *V,               /* set of basis vectors and computed eigenvectors */
              *W,               /* set of left basis vectors and computed left eigenvectors */
              *IS, *ISL,        /* placeholder for references to user-provided initial space */
              *defl;            /* deflation space */
  PetscScalar *eigr, *eigi;     /* real and imaginary parts of eigenvalues */
  PetscReal   *errest,          /* error estimates */
              *errest_left;     /* left error estimates */
  ST          st;               /* spectral transformation object */
  IP          ip;               /* innerproduct object */
  DS          ds;               /* direct solver object */
  void        *data;            /* placeholder for misc stuff associated 
                                   with a particular solver */
  PetscInt    nconv,            /* number of converged eigenvalues */
              its,              /* number of iterations so far computed */
              *perm,            /* permutation for eigenvalue ordering */
              nv,               /* size of current Schur decomposition */
              n, nloc,          /* problem dimensions (global, local) */
              allocated_ncv;    /* number of basis vectors allocated */
  PetscBool   evecsavailable;   /* computed eigenvectors */
  PetscRandom rand;             /* random number generator */
  Vec         t;                /* template vector */

  /* ---------------- Default work-area and status vars -------------------- */
  PetscInt   nwork;
  Vec        *work;

  PetscBool  ds_ortho;         /* if defl vectors have been stored & orthonormalized */  
  PetscInt   setupcalled;
  PetscBool  isgeneralized,
             ispositive,
             ishermitian;
  EPSConvergedReason reason;     

  PetscErrorCode (*monitor[MAXEPSMONITORS])(EPS,PetscInt,PetscInt,PetscScalar*,PetscScalar*,PetscReal*,PetscInt,void*); 
  PetscErrorCode (*monitordestroy[MAXEPSMONITORS])(void**);
  void           *monitorcontext[MAXEPSMONITORS];
  PetscInt       numbermonitors; 
};

PETSC_EXTERN PetscErrorCode EPSMonitor(EPS,PetscInt,PetscInt,PetscScalar*,PetscScalar*,PetscReal*,PetscInt);

PETSC_EXTERN PetscErrorCode EPSReset_Default(EPS);
PETSC_EXTERN PetscErrorCode EPSDefaultGetWork(EPS,PetscInt);
PETSC_EXTERN PetscErrorCode EPSDefaultFreeWork(EPS);
PETSC_EXTERN PetscErrorCode EPSDefaultSetWhich(EPS);
PETSC_EXTERN PetscErrorCode EPSAllocateSolution(EPS);
PETSC_EXTERN PetscErrorCode EPSFreeSolution(EPS);
PETSC_EXTERN PetscErrorCode EPSBackTransform_Default(EPS);
PETSC_EXTERN PetscErrorCode EPSComputeVectors_Default(EPS);
PETSC_EXTERN PetscErrorCode EPSComputeVectors_Hermitian(EPS);
PETSC_EXTERN PetscErrorCode EPSComputeVectors_Schur(EPS);
PETSC_EXTERN PetscErrorCode EPSComputeResidualNorm_Private(EPS,PetscScalar,PetscScalar,Vec,Vec,PetscReal*);
PETSC_EXTERN PetscErrorCode EPSComputeRelativeError_Private(EPS,PetscScalar,PetscScalar,Vec,Vec,PetscReal*);
PETSC_EXTERN PetscErrorCode EPSComputeRitzVector(EPS,PetscScalar*,PetscScalar*,Vec*,PetscInt,Vec,Vec);

/* Private functions of the solver implementations */

PETSC_EXTERN PetscErrorCode EPSBasicArnoldi(EPS,PetscBool,PetscScalar*,PetscInt,Vec*,PetscInt,PetscInt*,Vec,PetscReal*,PetscBool*);
PETSC_EXTERN PetscErrorCode EPSDelayedArnoldi(EPS,PetscScalar*,PetscInt,Vec*,PetscInt,PetscInt*,Vec,PetscReal*,PetscBool*);
PETSC_EXTERN PetscErrorCode EPSDelayedArnoldi1(EPS,PetscScalar*,PetscInt,Vec*,PetscInt,PetscInt*,Vec,PetscReal*,PetscBool*);
PETSC_EXTERN PetscErrorCode EPSKrylovConvergence(EPS,PetscBool,PetscInt,PetscInt,Vec*,PetscInt,PetscReal,PetscReal,PetscInt*);
PETSC_EXTERN PetscErrorCode EPSFullLanczos(EPS,PetscReal*,PetscReal*,Vec*,PetscInt,PetscInt*,Vec,PetscBool*);
PETSC_EXTERN PetscErrorCode EPSBuildBalance_Krylov(EPS);

#endif
