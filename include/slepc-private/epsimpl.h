/*
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
   SLEPc - Scalable Library for Eigenvalue Problem Computations
   Copyright (c) 2002-2011, Universitat Politecnica de Valencia, Spain

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

#ifndef _EPSIMPL
#define _EPSIMPL

#include <slepceps.h>

extern PetscFList EPSList;
extern PetscLogEvent EPS_SetUp, EPS_Solve, EPS_Dense;

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
              *DS;              /* deflation space */
  PetscScalar *eigr, *eigi,     /* real and imaginary parts of eigenvalues */
              *T, *Tl;          /* projected matrices */
  PetscReal   *errest,          /* error estimates */
              *errest_left;     /* left error estimates */
  ST          OP;               /* spectral transformation object */
  IP          ip;               /* innerproduct object */
  PS          ps;               /* projected system object */
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

  PetscBool  ds_ortho;         /* if DS vectors have been stored and orthonormalized */  
  PetscInt   setupcalled;
  PetscBool  isgeneralized,
             ispositive,
             ishermitian;
  EPSConvergedReason reason;     

  PetscErrorCode (*monitor[MAXEPSMONITORS])(EPS,PetscInt,PetscInt,PetscScalar*,PetscScalar*,PetscReal*,PetscInt,void*); 
  PetscErrorCode (*monitordestroy[MAXEPSMONITORS])(void**);
  void       *monitorcontext[MAXEPSMONITORS];
  PetscInt    numbermonitors; 
};

extern PetscErrorCode EPSMonitor(EPS,PetscInt,PetscInt,PetscScalar*,PetscScalar*,PetscReal*,PetscInt);

extern PetscErrorCode EPSReset_Default(EPS);
extern PetscErrorCode EPSDefaultGetWork(EPS,PetscInt);
extern PetscErrorCode EPSDefaultFreeWork(EPS);
extern PetscErrorCode EPSDefaultSetWhich(EPS);
extern PetscErrorCode EPSAllocateSolution(EPS);
extern PetscErrorCode EPSFreeSolution(EPS);
extern PetscErrorCode EPSBackTransform_Default(EPS);
extern PetscErrorCode EPSComputeVectors_Default(EPS);
extern PetscErrorCode EPSComputeVectors_Hermitian(EPS);
extern PetscErrorCode EPSComputeVectors_Schur(EPS);
extern PetscErrorCode EPSComputeResidualNorm_Private(EPS,PetscScalar,PetscScalar,Vec,Vec,PetscReal*);
extern PetscErrorCode EPSComputeRelativeError_Private(EPS,PetscScalar,PetscScalar,Vec,Vec,PetscReal*);
extern PetscErrorCode EPSComputeTrueResidual(EPS,PetscScalar,PetscScalar,PetscScalar*,Vec*,PetscInt,PetscReal*);

/* Private functions of the solver implementations */

extern PetscErrorCode EPSBasicArnoldi(EPS,PetscBool,PetscScalar*,PetscInt,Vec*,PetscInt,PetscInt*,Vec,PetscReal*,PetscBool*);
extern PetscErrorCode EPSDelayedArnoldi(EPS,PetscScalar*,PetscInt,Vec*,PetscInt,PetscInt*,Vec,PetscReal*,PetscBool*);
extern PetscErrorCode EPSDelayedArnoldi1(EPS,PetscScalar*,PetscInt,Vec*,PetscInt,PetscInt*,Vec,PetscReal*,PetscBool*);
extern PetscErrorCode EPSKrylovConvergence(EPS,PetscBool,PetscBool,PetscInt,PetscInt,PetscScalar*,PetscInt,PetscScalar*,Vec*,PetscInt,PetscReal,PetscReal,PetscInt*,PetscScalar*);
extern PetscErrorCode EPSFullLanczos(EPS,PetscReal*,PetscReal*,Vec*,PetscInt,PetscInt*,Vec,PetscBool*);
extern PetscErrorCode EPSTranslateHarmonic(PetscInt,PetscScalar*,PetscInt,PetscScalar,PetscScalar,PetscScalar*,PetscScalar*);
extern PetscErrorCode EPSBuildBalance_Krylov(EPS);
extern PetscErrorCode EPSProjectedKSNonsym(EPS,PetscInt,PetscScalar*,PetscInt,PetscScalar*,PetscInt);

#endif
