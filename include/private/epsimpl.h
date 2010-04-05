/*
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
   SLEPc - Scalable Library for Eigenvalue Problem Computations
   Copyright (c) 2002-2009, Universidad Politecnica de Valencia, Spain

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

#include "slepceps.h"

extern PetscFList EPSList;
extern PetscLogEvent EPS_SetUp, EPS_Solve, EPS_Dense;

typedef struct _EPSOps *EPSOps;

struct _EPSOps {
  PetscErrorCode  (*solve)(EPS);            /* one-sided solver */
  PetscErrorCode  (*solvets)(EPS);          /* two-sided solver */
  PetscErrorCode  (*setup)(EPS);
  PetscErrorCode  (*setfromoptions)(EPS);
  PetscErrorCode  (*publishoptions)(EPS);
  PetscErrorCode  (*destroy)(EPS);
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
  PetscTruth     *conv;            /* convergence test */
  PetscErrorCode (*conv_func)(EPS,PetscInt,PetscInt,PetscScalar*,PetscScalar*,PetscReal*,PetscTruth*,void*);
  void           *conv_ctx;
  EPSWhich       which;            /* which part of the spectrum to be sought */
  PetscErrorCode (*which_func)(EPS,PetscScalar,PetscScalar,PetscScalar,PetscScalar,PetscInt*,void*);
  void           *which_ctx;
  EPSProblemType problem_type;     /* which kind of problem to be solved */
  EPSExtraction  extraction;       /* which kind of extraction to be applied */
  EPSClass       solverclass;      /* whether the selected solver is one- or two-sided */
  EPSBalance     balance;          /* the balancing method */
  PetscInt       balance_its;      /* number of iterations of the balancing method */
  PetscReal      balance_cutoff;   /* cutoff value for balancing */

  /*------------------------- Working data --------------------------*/
  Vec         D,                /* diagonal matrix for balancing */
              *V,               /* set of basis vectors and computed eigenvectors */
              *AV,              /* auxiliar set of basis vectors */
              *W,               /* set of left basis vectors and computed left eigenvectors */
              *IS, *ISL,        /* placeholder for references to user-provided initial space */
              *DS;              /* deflation space */
  PetscScalar *eigr, *eigi,     /* real and imaginary parts of eigenvalues */
              *T, *Tl,          /* projected matrices */
              *Z;               /* matrix from the eigenvalue decomposition of T */
  PetscReal   *errest,          /* error estimates */
              *errest_left;     /* left error estimates */
  ST          OP;               /* spectral transformation object */
  IP          ip;               /* innerproduct object */
  void        *data;            /* placeholder for misc stuff associated 
                                   with a particular solver */
  PetscInt    nconv,            /* number of converged eigenvalues */
              its,              /* number of iterations so far computed */
              *perm,            /* permutation for eigenvalue ordering */
              ldz,              /* leading dimension of Z */
              n, nloc,          /* problem dimensions (global, local) */
              allocated_ncv;    /* number of basis vectors allocated */
  PetscTruth  evecsavailable;   /* computed eigenvectors */

  /* ---------------- Default work-area and status vars -------------------- */
  PetscInt   nwork;
  Vec        *work;

  PetscTruth ds_ortho;         /* if DS vectors have been stored and orthonormalized */  
  PetscInt   setupcalled;
  PetscTruth isgeneralized,
             ispositive,
             ishermitian;
  EPSConvergedReason reason;     

  PetscErrorCode (*monitor[MAXEPSMONITORS])(EPS,PetscInt,PetscInt,PetscScalar*,PetscScalar*,PetscReal*,PetscInt,void*); 
  PetscErrorCode (*monitordestroy[MAXEPSMONITORS])(void*);
  void       *monitorcontext[MAXEPSMONITORS];
  PetscInt    numbermonitors; 
};

#define EPSMonitor(eps,it,nconv,eigr,eigi,errest,nest) \
        { PetscErrorCode _ierr; PetscInt _i,_im = eps->numbermonitors; \
          for ( _i=0; _i<_im; _i++ ) {\
            _ierr=(*eps->monitor[_i])(eps,it,nconv,eigr,eigi,errest,nest,eps->monitorcontext[_i]);\
            CHKERRQ(_ierr); \
	  } \
	}

EXTERN PetscErrorCode EPSRegisterAll(char *);
EXTERN PetscErrorCode EPSInitializePackage(char *);
EXTERN PetscErrorCode EPSFinalizePackage(void);

EXTERN PetscErrorCode EPSDestroy_Default(EPS);
EXTERN PetscErrorCode EPSDefaultGetWork(EPS,PetscInt);
EXTERN PetscErrorCode EPSDefaultFreeWork(EPS);
EXTERN PetscErrorCode EPSAllocateSolution(EPS);
EXTERN PetscErrorCode EPSFreeSolution(EPS);
EXTERN PetscErrorCode EPSBackTransform_Default(EPS);
EXTERN PetscErrorCode EPSComputeVectors_Default(EPS);
EXTERN PetscErrorCode EPSComputeVectors_Hermitian(EPS);
EXTERN PetscErrorCode EPSComputeVectors_Schur(EPS);
EXTERN PetscErrorCode EPSComputeResidualNorm_Private(EPS,PetscScalar,PetscScalar,Vec,Vec,PetscReal*);
EXTERN PetscErrorCode EPSComputeRelativeError_Private(EPS,PetscScalar,PetscScalar,Vec,Vec,PetscReal*);

/* Private functions of the solver implementations */

EXTERN PetscErrorCode EPSBasicArnoldi(EPS,PetscTruth,PetscScalar*,PetscInt,Vec*,PetscInt,PetscInt*,Vec,PetscReal*,PetscTruth*);
EXTERN PetscErrorCode EPSDelayedArnoldi(EPS,PetscScalar*,PetscInt,Vec*,PetscInt,PetscInt*,Vec,PetscReal*,PetscTruth*);
EXTERN PetscErrorCode EPSDelayedArnoldi1(EPS,PetscScalar*,PetscInt,Vec*,PetscInt,PetscInt*,Vec,PetscReal*,PetscTruth*);
EXTERN PetscErrorCode ArnoldiResiduals(PetscScalar*,PetscInt,PetscScalar*,PetscScalar*,PetscReal,PetscInt,PetscInt,PetscScalar*,PetscScalar*,PetscReal*,PetscScalar*);
EXTERN PetscErrorCode EPSFullLanczos(EPS,PetscReal*,PetscReal*,Vec*,PetscInt,PetscInt*,Vec,PetscTruth*);
EXTERN PetscErrorCode EPSTranslateHarmonic(PetscInt,PetscScalar*,PetscInt,PetscScalar,PetscScalar,PetscScalar*,PetscScalar*);
EXTERN PetscErrorCode EPSBuildBalance_Krylov(EPS);

#endif
