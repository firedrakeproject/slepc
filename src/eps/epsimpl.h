/*
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
      SLEPc - Scalable Library for Eigenvalue Problem Computations
      Copyright (c) 2002-2007, Universidad Politecnica de Valencia, Spain

      This file is part of SLEPc. See the README file for conditions of use
      and additional information.
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
*/

#ifndef _EPSIMPL
#define _EPSIMPL

#include "slepceps.h"

extern PetscFList EPSList;
extern PetscEvent EPS_SetUp, EPS_Solve, EPS_Dense;

typedef struct _EPSOps *EPSOps;

struct _EPSOps {
  int  (*solve)(EPS);            /* one-sided solver */
  int  (*solvets)(EPS);          /* two-sided solver */
  int  (*setup)(EPS);
  int  (*setfromoptions)(EPS);
  int  (*publishoptions)(EPS);
  int  (*destroy)(EPS);
  int  (*view)(EPS,PetscViewer);
  int  (*backtransform)(EPS);
  int  (*computevectors)(EPS);
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
  int         max_it,           /* maximum number of iterations */
              nev,              /* number of eigenvalues to compute */
              ncv,              /* number of basis vectors */
              nv,               /* number of available basis vectors (<= ncv) */
              allocated_ncv,    /* number of basis vectors allocated */
              nds;              /* number of basis vectors of deflation space */
  PetscScalar target;           /* target value */
  PetscTruth  target_set;       /* flag indicating if target was specified */
  PetscReal   tol;              /* tolerance */
  EPSWhich    which;            /* which part of the spectrum to be sought */
  PetscTruth  evecsavailable;   /* computed eigenvectors */
  EPSProblemType problem_type;  /* which kind of problem to be solved */
  EPSClass    solverclass;      /* whether the selected solver is one- or two-sided */

  /*------------------------- Working data --------------------------*/
  Vec         vec_initial,      /* initial vector */
              vec_initial_left, /* left initial vector for two-sided solvers */
              *V,               /* set of basis vectors */
              *AV,              /* computed eigenvectors */
              *W,               /* set of left basis vectors */
              *AW,              /* computed left eigenvectors */
              *DS,              /* deflation space */
              *DSV;             /* deflation space and basis vectors*/
  PetscScalar *eigr, *eigi,     /* real and imaginary parts of eigenvalues */
              *T, *Tl;          /* projected matrices */
  PetscReal   *errest,          /* error estimates */
              *errest_left;     /* left error estimates */
  ST          OP;               /* spectral transformation object */
  IP          ip;               /* innerproduct object */
  void        *data;            /* placeholder for misc stuff associated 
                                   with a particular solver */
  int         nconv,            /* number of converged eigenvalues */
              its,              /* number of iterations so far computed */
              *perm;            /* permutation for eigenvalue ordering */

  /* ---------------- Default work-area and status vars -------------------- */
  int        nwork;
  Vec        *work;

  int        setupcalled;
  PetscTruth isgeneralized,
             ispositive,
             ishermitian;
  EPSConvergedReason reason;     

  int        (*monitor[MAXEPSMONITORS])(EPS,int,int,PetscScalar*,PetscScalar*,PetscReal*,int,void*); 
  int        (*monitordestroy[MAXEPSMONITORS])(void*);
  void       *monitorcontext[MAXEPSMONITORS];
  int        numbermonitors; 

  PetscTruth ds_ortho;    /* if vectors in DS have to be orthonormalized */  
};

#define EPSMonitor(eps,it,nconv,eigr,eigi,errest,nest) \
        { int _ierr,_i,_im = eps->numbermonitors; \
          for ( _i=0; _i<_im; _i++ ) {\
            _ierr=(*eps->monitor[_i])(eps,it,nconv,eigr,eigi,errest,nest,eps->monitorcontext[_i]);\
            CHKERRQ(_ierr); \
	  } \
	}

EXTERN PetscErrorCode EPSRegisterAll(char *);

EXTERN PetscErrorCode EPSDestroy_Default(EPS);
EXTERN PetscErrorCode EPSDefaultGetWork(EPS,int);
EXTERN PetscErrorCode EPSDefaultFreeWork(EPS);
EXTERN PetscErrorCode EPSAllocateSolution(EPS);
EXTERN PetscErrorCode EPSFreeSolution(EPS);
EXTERN PetscErrorCode EPSAllocateSolutionContiguous(EPS);
EXTERN PetscErrorCode EPSFreeSolutionContiguous(EPS);
EXTERN PetscErrorCode EPSBackTransform_Default(EPS);
EXTERN PetscErrorCode EPSComputeVectors_Default(EPS);
EXTERN PetscErrorCode EPSComputeVectors_Hermitian(EPS);
EXTERN PetscErrorCode EPSComputeVectors_Schur(EPS);

/* Private functions of the solver implementations */

EXTERN PetscErrorCode EPSBasicArnoldi(EPS,PetscTruth,PetscScalar*,Vec*,int,int*,Vec,PetscReal*,PetscTruth*);
EXTERN PetscErrorCode EPSDelayedArnoldi(EPS,PetscScalar*,Vec*,int,int*,Vec,PetscReal*,PetscTruth*);
EXTERN PetscErrorCode EPSDelayedArnoldi1(EPS,PetscScalar*,Vec*,int,int*,Vec,PetscReal*,PetscTruth*);
EXTERN PetscErrorCode ArnoldiResiduals(PetscScalar*,int,PetscScalar*,PetscReal,int,int,PetscScalar*,PetscScalar*,PetscReal*,PetscScalar*);

#endif
