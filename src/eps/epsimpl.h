
#ifndef _EPSIMPL
#define _EPSIMPL

#include "slepceps.h"

typedef struct _EPSOps *EPSOps;

struct _EPSOps {
  int  (*solve)(EPS);                   /* actual solver */
  int  (*setup)(EPS);
  int  (*setfromoptions)(EPS);
  int  (*publishoptions)(EPS);
  int  (*destroy)(EPS);
  int  (*view)(EPS,PetscViewer);
  int  (*backtransform)(EPS);
};

/*
     Maximum number of monitors you can run with a single EPS
*/
#define MAXEPSMONITORS 5 

/*
   Defines the EPS data structure.
*/
struct _p_EPS {
  PETSCHEADER(struct _EPSOps)
  /*------------------------- User parameters --------------------------*/
  int        max_it,            /* maximum number of iterations */
             nev,               /* number of eigenvalues to compute */
             ncv,               /* number of basis vectors */
             allocated_ncv,     /* number of basis vectors allocated */
             nds;               /* number of basis vectors of deflation space */
  PetscReal  tol;               /* tolerance */
  EPSWhich   which;             /* which part of the spectrum to be sought */
  PetscTruth evecsavailable;   /* computed eigenvectors */
  EPSProblemType problem_type;  /* which kind of problem to be solved */

  /*------------------------- Working data --------------------------*/
  Vec         vec_initial;      /* initial vector for iterative methods */
  Vec         *V,               /* set of basis vectors */
              *AV,              /* computed eigen vectors */
              *DS,              /* deflation space */
              *DSV;             /* deflation space and basis vectors*/
  PetscScalar *eigr, *eigi;     /* real and imaginary parts of eigenvalues */
  PetscReal  *errest;           /* error estimates */
  ST         OP;                /* spectral transformation object */
  void       *data;             /* holder for misc stuff associated 
                                   with a particular solver */
  int        nconv,             /* number of converged eigenvalues */
             its;               /* number of iterations so far computed */
  int        *perm;             /* permutation for eigenvalue ordering */

  /* ---------------- Default work-area and status vars -------------------- */
  int        nwork;
  Vec        *work;

  int        setupcalled;
  PetscTruth isgeneralized,
             ishermitian,
             vec_initial_set;
  EPSConvergedReason reason;     

  int        (*monitor[MAXEPSMONITORS])(EPS,int,int,PetscScalar*,PetscScalar*,PetscReal*,int,void*); 
  int        (*monitordestroy[MAXEPSMONITORS])(void*);
  void       *monitorcontext[MAXEPSMONITORS];
  int        numbermonitors; 

  int        (*computevectors)(EPS);

  /* --------------- Orthogonalization --------------------- */
  int        (*orthog)(EPS,int,Vec*,Vec,PetscScalar*,PetscReal*);
  PetscReal  orth_eta;
  PetscTruth ds_ortho;     /* if vectors in DS have to be orthonormalized */
  EPSOrthogonalizationType orth_type;   /* which orthogonalization to use */
  
};

#define EPSMonitor(eps,it,nconv,eigr,eigi,errest,nest) \
        { int _ierr,_i,_im = eps->numbermonitors; \
          for ( _i=0; _i<_im; _i++ ) {\
            _ierr=(*eps->monitor[_i])(eps,it,nconv,eigr,eigi,errest,nest,eps->monitorcontext[_i]);\
            CHKERRQ(_ierr); \
	  } \
	}

EXTERN PetscErrorCode EPSDestroy_Default(EPS);
EXTERN PetscErrorCode EPSDefaultGetWork(EPS,int);
EXTERN PetscErrorCode EPSDefaultFreeWork(EPS);
EXTERN PetscErrorCode EPSAllocateSolution(EPS);
EXTERN PetscErrorCode EPSFreeSolution(EPS);
EXTERN PetscErrorCode EPSAllocateSolutionContiguous(EPS);
EXTERN PetscErrorCode EPSFreeSolutionContiguous(EPS);
EXTERN PetscErrorCode EPSModifiedGramSchmidtOrthogonalization(EPS,int,Vec*,Vec,PetscScalar*,PetscReal*);
EXTERN PetscErrorCode EPSClassicalGramSchmidtOrthogonalization(EPS,int,Vec*,Vec,PetscScalar*,PetscReal*);
EXTERN PetscErrorCode EPSBackTransform_Default(EPS);
EXTERN PetscErrorCode EPSComputeVectors_Default(EPS);

#endif
