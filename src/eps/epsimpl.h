
#ifndef _EPSIMPL
#define _EPSIMPL

#include "slepceps.h"

typedef struct _EPSOps *EPSOps;

struct _EPSOps {
  int  (*solve)(EPS);                   /* actual solver */
  int  (*setup)(EPS);
  int  (*setdefaults)(EPS);
  int  (*setfromoptions)(EPS);
  int  (*publishoptions)(EPS);
  int  (*destroy)(EPS);
  int  (*view)(EPS,PetscViewer);
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
             ncv;               /* number of basis vectors */
  PetscReal  tol;               /* tolerance */
  EPSWhich   which;             /* which part of the spectrum to be sought */
  PetscTruth dropvectors;       /* do not compute eigenvectors */
  EPSProblemType problem_type;  /* which kind of problem to be solved */

  /*------------------------- Working data --------------------------*/
  Vec         vec_initial;      /* initial vector for iterative methods */
  Vec         *V;               /* set of basis vectors */
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
             ishermitian;
  EPSConvergedReason reason;     

  int        (*monitor[MAXEPSMONITORS])(EPS,int,int,PetscReal*,int,void*); 
  int        (*monitordestroy[MAXEPSMONITORS])(void*);
  void       *monitorcontext[MAXEPSMONITORS];
  int        numbermonitors; 
  int        (*vmonitor[MAXEPSMONITORS])(EPS,int,int,PetscScalar*,PetscScalar*,int,void*); 
  int        (*vmonitordestroy[MAXEPSMONITORS])(void*);
  void       *vmonitorcontext[MAXEPSMONITORS];
  int        numbervmonitors; 

  /* --------------- Orthogonalization --------------------- */
  int        (*orthog)(EPS,int,PetscScalar*);
  EPSOrthogonalizationType orth_type;   /* which orthogonalization to use */
  
};

#define EPSMonitorEstimates(eps,it,nconv,errest,nest) \
        { int _ierr,_i,_im = eps->numbermonitors; \
          for ( _i=0; _i<_im; _i++ ) {\
            _ierr=(*eps->monitor[_i])(eps,it,nconv,errest,nest,eps->monitorcontext[_i]);\
            CHKERRQ(_ierr); \
	  } \
	}

#define EPSMonitorValues(eps,it,nconv,eigr,eigi,neig) \
        { int _ierr,_i,_im = eps->numbervmonitors; \
          for ( _i=0; _i<_im; _i++ ) {\
            _ierr=(*eps->vmonitor[_i])(eps,it,nconv,eigr,eigi,neig,eps->monitorcontext[_i]);\
            CHKERRQ(_ierr); \
	  } \
	}

extern int EPSDefaultDestroy(EPS);
extern int EPSDefaultGetWork(EPS,int);
extern int EPSDefaultFreeWork(EPS);
extern int EPSModifiedGramSchmidtOrthogonalization(EPS,int,PetscScalar*);
extern int EPSUnmodifiedGramSchmidtOrthogonalization(EPS,int,PetscScalar*);
extern int EPSIROrthogonalization(EPS,int,PetscScalar*);

#endif
