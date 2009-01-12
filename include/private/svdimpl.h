/*
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
      SLEPc - Scalable Library for Eigenvalue Problem Computations
      Copyright (c) 2002-2007, Universidad Politecnica de Valencia, Spain

      This file is part of SLEPc. See the README file for conditions of use
      and additional information.
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
*/

#ifndef _SVDIMPL
#define _SVDIMPL

#include "slepcsvd.h"
#include "slepcip.h"

extern PetscFList SVDList;
extern PetscLogEvent SVD_SetUp, SVD_Solve, SVD_Dense;

typedef struct _SVDOps *SVDOps;

struct _SVDOps {
  PetscErrorCode  (*solve)(SVD);
  PetscErrorCode  (*setup)(SVD);
  PetscErrorCode  (*setfromoptions)(SVD);
  PetscErrorCode  (*publishoptions)(SVD);
  PetscErrorCode  (*destroy)(SVD);
  PetscErrorCode  (*view)(SVD,PetscViewer);
};

/*
     Maximum number of monitors you can run with a single SVD
*/
#define MAXSVDMONITORS 5 

/*
   Defines the SVD data structure.
*/
struct _p_SVD {
  PETSCHEADER(struct _SVDOps);
  Mat              OP;          /* problem matrix */
  Mat              A;	        /* problem matrix (m>n) */
  Mat              AT;          /* transposed matrix */
  SVDTransposeMode transmode;   /* transpose mode */
  PetscReal        *sigma;	/* singular values */
  PetscInt         *perm;       /* permutation for singular value ordering */
  Vec              *U,*V;	/* left and right singular vectors */
  Vec              vec_initial; /* initial vector */
  PetscInt         n;           /* maximun size of descomposition */
  SVDWhich         which;       /* which singular values are computed */
  PetscInt         nconv;	/* number of converged values */
  PetscInt         nsv;         /* number of requested values */
  PetscInt         ncv;         /* basis size */
  PetscInt         mpd;         /* maximum dimension of projected problem */
  PetscInt         its;         /* iteration counter */
  PetscInt         max_it;      /* max iterations */
  PetscReal        tol;         /* tolerance */
  PetscReal        *errest;     /* error estimates */
  void             *data;	/* placeholder for misc stuff associated
                   		   with a particular solver */
  PetscInt         setupcalled;
  SVDConvergedReason reason;
  IP               ip;
  
  PetscErrorCode  (*monitor[MAXSVDMONITORS])(SVD,PetscInt,PetscInt,PetscReal*,PetscReal*,PetscInt,void*);
  PetscErrorCode  (*monitordestroy[MAXSVDMONITORS])(void*);
  void            *monitorcontext[MAXSVDMONITORS];
  PetscInt        numbermonitors;
  
  PetscInt        matvecs;
};

EXTERN PetscErrorCode SVDRegisterAll(char *);

#define SVDMonitor(svd,it,nconv,sigma,errest,nest) \
        { PetscErrorCode _ierr; PetscInt _i,_im = svd->numbermonitors; \
          for ( _i=0; _i<_im; _i++ ) {\
            _ierr=(*svd->monitor[_i])(svd,it,nconv,sigma,errest,nest,svd->monitorcontext[_i]);\
            CHKERRQ(_ierr); \
	  } \
	}

#endif
EXTERN PetscErrorCode SVDDestroy_Default(SVD);
EXTERN PetscErrorCode SVDMatMult(SVD,PetscTruth,Vec,Vec);
EXTERN PetscErrorCode SVDMatGetVecs(SVD,Vec*,Vec*);
EXTERN PetscErrorCode SVDMatGetSize(SVD,PetscInt*,PetscInt*);
EXTERN PetscErrorCode SVDMatGetLocalSize(SVD,PetscInt*,PetscInt*);
EXTERN PetscErrorCode SVDTwoSideLanczos(SVD,PetscReal*,PetscReal*,Vec*,Vec,Vec*,PetscInt,PetscInt,PetscScalar*,Vec,Vec);

