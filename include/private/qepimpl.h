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

#ifndef _QEPIMPL
#define _QEPIMPL

#include "slepcqep.h"

extern PetscFList QEPList;
extern PetscLogEvent QEP_SetUp, QEP_Solve, QEP_Dense;

typedef struct _QEPOps *QEPOps;

struct _QEPOps {
  PetscErrorCode  (*solve)(QEP);
  PetscErrorCode  (*setup)(QEP);
  PetscErrorCode  (*setfromoptions)(QEP);
  PetscErrorCode  (*publishoptions)(QEP);
  PetscErrorCode  (*destroy)(QEP);
  PetscErrorCode  (*view)(QEP,PetscViewer);
};

/*
     Maximum number of monitors you can run with a single QEP
*/
#define MAXQEPMONITORS 5 

/*
   Defines the QEP data structure.
*/
struct _p_QEP {
  PETSCHEADER(struct _QEPOps);
  /*------------------------- User parameters --------------------------*/
  PetscInt       max_it,           /* maximum number of iterations */
                 nev,              /* number of eigenvalues to compute */
                 ncv,              /* number of basis vectors */
                 mpd,              /* maximum dimension of projected problem */
                 nini, ninil;      /* number of initial vectors (negative means not copied yet) */
  PetscReal      tol;              /* tolerance */
  PetscReal      sfactor;          /* scaling factor of the quadratic problem */
  PetscErrorCode (*conv_func)(QEP,PetscScalar,PetscScalar,PetscReal,PetscReal*,void*);
  void           *conv_ctx;
  QEPWhich       which;            /* which part of the spectrum to be sought */
  PetscTruth     leftvecs;         /* if left eigenvectors are requested */
  PetscErrorCode (*which_func)(QEP,PetscScalar,PetscScalar,PetscScalar,PetscScalar,PetscInt*,void*);
  void           *which_ctx;
  QEPProblemType problem_type;     /* which kind of problem to be solved */
  PetscTruth     trackall;         /* whether all the residuals must be computed */

  /*------------------------- Working data --------------------------*/
  Mat         M,C,K;            /* problem matrices */
  Vec         *V,               /* set of basis vectors and computed eigenvectors */
              *W,               /* set of left basis vectors and computed left eigenvectors */
              *IS, *ISL;        /* placeholder for references to user-provided initial space */
  PetscScalar *eigr, *eigi,     /* real and imaginary parts of eigenvalues */
              *T;               /* matrix for projected eigenproblem */
  PetscReal   *errest;          /* error estimates */
  IP          ip;               /* innerproduct object */
  void        *data;            /* placeholder for misc stuff associated 
                                   with a particular solver */
  PetscInt    nconv,            /* number of converged eigenvalues */
              its,              /* number of iterations so far computed */
              *perm,            /* permutation for eigenvalue ordering */
              matvecs, linits,  /* operation counters */
              n, nloc;          /* problem dimensions (global, local) */
  PetscRandom rand;             /* random number generator */

  /* ---------------- Default work-area and status vars -------------------- */
  PetscInt   nwork;
  Vec        *work;

  PetscInt   setupcalled;
  QEPConvergedReason reason;     

  PetscErrorCode (*monitor[MAXQEPMONITORS])(QEP,PetscInt,PetscInt,PetscScalar*,PetscScalar*,PetscReal*,PetscInt,void*); 
  PetscErrorCode (*monitordestroy[MAXQEPMONITORS])(void*);
  void       *monitorcontext[MAXQEPMONITORS];
  PetscInt    numbermonitors; 
};

#define QEPMonitor(qep,it,nconv,eigr,eigi,errest,nest) \
        { PetscErrorCode _ierr; PetscInt _i,_im = qep->numbermonitors; \
          for ( _i=0; _i<_im; _i++ ) {\
            _ierr=(*qep->monitor[_i])(qep,it,nconv,eigr,eigi,errest,nest,qep->monitorcontext[_i]);\
            CHKERRQ(_ierr); \
	  } \
	}

/* context for QEPMonitorConverged */
typedef struct {
  PetscViewerASCIIMonitor viewer;
  PetscInt oldnconv;
} QEPMONITOR_CONV;
EXTERN PetscErrorCode QEPMonitorDestroy_Converged(QEPMONITOR_CONV*);

EXTERN PetscErrorCode QEPRegisterAll(char *);
EXTERN PetscErrorCode QEPInitializePackage(char *);
EXTERN PetscErrorCode QEPFinalizePackage(void);

EXTERN PetscErrorCode QEPDestroy_Default(QEP);
EXTERN PetscErrorCode QEPDefaultGetWork(QEP,PetscInt);
EXTERN PetscErrorCode QEPDefaultFreeWork(QEP);
EXTERN PetscErrorCode QEPComputeVectors_Schur(QEP);
EXTERN PetscErrorCode QEPComputeResidualNorm_Private(QEP,PetscScalar,PetscScalar,Vec,Vec,PetscReal*);
EXTERN PetscErrorCode QEPComputeRelativeError_Private(QEP,PetscScalar,PetscScalar,Vec,Vec,PetscReal*);
EXTERN PetscErrorCode QEPKrylovConvergence(QEP,PetscInt,PetscInt,PetscScalar*,PetscInt,PetscScalar*,PetscInt,PetscReal,PetscInt*,PetscScalar*);

#endif
