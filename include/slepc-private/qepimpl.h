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

#ifndef _QEPIMPL
#define _QEPIMPL

#include <slepcqep.h>

PETSC_EXTERN PetscFList QEPList;
PETSC_EXTERN PetscLogEvent QEP_SetUp, QEP_Solve, QEP_Dense;

typedef struct _QEPOps *QEPOps;

struct _QEPOps {
  PetscErrorCode  (*solve)(QEP);
  PetscErrorCode  (*setup)(QEP);
  PetscErrorCode  (*setfromoptions)(QEP);
  PetscErrorCode  (*publishoptions)(QEP);
  PetscErrorCode  (*destroy)(QEP);
  PetscErrorCode  (*reset)(QEP);
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
                 nini, ninil,      /* number of initial vectors (negative means not copied yet) */
                 allocated_ncv;    /* number of basis vectors allocated */
  PetscReal      tol;              /* tolerance */
  PetscReal      sfactor;          /* scaling factor of the quadratic problem */
  PetscErrorCode (*conv_func)(QEP,PetscScalar,PetscScalar,PetscReal,PetscReal*,void*);
  void           *conv_ctx;
  QEPWhich       which;            /* which part of the spectrum to be sought */
  PetscBool      leftvecs;         /* if left eigenvectors are requested */
  PetscErrorCode (*which_func)(PetscScalar,PetscScalar,PetscScalar,PetscScalar,PetscInt*,void*);
  void           *which_ctx;
  QEPProblemType problem_type;     /* which kind of problem to be solved */
  PetscBool      trackall;         /* whether all the residuals must be computed */

  /*------------------------- Working data --------------------------*/
  Mat         M,C,K;            /* problem matrices */
  Vec         *V,               /* set of basis vectors and computed eigenvectors */
              *W,               /* set of left basis vectors and computed left eigenvectors */
              *IS, *ISL;        /* placeholder for references to user-provided initial space */
  PetscScalar *eigr, *eigi;     /* real and imaginary parts of eigenvalues */
  PetscReal   *errest;          /* error estimates */
  IP          ip;               /* innerproduct object */
  DS          ds;               /* direct solver object */
  void        *data;            /* placeholder for misc stuff associated 
                                   with a particular solver */
  PetscInt    nconv,            /* number of converged eigenvalues */
              its,              /* number of iterations so far computed */
              *perm,            /* permutation for eigenvalue ordering */
              matvecs, linits,  /* operation counters */
              n, nloc;          /* problem dimensions (global, local) */
  PetscRandom rand;             /* random number generator */
  Vec         t;                /* template vector */

  /* ---------------- Default work-area and status vars -------------------- */
  PetscInt   nwork;
  Vec        *work;

  PetscInt   setupcalled;
  QEPConvergedReason reason;     

  PetscErrorCode (*monitor[MAXQEPMONITORS])(QEP,PetscInt,PetscInt,PetscScalar*,PetscScalar*,PetscReal*,PetscInt,void*); 
  PetscErrorCode (*monitordestroy[MAXQEPMONITORS])(void**);
  void       *monitorcontext[MAXQEPMONITORS];
  PetscInt    numbermonitors; 
};

PETSC_EXTERN PetscErrorCode QEPMonitor(QEP,PetscInt,PetscInt,PetscScalar*,PetscScalar*,PetscReal*,PetscInt);

PETSC_EXTERN PetscErrorCode QEPDefaultGetWork(QEP,PetscInt);
PETSC_EXTERN PetscErrorCode QEPDefaultFreeWork(QEP);
PETSC_EXTERN PetscErrorCode QEPAllocateSolution(QEP);
PETSC_EXTERN PetscErrorCode QEPFreeSolution(QEP);
PETSC_EXTERN PetscErrorCode QEPComputeVectors_Schur(QEP);
PETSC_EXTERN PetscErrorCode QEPComputeResidualNorm_Private(QEP,PetscScalar,PetscScalar,Vec,Vec,PetscReal*);
PETSC_EXTERN PetscErrorCode QEPComputeRelativeError_Private(QEP,PetscScalar,PetscScalar,Vec,Vec,PetscReal*);
PETSC_EXTERN PetscErrorCode QEPKrylovConvergence(QEP,PetscBool,PetscInt,PetscInt,PetscInt,PetscReal,PetscInt*);

#endif
