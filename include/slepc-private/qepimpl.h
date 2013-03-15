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

#if !defined(_QEPIMPL)
#define _QEPIMPL

#include <slepcqep.h>
#include <slepc-private/slepcimpl.h>

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
  PetscInt       max_it;           /* maximum number of iterations */
  PetscInt       nev;              /* number of eigenvalues to compute */
  PetscInt       ncv;              /* number of basis vectors */
  PetscInt       mpd;              /* maximum dimension of projected problem */
  PetscInt       nini,ninil;       /* number of initial vectors (negative means not copied yet) */
  PetscScalar    target;           /* target value */
  PetscReal      tol;              /* tolerance */
  PetscReal      sfactor;          /* scaling factor of the quadratic problem */
  PetscBool      sfactor_set;      /* flag to indicate the user gave sfactor */
  QEPWhich       which;            /* which part of the spectrum to be sought */
  PetscBool      leftvecs;         /* if left eigenvectors are requested */
  QEPProblemType problem_type;     /* which kind of problem to be solved */
  PetscBool      trackall;         /* whether all the residuals must be computed */

  /*-------------- User-provided functions and contexts -----------------*/
  PetscErrorCode (*comparison)(PetscScalar,PetscScalar,PetscScalar,PetscScalar,PetscInt*,void*);
  PetscErrorCode (*converged)(QEP,PetscScalar,PetscScalar,PetscReal,PetscReal*,void*);
  void           *comparisonctx;
  void           *convergedctx;

  /*------------------------- Working data --------------------------*/
  Mat            M,C,K;            /* problem matrices */
  Vec            *V;               /* set of basis vectors and computed eigenvectors */
  Vec            *W;               /* set of left basis vectors and computed left eigenvectors */
  Vec            *IS,*ISL;         /* placeholder for references to user-provided initial space */
  PetscScalar    *eigr,*eigi;      /* real and imaginary parts of eigenvalues */
  PetscReal      *errest;          /* error estimates */
  IP             ip;               /* innerproduct object */
  DS             ds;               /* direct solver object */
  ST             st;               /* spectral transformation object */
  void           *data;            /* placeholder for misc stuff associated 
                                      with a particular solver */
  PetscInt       allocated_ncv;    /* number of basis vectors allocated */
  PetscInt       nconv;            /* number of converged eigenvalues */
  PetscInt       its;              /* number of iterations so far computed */
  PetscInt       *perm;            /* permutation for eigenvalue ordering */
  PetscInt       matvecs,linits;   /* operation counters */
  PetscInt       n,nloc;           /* problem dimensions (global, local) */
  PetscRandom    rand;             /* random number generator */
  Vec            t;                /* template vector */

  /* ---------------- Default work-area and status vars -------------------- */
  PetscInt       nwork;
  Vec            *work;

  PetscInt       setupcalled;
  QEPConvergedReason reason;     

  PetscErrorCode (*monitor[MAXQEPMONITORS])(QEP,PetscInt,PetscInt,PetscScalar*,PetscScalar*,PetscReal*,PetscInt,void*); 
  PetscErrorCode (*monitordestroy[MAXQEPMONITORS])(void**);
  void           *monitorcontext[MAXQEPMONITORS];
  PetscInt        numbermonitors; 
};

PETSC_INTERN PetscErrorCode QEPReset_Default(QEP);
PETSC_INTERN PetscErrorCode QEPAllocateSolution(QEP);
PETSC_INTERN PetscErrorCode QEPFreeSolution(QEP);
PETSC_INTERN PetscErrorCode QEPComputeVectors_Schur(QEP);
PETSC_INTERN PetscErrorCode QEPComputeResidualNorm_Private(QEP,PetscScalar,PetscScalar,Vec,Vec,PetscReal*);
PETSC_INTERN PetscErrorCode QEPComputeRelativeError_Private(QEP,PetscScalar,PetscScalar,Vec,Vec,PetscReal*);
PETSC_INTERN PetscErrorCode QEPKrylovConvergence(QEP,PetscBool,PetscInt,PetscInt,PetscInt,PetscReal,PetscInt*);

#endif
