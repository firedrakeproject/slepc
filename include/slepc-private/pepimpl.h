/*
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
   SLEPc - Scalable Library for Eigenvalue Problem Computations
   Copyright (c) 2002-2013, Universitat Politecnica de Valencia, Spain

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

#if !defined(_PEPIMPL)
#define _PEPIMPL

#include <slepcpep.h>
#include <slepc-private/slepcimpl.h>

PETSC_EXTERN PetscLogEvent PEP_SetUp, PEP_Solve, PEP_Dense;

typedef struct _PEPOps *PEPOps;

struct _PEPOps {
  PetscErrorCode  (*solve)(PEP);
  PetscErrorCode  (*setup)(PEP);
  PetscErrorCode  (*setfromoptions)(PEP);
  PetscErrorCode  (*publishoptions)(PEP);
  PetscErrorCode  (*destroy)(PEP);
  PetscErrorCode  (*reset)(PEP);
  PetscErrorCode  (*view)(PEP,PetscViewer);
};

/*
     Maximum number of monitors you can run with a single PEP
*/
#define MAXPEPMONITORS 5

/*
   Defines the PEP data structure.
*/
struct _p_PEP {
  PETSCHEADER(struct _PEPOps);
  /*------------------------- User parameters --------------------------*/
  PetscInt       max_it;           /* maximum number of iterations */
  PetscInt       nev;              /* number of eigenvalues to compute */
  PetscInt       ncv;              /* number of basis vectors */
  PetscInt       mpd;              /* maximum dimension of projected problem */
  PetscInt       nini,ninil;       /* number of initial vectors (negative means not copied yet) */
  PetscScalar    target;           /* target value */
  PetscReal      tol;              /* tolerance */
  PEPConv        conv;             /* convergence test */
  PetscReal      sfactor;          /* scaling factor */
  PetscBool      sfactor_set;      /* flag to indicate the user gave sfactor */
  PEPWhich       which;            /* which part of the spectrum to be sought */
  PetscBool      leftvecs;         /* if left eigenvectors are requested */
  PEPProblemType problem_type;     /* which kind of problem to be solved */
  PetscBool      trackall;         /* whether all the residuals must be computed */

  /*-------------- User-provided functions and contexts -----------------*/
  PetscErrorCode (*comparison)(PetscScalar,PetscScalar,PetscScalar,PetscScalar,PetscInt*,void*);
  PetscErrorCode (*converged)(PEP,PetscScalar,PetscScalar,PetscReal,PetscReal*,void*);
  void           *comparisonctx;
  void           *convergedctx;

  /*------------------------- Working data --------------------------*/
  Mat            *A;               /* coefficient matrices of the polynomial */
  PetscInt       nmat;             /* number of matrices */
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
  PEPConvergedReason reason;

  PetscErrorCode (*monitor[MAXPEPMONITORS])(PEP,PetscInt,PetscInt,PetscScalar*,PetscScalar*,PetscReal*,PetscInt,void*);
  PetscErrorCode (*monitordestroy[MAXPEPMONITORS])(void**);
  void           *monitorcontext[MAXPEPMONITORS];
  PetscInt        numbermonitors;
};

PETSC_INTERN PetscErrorCode PEPReset_Default(PEP);
PETSC_INTERN PetscErrorCode PEPAllocateSolution(PEP,PetscInt);
PETSC_INTERN PetscErrorCode PEPFreeSolution(PEP);
PETSC_INTERN PetscErrorCode PEPComputeVectors_Schur(PEP);
PETSC_INTERN PetscErrorCode PEPComputeVectors_Indefinite(PEP);
PETSC_INTERN PetscErrorCode PEPComputeResidualNorm_Private(PEP,PetscScalar,PetscScalar,Vec,Vec,PetscReal*);
PETSC_INTERN PetscErrorCode PEPComputeRelativeError_Private(PEP,PetscScalar,PetscScalar,Vec,Vec,PetscReal*);
PETSC_INTERN PetscErrorCode PEPKrylovConvergence(PEP,PetscBool,PetscInt,PetscInt,PetscInt,PetscReal,PetscInt*);

#endif
