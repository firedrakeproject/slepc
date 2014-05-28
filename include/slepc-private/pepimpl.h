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

PETSC_EXTERN PetscLogEvent PEP_SetUp,PEP_Solve;

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
  /*------------------------- User parameters ---------------------------*/
  PetscInt       max_it;           /* maximum number of iterations */
  PetscInt       nev;              /* number of eigenvalues to compute */
  PetscInt       ncv;              /* number of basis vectors */
  PetscInt       mpd;              /* maximum dimension of projected problem */
  PetscInt       nini,ninil;       /* number of initial vectors (negative means not copied yet) */
  PetscScalar    target;           /* target value */
  PetscReal      tol;              /* tolerance */
  PEPConv        conv;             /* convergence test */
  PetscReal      sfactor;          /* scaling factor */
  PEPWhich       which;            /* which part of the spectrum to be sought */
  PEPBasis       basis;            /* polynomial basis used to represent the problem */
  PetscBool      leftvecs;         /* if left eigenvectors are requested */
  PEPProblemType problem_type;     /* which kind of problem to be solved */
  PetscBool      balance;          /* whether balancing must be performed*/
  PetscInt       balance_its;      /* number of iterations of the balancing method */
  PetscReal      balance_lambda;   /* norm eigenvalue approximation for balancing*/
  PetscBool      trackall;         /* whether all the residuals must be computed */

  /*-------------- User-provided functions and contexts -----------------*/
  PetscErrorCode (*comparison)(PetscScalar,PetscScalar,PetscScalar,PetscScalar,PetscInt*,void*);
  PetscErrorCode (*converged)(PEP,PetscScalar,PetscScalar,PetscReal,PetscReal*,void*);
  void           *comparisonctx;
  void           *convergedctx;
  PetscErrorCode (*monitor[MAXPEPMONITORS])(PEP,PetscInt,PetscInt,PetscScalar*,PetscScalar*,PetscReal*,PetscInt,void*);
  PetscErrorCode (*monitordestroy[MAXPEPMONITORS])(void**);
  void           *monitorcontext[MAXPEPMONITORS];
  PetscInt        numbermonitors;

  /*----------------- Child objects and working data -------------------*/
  ST             st;               /* spectral transformation object */
  DS             ds;               /* direct solver object */
  BV             V;                /* set of basis vectors and computed eigenvectors */
  PetscRandom    rand;             /* random number generator */
  Mat            *A;               /* coefficient matrices of the polynomial */
  PetscInt       nmat;             /* number of matrices */
  Vec            Dl,Dr;            /* diagonal matrices for balancing */
  Vec            *IS,*ISL;         /* references to user-provided initial space */
  PetscScalar    *eigr,*eigi;      /* real and imaginary parts of eigenvalues */
  PetscReal      *errest;          /* error estimates */
  PetscInt       *perm;            /* permutation for eigenvalue ordering */
  PetscReal      *pbc;             /* coefficients defining the polynomial basis */
  PetscScalar    *solvematcoeffs;  /* coefficients to compute the matrix to be inverted */
  PetscInt       nwork;            /* number of work vectors */
  Vec            *work;            /* work vectors */
  void           *data;            /* placeholder for solver-specific stuff */

  /* ----------------------- Status variables --------------------------*/
  PetscInt       nconv;            /* number of converged eigenvalues */
  PetscInt       its;              /* number of iterations so far computed */
  PetscInt       n,nloc;           /* problem dimensions (global, local) */
  PetscReal      *nrma;            /* computed matrix norms */
  PetscBool      sfactor_set;      /* flag to indicate the user gave sfactor */
  PetscInt       setupcalled;
  PEPConvergedReason reason;
};

PETSC_INTERN PetscErrorCode PEPReset_Default(PEP);
PETSC_INTERN PetscErrorCode PEPAllocateSolution(PEP,PetscInt);
PETSC_INTERN PetscErrorCode PEPComputeVectors_Schur(PEP);
PETSC_INTERN PetscErrorCode PEPComputeVectors_Indefinite(PEP);
PETSC_INTERN PetscErrorCode PEPComputeResidualNorm_Private(PEP,PetscScalar,PetscScalar,Vec,Vec,PetscReal*);
PETSC_INTERN PetscErrorCode PEPComputeRelativeError_Private(PEP,PetscScalar,PetscScalar,Vec,Vec,PetscReal*);
PETSC_INTERN PetscErrorCode PEPKrylovConvergence(PEP,PetscBool,PetscInt,PetscInt,PetscInt,PetscReal,PetscInt*);
PETSC_INTERN PetscErrorCode PEPComputeScaleFactor(PEP);
PETSC_INTERN PetscErrorCode PEPBuildBalance(PEP);
PETSC_INTERN PetscErrorCode PEPBasisCoefficients(PEP,PetscReal*);
PETSC_INTERN PetscErrorCode PEPEvaluateBasis(PEP,PetscScalar,PetscScalar,PetscScalar*,PetscScalar*);
PETSC_INTERN PetscErrorCode PEPNewtonRefinement_TOAR(PEP,PetscInt*,PetscReal*,PetscInt,PetscScalar*,PetscInt);

#endif
