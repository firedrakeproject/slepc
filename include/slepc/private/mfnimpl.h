/*
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
   SLEPc - Scalable Library for Eigenvalue Problem Computations
   Copyright (c) 2002-2015, Universitat Politecnica de Valencia, Spain

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

#if !defined(_MFNIMPL)
#define _MFNIMPL

#include <slepcmfn.h>
#include <slepc/private/slepcimpl.h>

PETSC_EXTERN PetscBool MFNRegisterAllCalled;
PETSC_EXTERN PetscErrorCode MFNRegisterAll(void);
PETSC_EXTERN PetscLogEvent MFN_SetUp, MFN_Solve;

typedef struct _MFNOps *MFNOps;

struct _MFNOps {
  PetscErrorCode (*solve)(MFN,Vec,Vec);
  PetscErrorCode (*setup)(MFN);
  PetscErrorCode (*setfromoptions)(PetscOptions*,MFN);
  PetscErrorCode (*publishoptions)(MFN);
  PetscErrorCode (*destroy)(MFN);
  PetscErrorCode (*reset)(MFN);
  PetscErrorCode (*view)(MFN,PetscViewer);
};

/*
     Maximum number of monitors you can run with a single MFN
*/
#define MAXMFNMONITORS 5

/*
   Defines the MFN data structure.
*/
struct _p_MFN {
  PETSCHEADER(struct _MFNOps);
  /*------------------------- User parameters ---------------------------*/
  Mat            A;              /* the problem matrix */
  FN             fn;             /* which function to compute */
  PetscInt       max_it;         /* maximum number of iterations */
  PetscInt       ncv;            /* number of basis vectors */
  PetscReal      tol;            /* tolerance */
  PetscScalar    sfactor;        /* scaling factor */
  PetscBool      errorifnotconverged;    /* error out if MFNSolve() does not converge */

  /*-------------- User-provided functions and contexts -----------------*/
  PetscErrorCode (*monitor[MAXMFNMONITORS])(MFN,PetscInt,PetscReal,void*);
  PetscErrorCode (*monitordestroy[MAXMFNMONITORS])(void**);
  void           *monitorcontext[MAXMFNMONITORS];
  PetscInt       numbermonitors;

  /*----------------- Child objects and working data -------------------*/
  BV             V;              /* set of basis vectors */
  PetscRandom    rand;           /* random number generator */
  PetscInt       nwork;          /* number of work vectors */
  Vec            *work;          /* work vectors */
  void           *data;          /* placeholder for solver-specific stuff */

  /* ----------------------- Status variables -------------------------- */
  PetscInt       its;            /* number of iterations so far computed */
  PetscInt       nv;             /* size of current Schur decomposition */
  PetscReal      errest;         /* error estimate */
  PetscInt       setupcalled;
  MFNConvergedReason reason;
};

#endif
