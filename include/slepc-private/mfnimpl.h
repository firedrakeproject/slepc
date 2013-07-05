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

#if !defined(_MFNIMPL)
#define _MFNIMPL

#include <slepcmfn.h>
#include <slepc-private/slepcimpl.h>

PETSC_EXTERN PetscLogEvent MFN_SetUp, MFN_Solve;

typedef struct _MFNOps *MFNOps;

struct _MFNOps {
  PetscErrorCode (*solve)(MFN,Vec,Vec);
  PetscErrorCode (*setup)(MFN);
  PetscErrorCode (*setfromoptions)(MFN);
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
  /*------------------------- User parameters --------------------------*/
  PetscInt        max_it;         /* maximum number of iterations */
  PetscInt        ncv;            /* number of basis vectors */
  PetscReal       tol;            /* tolerance */
  SlepcFunction   function;       /* which function to compute */
  PetscScalar     sfactor;        /* scaling factor */
  PetscBool       errorifnotconverged;    /* error out if MFNSolve() does not converge */

  /*------------------------- Working data --------------------------*/
  Mat             A;              /* the problem matrix */
  Vec             *V;             /* set of basis vectors */
  PetscReal       errest;         /* error estimate */
  IP              ip;             /* innerproduct object */
  DS              ds;             /* direct solver object */
  void            *data;          /* placeholder for misc stuff associated
                                     with a particular solver */
  PetscInt        its;            /* number of iterations so far computed */
  PetscInt        nv;             /* size of current Schur decomposition */
  PetscInt        n,nloc;         /* problem dimensions (global, local) */
  PetscInt        allocated_ncv;  /* number of basis vectors allocated */
  PetscRandom     rand;           /* random number generator */
  Vec             t;              /* template vector */

  /* ---------------- Default work-area and status vars -------------------- */
  PetscInt       nwork;
  Vec            *work;

  PetscInt       setupcalled;
  MFNConvergedReason reason;

  PetscErrorCode (*monitor[MAXMFNMONITORS])(MFN,PetscInt,PetscReal,void*);
  PetscErrorCode (*monitordestroy[MAXMFNMONITORS])(void**);
  void           *monitorcontext[MAXMFNMONITORS];
  PetscInt       numbermonitors;
};

PETSC_INTERN PetscErrorCode MFNReset_Default(MFN);
PETSC_INTERN PetscErrorCode MFNDefaultGetWork(MFN,PetscInt);
PETSC_INTERN PetscErrorCode MFNDefaultFreeWork(MFN);
PETSC_INTERN PetscErrorCode MFNAllocateSolution(MFN);
PETSC_INTERN PetscErrorCode MFNFreeSolution(MFN);

#endif
