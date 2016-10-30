/*
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
   SLEPc - Scalable Library for Eigenvalue Problem Computations
   Copyright (c) 2002-2016, Universitat Politecnica de Valencia, Spain

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

#if !defined(_LMEIMPL)
#define _LMEIMPL

#include <slepclme.h>
#include <slepc/private/slepcimpl.h>

PETSC_EXTERN PetscBool LMERegisterAllCalled;
PETSC_EXTERN PetscErrorCode LMERegisterAll(void);
PETSC_EXTERN PetscLogEvent LME_SetUp, LME_Solve;

typedef struct _LMEOps *LMEOps;

struct _LMEOps {
  PetscErrorCode (*solve)(LME);
  PetscErrorCode (*setup)(LME);
  PetscErrorCode (*setfromoptions)(PetscOptionItems*,LME);
  PetscErrorCode (*publishoptions)(LME);
  PetscErrorCode (*destroy)(LME);
  PetscErrorCode (*reset)(LME);
  PetscErrorCode (*view)(LME,PetscViewer);
};

/*
     Maximum number of monitors you can run with a single LME
*/
#define MAXLMEMONITORS 5

/*
   Defines the LME data structure.
*/
struct _p_LME {
  PETSCHEADER(struct _LMEOps);
  /*------------------------- User parameters ---------------------------*/
  Mat            A,B,D,E;        /* the coefficient matrices */
  PetscInt       max_it;         /* maximum number of iterations */
  PetscInt       ncv;            /* number of basis vectors */
  PetscReal      tol;            /* tolerance */
  PetscBool      errorifnotconverged;    /* error out if LMESolve() does not converge */

  /*-------------- User-provided functions and contexts -----------------*/
  PetscErrorCode (*monitor[MAXLMEMONITORS])(LME,PetscInt,PetscReal,void*);
  PetscErrorCode (*monitordestroy[MAXLMEMONITORS])(void**);
  void           *monitorcontext[MAXLMEMONITORS];
  PetscInt       numbermonitors;

  /*----------------- Child objects and working data -------------------*/
  BV             V;              /* set of basis vectors */
  PetscInt       nwork;          /* number of work vectors */
  Vec            *work;          /* work vectors */
  void           *data;          /* placeholder for solver-specific stuff */

  /* ----------------------- Status variables -------------------------- */
  PetscInt       its;            /* number of iterations so far computed */
  PetscReal      errest;         /* error estimate */
  PetscInt       setupcalled;
  LMEConvergedReason reason;
};

#endif
