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

#if !defined(_NEPIMPL)
#define _NEPIMPL

#include <slepcnep.h>

PETSC_EXTERN PetscFunctionList NEPList;
PETSC_EXTERN PetscLogEvent     NEP_SetUp,NEP_Solve,NEP_Dense;

typedef struct _NEPOps *NEPOps;

struct _NEPOps {
  PetscErrorCode  (*solve)(NEP);
  PetscErrorCode  (*setup)(NEP);
  PetscErrorCode  (*setfromoptions)(NEP);
  PetscErrorCode  (*publishoptions)(NEP);
  PetscErrorCode  (*destroy)(NEP);
  PetscErrorCode  (*reset)(NEP);
  PetscErrorCode  (*view)(NEP,PetscViewer);
};

/*
     Maximum number of monitors you can run with a single NEP
*/
#define MAXNEPMONITORS 5 

/*
   Defines the NEP data structure.
*/
struct _p_NEP {
  PETSCHEADER(struct _NEPOps);
  /*------------------------- User parameters --------------------------*/
  PetscInt       max_it,           /* maximum number of iterations */
                 nev,              /* number of eigenvalues to compute */
                 ncv,              /* number of basis vectors */
                 mpd,              /* maximum dimension of projected problem */
                 nini,             /* number of initial vectors (negative means not copied yet) */
                 allocated_ncv;    /* number of basis vectors allocated */
  PetscScalar    target;           /* target value */
  PetscReal      atol,rtol,stol;   /* tolerances */
  PetscErrorCode (*conv_func)(NEP,PetscInt,PetscReal,PetscReal,PetscReal,NEPConvergedReason*,void*);
  void           *conv_ctx;
  NEPWhich       which;            /* which part of the spectrum to be sought */
  PetscErrorCode (*which_func)(PetscScalar,PetscScalar,PetscScalar,PetscScalar,PetscInt*,void*);
  void           *which_ctx;
  PetscBool      trackall;         /* whether all the residuals must be computed */

  /*------------------------- Working data --------------------------*/
  Vec         *V,               /* set of basis vectors and computed eigenvectors */
              *IS;              /* placeholder for references to user-provided initial space */
  PetscScalar *eigr, *eigi;     /* real and imaginary parts of eigenvalues */
  PetscReal   *errest;          /* error estimates */
  IP          ip;               /* innerproduct object */
  DS          ds;               /* direct solver object */
  KSP         ksp;              /* linear solver object */
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
  NEPConvergedReason reason;     

  PetscErrorCode (*monitor[MAXNEPMONITORS])(NEP,PetscInt,PetscInt,PetscScalar*,PetscScalar*,PetscReal*,PetscInt,void*); 
  PetscErrorCode (*monitordestroy[MAXNEPMONITORS])(void**);
  void       *monitorcontext[MAXNEPMONITORS];
  PetscInt    numbermonitors; 
};

PETSC_EXTERN PetscErrorCode NEPMonitor(NEP,PetscInt,PetscInt,PetscScalar*,PetscScalar*,PetscReal*,PetscInt);

PETSC_EXTERN PetscErrorCode NEPDefaultGetWork(NEP,PetscInt);
PETSC_EXTERN PetscErrorCode NEPDefaultFreeWork(NEP);
PETSC_EXTERN PetscErrorCode NEPAllocateSolution(NEP);
PETSC_EXTERN PetscErrorCode NEPFreeSolution(NEP);

#endif
