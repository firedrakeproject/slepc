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

#if !defined(_NEPIMPL)
#define _NEPIMPL

#include <slepcnep.h>
#include <slepc/private/slepcimpl.h>

PETSC_EXTERN PetscBool NEPRegisterAllCalled;
PETSC_EXTERN PetscErrorCode NEPRegisterAll(void);
PETSC_EXTERN PetscLogEvent NEP_SetUp,NEP_Solve,NEP_Refine,NEP_FunctionEval,NEP_JacobianEval,NEP_DerivativesEval;

typedef struct _NEPOps *NEPOps;

struct _NEPOps {
  PetscErrorCode (*solve)(NEP);
  PetscErrorCode (*setup)(NEP);
  PetscErrorCode (*setfromoptions)(PetscOptionItems*,NEP);
  PetscErrorCode (*publishoptions)(NEP);
  PetscErrorCode (*destroy)(NEP);
  PetscErrorCode (*reset)(NEP);
  PetscErrorCode (*view)(NEP,PetscViewer);
  PetscErrorCode (*computevectors)(NEP);
};

/*
     Maximum number of monitors you can run with a single NEP
*/
#define MAXNEPMONITORS 5

typedef enum { NEP_STATE_INITIAL,
               NEP_STATE_SETUP,
               NEP_STATE_SOLVED,
               NEP_STATE_EIGENVECTORS } NEPStateType;

/*
     How the problem function T(lambda) has been defined by the user
     - Callback: one callback to build the function matrix, another one for the Jacobian
     - Split: in split form sum_j(A_j*f_j(lambda))
     - Derivatives: a single callback for all the derivatives (including the 0th derivative)
*/
typedef enum { NEP_USER_INTERFACE_CALLBACK=1,
               NEP_USER_INTERFACE_SPLIT,
               NEP_USER_INTERFACE_DERIVATIVES } NEPUserInterface;

/*
   Defines the NEP data structure.
*/
struct _p_NEP {
  PETSCHEADER(struct _NEPOps);
  /*------------------------- User parameters ---------------------------*/
  PetscInt       max_it;           /* maximum number of iterations */
  PetscInt       nev;              /* number of eigenvalues to compute */
  PetscInt       ncv;              /* number of basis vectors */
  PetscInt       mpd;              /* maximum dimension of projected problem */
  PetscInt       nini;             /* number of initial vectors (negative means not copied yet) */
  PetscScalar    target;           /* target value */
  PetscReal      tol;              /* tolerance */
  NEPConv        conv;             /* convergence test */
  NEPStop        stop;             /* stopping test */
  NEPWhich       which;            /* which part of the spectrum to be sought */
  NEPRefine      refine;           /* type of refinement to be applied after solve */
  PetscInt       npart;            /* number of partitions of the communicator */
  PetscReal      rtol;             /* tolerance for refinement */
  PetscInt       rits;             /* number of iterations of the refinement method */
  NEPRefineScheme scheme;          /* scheme for solving linear systems within refinement */
  PetscBool      trackall;         /* whether all the residuals must be computed */

  /*-------------- User-provided functions and contexts -----------------*/
  PetscErrorCode (*computefunction)(NEP,PetscScalar,Mat,Mat,void*);
  PetscErrorCode (*computejacobian)(NEP,PetscScalar,Mat,void*);
  void           *functionctx;
  void           *jacobianctx;
  PetscErrorCode (*computederivatives)(NEP,PetscScalar,PetscInt,Mat,void*);
  void           *derivativesctx;
  PetscErrorCode (*converged)(NEP,PetscScalar,PetscScalar,PetscReal,PetscReal*,void*);
  PetscErrorCode (*convergeddestroy)(void*);
  PetscErrorCode (*stopping)(NEP,PetscInt,PetscInt,PetscInt,PetscInt,NEPConvergedReason*,void*);
  PetscErrorCode (*stoppingdestroy)(void*);
  void           *convergedctx;
  void           *stoppingctx;
  PetscErrorCode (*monitor[MAXNEPMONITORS])(NEP,PetscInt,PetscInt,PetscScalar*,PetscScalar*,PetscReal*,PetscInt,void*);
  PetscErrorCode (*monitordestroy[MAXNEPMONITORS])(void**);
  void           *monitorcontext[MAXNEPMONITORS];
  PetscInt       numbermonitors;

  /*----------------- Child objects and working data -------------------*/
  DS             ds;               /* direct solver object */
  BV             V;                /* set of basis vectors and computed eigenvectors */
  RG             rg;               /* optional region for filtering */
  SlepcSC        sc;               /* sorting criterion data */
  Mat            function;         /* function matrix */
  Mat            function_pre;     /* function matrix (preconditioner) */
  Mat            jacobian;         /* Jacobian matrix */
  Mat            derivatives;      /* derivatives matrix */
  Mat            *A;               /* matrix coefficients of split form */
  FN             *f;               /* matrix functions of split form */
  PetscInt       nt;               /* number of terms in split form */
  MatStructure   mstr;             /* pattern of split matrices */
  Vec            *IS;              /* references to user-provided initial space */
  PetscScalar    *eigr,*eigi;      /* real and imaginary parts of eigenvalues */
  PetscReal      *errest;          /* error estimates */
  PetscInt       *perm;            /* permutation for eigenvalue ordering */
  PetscInt       nwork;            /* number of work vectors */
  Vec            *work;            /* work vectors */
  KSP            refineksp;        /* ksp used in refinement */
  PetscSubcomm   refinesubc;       /* context for sub-communicators */
  void           *data;            /* placeholder for solver-specific stuff */

  /* ----------------------- Status variables --------------------------*/
  NEPStateType   state;            /* initial -> setup -> solved -> eigenvectors */
  PetscInt       nconv;            /* number of converged eigenvalues */
  PetscInt       its;              /* number of iterations so far computed */
  PetscInt       n,nloc;           /* problem dimensions (global, local) */
  PetscReal      *nrma;            /* computed matrix norms */
  NEPUserInterface fui;            /* how the user has defined the nonlinear operator */
  NEPConvergedReason reason;
};

/*
    Macros to test valid NEP arguments
*/
#if !defined(PETSC_USE_DEBUG)

#define NEPCheckProblem(h,arg) do {} while (0)
#define NEPCheckCallback(h,arg) do {} while (0)
#define NEPCheckSplit(h,arg) do {} while (0)
#define NEPCheckDerivatives(h,arg) do {} while (0)
#define NEPCheckSolved(h,arg) do {} while (0)

#else

#define NEPCheckProblem(h,arg) \
  do { \
    if (!(h->fui)) SETERRQ1(PetscObjectComm((PetscObject)h),PETSC_ERR_ARG_WRONGSTATE,"The nonlinear eigenproblem has not been specified yet. Parameter #%d",arg); \
  } while (0)

#define NEPCheckCallback(h,arg) \
  do { \
    if (h->fui!=NEP_USER_INTERFACE_CALLBACK) SETERRQ1(PetscObjectComm((PetscObject)h),PETSC_ERR_ARG_WRONGSTATE,"This operation requires the nonlinear eigenproblem specified with callbacks. Parameter #%d",arg); \
  } while (0)

#define NEPCheckSplit(h,arg) \
  do { \
    if (h->fui!=NEP_USER_INTERFACE_SPLIT) SETERRQ1(PetscObjectComm((PetscObject)h),PETSC_ERR_ARG_WRONGSTATE,"This operation requires the nonlinear eigenproblem in split form. Parameter #%d",arg); \
  } while (0)

#define NEPCheckDerivatives(h,arg) \
  do { \
    if (h->fui!=NEP_USER_INTERFACE_DERIVATIVES) SETERRQ1(PetscObjectComm((PetscObject)h),PETSC_ERR_ARG_WRONGSTATE,"This operation requires the nonlinear eigenproblem specified with derivatives callback. Parameter #%d",arg); \
  } while (0)

#define NEPCheckSolved(h,arg) \
  do { \
    if (h->state<NEP_STATE_SOLVED) SETERRQ1(PetscObjectComm((PetscObject)h),PETSC_ERR_ARG_WRONGSTATE,"Must call NEPSolve() first: Parameter #%d",arg); \
  } while (0)

#endif

PETSC_INTERN PetscErrorCode NEPSetDimensions_Default(NEP,PetscInt,PetscInt*,PetscInt*);
PETSC_INTERN PetscErrorCode NEPComputeVectors(NEP);
PETSC_INTERN PetscErrorCode NEPReset_Problem(NEP);
PETSC_INTERN PetscErrorCode NEPGetDefaultShift(NEP,PetscScalar*);
PETSC_INTERN PetscErrorCode NEPComputeVectors_Schur(NEP);
PETSC_INTERN PetscErrorCode NEPComputeResidualNorm_Private(NEP,PetscScalar,Vec,Vec*,PetscReal*);
PETSC_INTERN PetscErrorCode NEPNewtonRefinementSimple(NEP,PetscInt*,PetscReal,PetscInt);

#endif
