/*
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
   SLEPc - Scalable Library for Eigenvalue Problem Computations
   Copyright (c) 2002-2017, Universitat Politecnica de Valencia, Spain

   This file is part of SLEPc.
   SLEPc is distributed under a 2-clause BSD license (see LICENSE).
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
*/
/*
   Private header for Deflation in NEP
*/
#if !defined(__NEPDEFL_H)
#define __NEPDEFL_H

# define MAX_MINIDX 1

typedef struct _n_nep_ext_op *NEP_EXT_OP;
typedef struct _n_nep_def_fun_solve *NEP_DEF_FUN_SOLVE;
/* Structure characterizing a deflation context */
struct _n_nep_ext_op {
  NEP          nep;
  PetscScalar *H;     /* invariant pair (X,H) */
  BV           X;     /* locked eigenvectors */
  PetscScalar  *bc;   /* polinomial basis roots */
  RG           rg;
  PetscInt     midx;  /* minimality index */
  PetscInt     max_midx;
  PetscInt     szd;   /* maxim size for deflation */
  PetscInt     n;     /* invariant pair size */
  Mat          MF;    /* function shell matrix */
  Mat          MJ;    /* Jacobian shell matrix */
  PetscBool    simpU; /* the way U is computed */
  NEP_DEF_FUN_SOLVE solve;
  /* auxiliary computations */
  BV           W;
  PetscScalar *Hj;    /* matrix containing the powers of the invariant pair matrix */
  PetscScalar *XpX;   /* X^*X */
};

struct _n_nep_def_fun_solve {
  KSP          ksp;   /* */
  PetscScalar  theta;
  PetscInt     n;
  PetscScalar  *M;
  PetscScalar  *work;
  Vec          w[2];
  BV           T_1U;
  NEP_EXT_OP   extop;
};

typedef struct {
  NEP          nep;
  Mat          T;
  BV           U;
  PetscScalar  *A;
  PetscScalar  *B;  
  PetscScalar  theta;
  PetscInt     n;
  NEP_EXT_OP   extop;
  PetscBool    jacob;
  Vec          w[2];
  PetscScalar  *work;
  PetscScalar  *hfj;
  PetscScalar  *hfjp;
  PetscBool    hfjset;
} NEP_DEF_MATSHELL;

#if 0
typedef struct {
  PC          pc;      /* basic preconditioner */
  PetscScalar *M;
  PetscScalar *ps;
  PetscInt    ld;
  Vec         *work;
  BV          X;
  PetscInt    n;
} NEP_DEF_PCSHELL;
#endif
#endif

PETSC_INTERN PetscErrorCode NEPDeflationCopyToExtendedVec(NEP_EXT_OP,Vec,PetscScalar*,Vec,PetscBool);
PETSC_INTERN PetscErrorCode NEPDeflationReset(NEP_EXT_OP);
PETSC_INTERN PetscErrorCode NEPDeflationInitialize(NEP,BV,KSP,PetscInt,NEP_EXT_OP*);
PETSC_INTERN PetscErrorCode NEPDeflationCreateVec(NEP_EXT_OP,Vec*);
PETSC_INTERN PetscErrorCode NEPDeflationComputeFunction(NEP_EXT_OP,PetscScalar,Mat*);
PETSC_INTERN PetscErrorCode NEPDeflationComputeJacobian(NEP_EXT_OP,PetscScalar,Mat*);
PETSC_INTERN PetscErrorCode NEPDeflationSolveSetUp(NEP_EXT_OP,PetscScalar);
PETSC_INTERN PetscErrorCode NEPDeflationFunctionSolve(NEP_EXT_OP,Vec,Vec);
PETSC_INTERN PetscErrorCode NEPDeflationGetInvariantPair(NEP_EXT_OP,BV*,Mat*);
PETSC_INTERN PetscErrorCode NEPDeflationLocking(NEP_EXT_OP,Vec,PetscScalar);
