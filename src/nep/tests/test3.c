/*
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
   SLEPc - Scalable Library for Eigenvalue Problem Computations
   Copyright (c) 2002-, Universitat Politecnica de Valencia, Spain

   This file is part of SLEPc.
   SLEPc is distributed under a 2-clause BSD license (see LICENSE).
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
*/

static char help[] = "Test the SLP solver with a user-provided EPS.\n\n"
  "This is a simplified version of ex20.\n"
  "The command line options are:\n"
  "  -n <n>, where <n> = number of grid subdivisions.\n";

/*
   Solve 1-D PDE
            -u'' = lambda*u
   on [0,1] subject to
            u(0)=0, u'(1)=u(1)*lambda*kappa/(kappa-lambda)
*/

#include <slepcnep.h>

/*
   User-defined routines
*/
PetscErrorCode FormFunction(NEP,PetscScalar,Mat,Mat,void*);
PetscErrorCode FormJacobian(NEP,PetscScalar,Mat,void*);

/*
   User-defined application context
*/
typedef struct {
  PetscScalar kappa;   /* ratio between stiffness of spring and attached mass */
  PetscReal   h;       /* mesh spacing */
} ApplicationCtx;

int main(int argc,char **argv)
{
  NEP            nep;
  EPS            eps;
  KSP            ksp;
  PC             pc;
  Mat            F,J;
  ApplicationCtx ctx;
  PetscInt       n=128;
  PetscReal      deftol;
  PetscBool      terse,flag,ts;

  PetscFunctionBeginUser;
  PetscCall(SlepcInitialize(&argc,&argv,(char*)0,help));
  PetscCall(PetscOptionsGetInt(NULL,NULL,"-n",&n,NULL));
  PetscCall(PetscPrintf(PETSC_COMM_WORLD,"\n1-D Nonlinear Eigenproblem, n=%" PetscInt_FMT "\n\n",n));
  ctx.h = 1.0/(PetscReal)n;
  ctx.kappa = 1.0;

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        Create a standalone EPS with appropriate settings
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

  PetscCall(EPSCreate(PETSC_COMM_WORLD,&eps));
  PetscCall(EPSSetType(eps,EPSGD));
  PetscCall(EPSSetFromOptions(eps));

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        Create a standalone KSP with appropriate settings
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

  PetscCall(KSPCreate(PETSC_COMM_WORLD,&ksp));
  PetscCall(KSPSetType(ksp,KSPBCGS));
  PetscCall(KSPGetPC(ksp,&pc));
  PetscCall(PCSetType(pc,PCBJACOBI));
  PetscCall(KSPSetFromOptions(ksp));

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
               Prepare nonlinear eigensolver context
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

  PetscCall(NEPCreate(PETSC_COMM_WORLD,&nep));

  /* Create Function and Jacobian matrices; set evaluation routines */
  PetscCall(MatCreate(PETSC_COMM_WORLD,&F));
  PetscCall(MatSetSizes(F,PETSC_DECIDE,PETSC_DECIDE,n,n));
  PetscCall(MatSetFromOptions(F));
  PetscCall(MatSeqAIJSetPreallocation(F,3,NULL));
  PetscCall(MatMPIAIJSetPreallocation(F,3,NULL,1,NULL));
  PetscCall(MatSetUp(F));
  PetscCall(NEPSetFunction(nep,F,F,FormFunction,&ctx));

  PetscCall(MatCreate(PETSC_COMM_WORLD,&J));
  PetscCall(MatSetSizes(J,PETSC_DECIDE,PETSC_DECIDE,n,n));
  PetscCall(MatSetFromOptions(J));
  PetscCall(MatSeqAIJSetPreallocation(J,3,NULL));
  PetscCall(MatMPIAIJSetPreallocation(F,3,NULL,1,NULL));
  PetscCall(MatSetUp(J));
  PetscCall(NEPSetJacobian(nep,J,FormJacobian,&ctx));

  /* Set options */
  PetscCall(NEPSetType(nep,NEPSLP));
  PetscCall(NEPSLPSetEPS(nep,eps));
  PetscCall(NEPSLPSetKSP(nep,ksp));
  PetscCall(NEPSetFromOptions(nep));

  /* Print some options */
  PetscCall(PetscObjectTypeCompare((PetscObject)nep,NEPSLP,&flag));
  if (flag) {
    PetscCall(NEPGetTwoSided(nep,&ts));
    if (ts) {
      PetscCall(NEPSLPGetDeflationThreshold(nep,&deftol));
      PetscCall(PetscPrintf(PETSC_COMM_WORLD," Two-sided solve with deflation threshold=%g\n",(double)deftol));
    }
  }

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
                      Solve the eigensystem
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

  PetscCall(NEPSolve(nep));

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
                    Display solution and clean up
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

  /* show detailed info unless -terse option is given by user */
  PetscCall(PetscOptionsHasName(NULL,NULL,"-terse",&terse));
  if (terse) PetscCall(NEPErrorView(nep,NEP_ERROR_RELATIVE,NULL));
  else {
    PetscCall(PetscViewerPushFormat(PETSC_VIEWER_STDOUT_WORLD,PETSC_VIEWER_ASCII_INFO_DETAIL));
    PetscCall(NEPConvergedReasonView(nep,PETSC_VIEWER_STDOUT_WORLD));
    PetscCall(NEPErrorView(nep,NEP_ERROR_RELATIVE,PETSC_VIEWER_STDOUT_WORLD));
    PetscCall(PetscViewerPopFormat(PETSC_VIEWER_STDOUT_WORLD));
  }

  PetscCall(NEPDestroy(&nep));
  PetscCall(EPSDestroy(&eps));
  PetscCall(KSPDestroy(&ksp));
  PetscCall(MatDestroy(&F));
  PetscCall(MatDestroy(&J));
  PetscCall(SlepcFinalize());
  return 0;
}

/* ------------------------------------------------------------------- */
/*
   FormFunction - Computes Function matrix  T(lambda)

   Input Parameters:
.  nep    - the NEP context
.  lambda - the scalar argument
.  ctx    - optional user-defined context, as set by NEPSetFunction()

   Output Parameters:
.  fun - Function matrix
.  B   - optionally different preconditioning matrix
*/
PetscErrorCode FormFunction(NEP nep,PetscScalar lambda,Mat fun,Mat B,void *ctx)
{
  ApplicationCtx *user = (ApplicationCtx*)ctx;
  PetscScalar    A[3],c,d;
  PetscReal      h;
  PetscInt       i,n,j[3],Istart,Iend;
  PetscBool      FirstBlock=PETSC_FALSE,LastBlock=PETSC_FALSE;

  PetscFunctionBeginUser;
  /*
     Compute Function entries and insert into matrix
  */
  PetscCall(MatGetSize(fun,&n,NULL));
  PetscCall(MatGetOwnershipRange(fun,&Istart,&Iend));
  if (Istart==0) FirstBlock=PETSC_TRUE;
  if (Iend==n) LastBlock=PETSC_TRUE;
  h = user->h;
  c = user->kappa/(lambda-user->kappa);
  d = n;

  /*
     Interior grid points
  */
  for (i=(FirstBlock? Istart+1: Istart);i<(LastBlock? Iend-1: Iend);i++) {
    j[0] = i-1; j[1] = i; j[2] = i+1;
    A[0] = A[2] = -d-lambda*h/6.0; A[1] = 2.0*(d-lambda*h/3.0);
    PetscCall(MatSetValues(fun,1,&i,3,j,A,INSERT_VALUES));
  }

  /*
     Boundary points
  */
  if (FirstBlock) {
    i = 0;
    j[0] = 0; j[1] = 1;
    A[0] = 2.0*(d-lambda*h/3.0); A[1] = -d-lambda*h/6.0;
    PetscCall(MatSetValues(fun,1,&i,2,j,A,INSERT_VALUES));
  }

  if (LastBlock) {
    i = n-1;
    j[0] = n-2; j[1] = n-1;
    A[0] = -d-lambda*h/6.0; A[1] = d-lambda*h/3.0+lambda*c;
    PetscCall(MatSetValues(fun,1,&i,2,j,A,INSERT_VALUES));
  }

  /*
     Assemble matrix
  */
  PetscCall(MatAssemblyBegin(B,MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(B,MAT_FINAL_ASSEMBLY));
  if (fun != B) {
    PetscCall(MatAssemblyBegin(fun,MAT_FINAL_ASSEMBLY));
    PetscCall(MatAssemblyEnd(fun,MAT_FINAL_ASSEMBLY));
  }
  PetscFunctionReturn(0);
}

/* ------------------------------------------------------------------- */
/*
   FormJacobian - Computes Jacobian matrix  T'(lambda)

   Input Parameters:
.  nep    - the NEP context
.  lambda - the scalar argument
.  ctx    - optional user-defined context, as set by NEPSetJacobian()

   Output Parameters:
.  jac - Jacobian matrix
.  B   - optionally different preconditioning matrix
*/
PetscErrorCode FormJacobian(NEP nep,PetscScalar lambda,Mat jac,void *ctx)
{
  ApplicationCtx *user = (ApplicationCtx*)ctx;
  PetscScalar    A[3],c;
  PetscReal      h;
  PetscInt       i,n,j[3],Istart,Iend;
  PetscBool      FirstBlock=PETSC_FALSE,LastBlock=PETSC_FALSE;

  PetscFunctionBeginUser;
  /*
     Compute Jacobian entries and insert into matrix
  */
  PetscCall(MatGetSize(jac,&n,NULL));
  PetscCall(MatGetOwnershipRange(jac,&Istart,&Iend));
  if (Istart==0) FirstBlock=PETSC_TRUE;
  if (Iend==n) LastBlock=PETSC_TRUE;
  h = user->h;
  c = user->kappa/(lambda-user->kappa);

  /*
     Interior grid points
  */
  for (i=(FirstBlock? Istart+1: Istart);i<(LastBlock? Iend-1: Iend);i++) {
    j[0] = i-1; j[1] = i; j[2] = i+1;
    A[0] = A[2] = -h/6.0; A[1] = -2.0*h/3.0;
    PetscCall(MatSetValues(jac,1,&i,3,j,A,INSERT_VALUES));
  }

  /*
     Boundary points
  */
  if (FirstBlock) {
    i = 0;
    j[0] = 0; j[1] = 1;
    A[0] = -2.0*h/3.0; A[1] = -h/6.0;
    PetscCall(MatSetValues(jac,1,&i,2,j,A,INSERT_VALUES));
  }

  if (LastBlock) {
    i = n-1;
    j[0] = n-2; j[1] = n-1;
    A[0] = -h/6.0; A[1] = -h/3.0-c*c;
    PetscCall(MatSetValues(jac,1,&i,2,j,A,INSERT_VALUES));
  }

  /*
     Assemble matrix
  */
  PetscCall(MatAssemblyBegin(jac,MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(jac,MAT_FINAL_ASSEMBLY));
  PetscFunctionReturn(0);
}

/*TEST

   test:
      args: -nep_target 21 -terse
      requires: !single
      test:
         suffix: 1
      test:
         suffix: 1_ts
         args: -nep_two_sided

TEST*/
