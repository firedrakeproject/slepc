/*
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
   SLEPc - Scalable Library for Eigenvalue Problem Computations
   Copyright (c) 2002-, Universitat Politecnica de Valencia, Spain

   This file is part of SLEPc.
   SLEPc is distributed under a 2-clause BSD license (see LICENSE).
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
*/
/*
   Example based on spring problem in NLEVP collection [1]. See the parameters
   meaning at Example 2 in [2].

   [1] T. Betcke, N. J. Higham, V. Mehrmann, C. Schroder, and F. Tisseur,
       NLEVP: A Collection of Nonlinear Eigenvalue Problems, MIMS EPrint
       2010.98, November 2010.
   [2] F. Tisseur, Backward error and condition of polynomial eigenvalue
       problems, Linear Algebra and its Applications, 309 (2000), pp. 339--361,
       April 2000.
*/

static char help[] = "Illustrates the use of a user-defined stopping test.\n\n"
  "The command line options are:\n"
  "  -n <n> ... number of grid subdivisions.\n"
  "  -mu <value> ... mass (default 1).\n"
  "  -tau <value> ... damping constant of the dampers (default 10).\n"
  "  -kappa <value> ... damping constant of the springs (default 5).\n\n";

#include <slepcpep.h>

/*
   User-defined routines
*/
PetscErrorCode MyStoppingTest(PEP,PetscInt,PetscInt,PetscInt,PetscInt,PEPConvergedReason*,void*);

typedef struct {
  PetscInt    lastnconv;      /* last value of nconv; used in stopping test */
  PetscInt    nreps;          /* number of repetitions of nconv; used in stopping test */
} CTX_SPRING;

int main(int argc,char **argv)
{
  Mat            M,C,K,A[3];      /* problem matrices */
  PEP            pep;             /* polynomial eigenproblem solver context */
  RG             rg;              /* region object */
  ST             st;
  CTX_SPRING     *ctx;
  PetscBool      terse;
  PetscViewer    viewer;
  PetscInt       n=30,Istart,Iend,i,mpd;
  PetscReal      mu=1.0,tau=10.0,kappa=5.0;

  PetscFunctionBeginUser;
  PetscCall(SlepcInitialize(&argc,&argv,(char*)0,help));

  PetscCall(PetscOptionsGetInt(NULL,NULL,"-n",&n,NULL));
  PetscCall(PetscOptionsGetReal(NULL,NULL,"-mu",&mu,NULL));
  PetscCall(PetscOptionsGetReal(NULL,NULL,"-tau",&tau,NULL));
  PetscCall(PetscOptionsGetReal(NULL,NULL,"-kappa",&kappa,NULL));
  PetscCall(PetscPrintf(PETSC_COMM_WORLD,"\nDamped mass-spring system, n=%" PetscInt_FMT " mu=%g tau=%g kappa=%g\n\n",n,(double)mu,(double)tau,(double)kappa));

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Compute the matrices that define the eigensystem, (k^2*M+k*C+K)x=0
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

  /* K is a tridiagonal */
  PetscCall(MatCreate(PETSC_COMM_WORLD,&K));
  PetscCall(MatSetSizes(K,PETSC_DECIDE,PETSC_DECIDE,n,n));
  PetscCall(MatSetFromOptions(K));
  PetscCall(MatSetUp(K));
  PetscCall(MatGetOwnershipRange(K,&Istart,&Iend));
  for (i=Istart;i<Iend;i++) {
    if (i>0) PetscCall(MatSetValue(K,i,i-1,-kappa,INSERT_VALUES));
    PetscCall(MatSetValue(K,i,i,kappa*3.0,INSERT_VALUES));
    if (i<n-1) PetscCall(MatSetValue(K,i,i+1,-kappa,INSERT_VALUES));
  }
  PetscCall(MatAssemblyBegin(K,MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(K,MAT_FINAL_ASSEMBLY));

  /* C is a tridiagonal */
  PetscCall(MatCreate(PETSC_COMM_WORLD,&C));
  PetscCall(MatSetSizes(C,PETSC_DECIDE,PETSC_DECIDE,n,n));
  PetscCall(MatSetFromOptions(C));
  PetscCall(MatSetUp(C));
  PetscCall(MatGetOwnershipRange(C,&Istart,&Iend));
  for (i=Istart;i<Iend;i++) {
    if (i>0) PetscCall(MatSetValue(C,i,i-1,-tau,INSERT_VALUES));
    PetscCall(MatSetValue(C,i,i,tau*3.0,INSERT_VALUES));
    if (i<n-1) PetscCall(MatSetValue(C,i,i+1,-tau,INSERT_VALUES));
  }
  PetscCall(MatAssemblyBegin(C,MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(C,MAT_FINAL_ASSEMBLY));

  /* M is a diagonal matrix */
  PetscCall(MatCreate(PETSC_COMM_WORLD,&M));
  PetscCall(MatSetSizes(M,PETSC_DECIDE,PETSC_DECIDE,n,n));
  PetscCall(MatSetFromOptions(M));
  PetscCall(MatSetUp(M));
  PetscCall(MatGetOwnershipRange(M,&Istart,&Iend));
  for (i=Istart;i<Iend;i++) PetscCall(MatSetValue(M,i,i,mu,INSERT_VALUES));
  PetscCall(MatAssemblyBegin(M,MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(M,MAT_FINAL_ASSEMBLY));

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
                Create the eigensolver and set various options
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

  PetscCall(PEPCreate(PETSC_COMM_WORLD,&pep));
  A[0] = K; A[1] = C; A[2] = M;
  PetscCall(PEPSetOperators(pep,3,A));
  PetscCall(PEPSetProblemType(pep,PEP_GENERAL));
  PetscCall(PEPSetTolerances(pep,PETSC_SMALL,PETSC_DEFAULT));

  /*
     Define the region containing the eigenvalues of interest
  */
  PetscCall(PEPGetRG(pep,&rg));
  PetscCall(RGSetType(rg,RGINTERVAL));
  PetscCall(RGIntervalSetEndpoints(rg,-0.5057,-0.5052,-0.001,0.001));
  PetscCall(PEPSetTarget(pep,-0.43));
  PetscCall(PEPSetWhichEigenpairs(pep,PEP_TARGET_MAGNITUDE));
  PetscCall(PEPGetST(pep,&st));
  PetscCall(STSetType(st,STSINVERT));

  /*
     Set solver options. In particular, we must allocate sufficient
     storage for all eigenpairs that may converge (ncv). This is
     application-dependent.
  */
  mpd = 40;
  PetscCall(PEPSetDimensions(pep,2*mpd,3*mpd,mpd));
  PetscCall(PEPSetTolerances(pep,PETSC_DEFAULT,2000));
  PetscCall(PetscNew(&ctx));
  ctx->lastnconv = 0;
  ctx->nreps     = 0;
  PetscCall(PEPSetStoppingTestFunction(pep,MyStoppingTest,(void*)ctx,NULL));

  PetscCall(PEPSetFromOptions(pep));

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
                      Solve the eigensystem
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

  PetscCall(PEPSolve(pep));

  /* show detailed info unless -terse option is given by user */
  PetscCall(PetscViewerASCIIGetStdout(PETSC_COMM_WORLD,&viewer));
  PetscCall(PetscViewerPushFormat(viewer,PETSC_VIEWER_ASCII_INFO_DETAIL));
  PetscCall(PEPConvergedReasonView(pep,viewer));
  PetscCall(PetscOptionsHasName(NULL,NULL,"-terse",&terse));
  if (!terse) PetscCall(PEPErrorView(pep,PEP_ERROR_BACKWARD,viewer));
  PetscCall(PetscViewerPopFormat(viewer));

  PetscCall(PEPDestroy(&pep));
  PetscCall(MatDestroy(&M));
  PetscCall(MatDestroy(&C));
  PetscCall(MatDestroy(&K));
  PetscCall(PetscFree(ctx));
  PetscCall(SlepcFinalize());
  return 0;
}

/*
    Function for user-defined stopping test.

    Ignores the value of nev. It only takes into account the number of
    eigenpairs that have converged in recent outer iterations (restarts);
    if no new eigenvalues have converged in the last few restarts,
    we stop the iteration, assuming that no more eigenvalues are present
    inside the region.
*/
PetscErrorCode MyStoppingTest(PEP pep,PetscInt its,PetscInt max_it,PetscInt nconv,PetscInt nev,PEPConvergedReason *reason,void *ptr)
{
  CTX_SPRING     *ctx = (CTX_SPRING*)ptr;

  PetscFunctionBeginUser;
  /* check usual termination conditions, but ignoring the case nconv>=nev */
  PetscCall(PEPStoppingBasic(pep,its,max_it,nconv,PETSC_MAX_INT,reason,NULL));
  if (*reason==PEP_CONVERGED_ITERATING) {
    /* check if nconv is the same as before */
    if (nconv==ctx->lastnconv) ctx->nreps++;
    else {
      ctx->lastnconv = nconv;
      ctx->nreps     = 0;
    }
    /* check if no eigenvalues converged in last 10 restarts */
    if (nconv && ctx->nreps>10) *reason = PEP_CONVERGED_USER;
  }
  PetscFunctionReturn(0);
}

/*TEST

   test:
      args: -terse
      suffix: 1

TEST*/
