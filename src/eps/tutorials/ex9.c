/*
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
   SLEPc - Scalable Library for Eigenvalue Problem Computations
   Copyright (c) 2002-2021, Universitat Politecnica de Valencia, Spain

   This file is part of SLEPc.
   SLEPc is distributed under a 2-clause BSD license (see LICENSE).
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
*/

static char help[] = "Solves a problem associated to the Brusselator wave model in chemical reactions, illustrating the use of shell matrices.\n\n"
  "The command line options are:\n"
  "  -n <n>, where <n> = block dimension of the 2x2 block matrix.\n"
  "  -L <L>, where <L> = bifurcation parameter.\n"
  "  -alpha <alpha>, -beta <beta>, -delta1 <delta1>,  -delta2 <delta2>,\n"
  "       where <alpha> <beta> <delta1> <delta2> = model parameters.\n\n";

#include <slepceps.h>

/*
   This example computes the eigenvalues with largest real part of the
   following matrix

        A = [ tau1*T+(beta-1)*I     alpha^2*I
                  -beta*I        tau2*T-alpha^2*I ],

   where

        T = tridiag{1,-2,1}
        h = 1/(n+1)
        tau1 = delta1/(h*L)^2
        tau2 = delta2/(h*L)^2
*/

/*
   Matrix operations
*/
PetscErrorCode MatMult_Brussel(Mat,Vec,Vec);
PetscErrorCode MatMultTranspose_Brussel(Mat,Vec,Vec);
PetscErrorCode MatGetDiagonal_Brussel(Mat,Vec);

typedef struct {
  Mat         T;
  Vec         x1,x2,y1,y2;
  PetscScalar alpha,beta,tau1,tau2,sigma;
} CTX_BRUSSEL;

int main(int argc,char **argv)
{
  Mat            A;               /* eigenvalue problem matrix */
  EPS            eps;             /* eigenproblem solver context */
  EPSType        type;
  PetscScalar    delta1,delta2,L,h;
  PetscInt       N=30,n,i,Istart,Iend,nev;
  CTX_BRUSSEL    *ctx;
  PetscBool      terse;
  PetscViewer    viewer;
  PetscErrorCode ierr;

  ierr = SlepcInitialize(&argc,&argv,(char*)0,help);if (ierr) return ierr;

  CHKERRQ(PetscOptionsGetInt(NULL,NULL,"-n",&N,NULL));
  CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"\nBrusselator wave model, n=%" PetscInt_FMT "\n\n",N));

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        Generate the matrix
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

  /*
     Create shell matrix context and set default parameters
  */
  CHKERRQ(PetscNew(&ctx));
  ctx->alpha = 2.0;
  ctx->beta  = 5.45;
  delta1     = 0.008;
  delta2     = 0.004;
  L          = 0.51302;

  /*
     Look the command line for user-provided parameters
  */
  CHKERRQ(PetscOptionsGetScalar(NULL,NULL,"-L",&L,NULL));
  CHKERRQ(PetscOptionsGetScalar(NULL,NULL,"-alpha",&ctx->alpha,NULL));
  CHKERRQ(PetscOptionsGetScalar(NULL,NULL,"-beta",&ctx->beta,NULL));
  CHKERRQ(PetscOptionsGetScalar(NULL,NULL,"-delta1",&delta1,NULL));
  CHKERRQ(PetscOptionsGetScalar(NULL,NULL,"-delta2",&delta2,NULL));

  /*
     Create matrix T
  */
  CHKERRQ(MatCreate(PETSC_COMM_WORLD,&ctx->T));
  CHKERRQ(MatSetSizes(ctx->T,PETSC_DECIDE,PETSC_DECIDE,N,N));
  CHKERRQ(MatSetFromOptions(ctx->T));
  CHKERRQ(MatSetUp(ctx->T));

  CHKERRQ(MatGetOwnershipRange(ctx->T,&Istart,&Iend));
  for (i=Istart;i<Iend;i++) {
    if (i>0) CHKERRQ(MatSetValue(ctx->T,i,i-1,1.0,INSERT_VALUES));
    if (i<N-1) CHKERRQ(MatSetValue(ctx->T,i,i+1,1.0,INSERT_VALUES));
    CHKERRQ(MatSetValue(ctx->T,i,i,-2.0,INSERT_VALUES));
  }
  CHKERRQ(MatAssemblyBegin(ctx->T,MAT_FINAL_ASSEMBLY));
  CHKERRQ(MatAssemblyEnd(ctx->T,MAT_FINAL_ASSEMBLY));
  CHKERRQ(MatGetLocalSize(ctx->T,&n,NULL));

  /*
     Fill the remaining information in the shell matrix context
     and create auxiliary vectors
  */
  h = 1.0 / (PetscReal)(N+1);
  ctx->tau1 = delta1 / ((h*L)*(h*L));
  ctx->tau2 = delta2 / ((h*L)*(h*L));
  ctx->sigma = 0.0;
  CHKERRQ(VecCreateMPIWithArray(PETSC_COMM_WORLD,1,n,PETSC_DECIDE,NULL,&ctx->x1));
  CHKERRQ(VecCreateMPIWithArray(PETSC_COMM_WORLD,1,n,PETSC_DECIDE,NULL,&ctx->x2));
  CHKERRQ(VecCreateMPIWithArray(PETSC_COMM_WORLD,1,n,PETSC_DECIDE,NULL,&ctx->y1));
  CHKERRQ(VecCreateMPIWithArray(PETSC_COMM_WORLD,1,n,PETSC_DECIDE,NULL,&ctx->y2));

  /*
     Create the shell matrix
  */
  CHKERRQ(MatCreateShell(PETSC_COMM_WORLD,2*n,2*n,2*N,2*N,(void*)ctx,&A));
  CHKERRQ(MatShellSetOperation(A,MATOP_MULT,(void(*)(void))MatMult_Brussel));
  CHKERRQ(MatShellSetOperation(A,MATOP_MULT_TRANSPOSE,(void(*)(void))MatMultTranspose_Brussel));
  CHKERRQ(MatShellSetOperation(A,MATOP_GET_DIAGONAL,(void(*)(void))MatGetDiagonal_Brussel));

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
                Create the eigensolver and set various options
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

  /*
     Create eigensolver context
  */
  CHKERRQ(EPSCreate(PETSC_COMM_WORLD,&eps));

  /*
     Set operators. In this case, it is a standard eigenvalue problem
  */
  CHKERRQ(EPSSetOperators(eps,A,NULL));
  CHKERRQ(EPSSetProblemType(eps,EPS_NHEP));

  /*
     Ask for the rightmost eigenvalues
  */
  CHKERRQ(EPSSetWhichEigenpairs(eps,EPS_LARGEST_REAL));

  /*
     Set other solver options at runtime
  */
  CHKERRQ(EPSSetFromOptions(eps));

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
                      Solve the eigensystem
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

  CHKERRQ(EPSSolve(eps));

  /*
     Optional: Get some information from the solver and display it
  */
  CHKERRQ(EPSGetType(eps,&type));
  CHKERRQ(PetscPrintf(PETSC_COMM_WORLD," Solution method: %s\n\n",type));
  CHKERRQ(EPSGetDimensions(eps,&nev,NULL,NULL));
  CHKERRQ(PetscPrintf(PETSC_COMM_WORLD," Number of requested eigenvalues: %" PetscInt_FMT "\n",nev));

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
                    Display solution and clean up
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

  /* show detailed info unless -terse option is given by user */
  CHKERRQ(PetscOptionsHasName(NULL,NULL,"-terse",&terse));
  if (terse) {
    CHKERRQ(EPSErrorView(eps,EPS_ERROR_RELATIVE,NULL));
  } else {
    CHKERRQ(PetscViewerASCIIGetStdout(PETSC_COMM_WORLD,&viewer));
    CHKERRQ(PetscViewerPushFormat(viewer,PETSC_VIEWER_ASCII_INFO_DETAIL));
    CHKERRQ(EPSConvergedReasonView(eps,viewer));
    CHKERRQ(EPSErrorView(eps,EPS_ERROR_RELATIVE,viewer));
    CHKERRQ(PetscViewerPopFormat(viewer));
  }
  CHKERRQ(EPSDestroy(&eps));
  CHKERRQ(MatDestroy(&A));
  CHKERRQ(MatDestroy(&ctx->T));
  CHKERRQ(VecDestroy(&ctx->x1));
  CHKERRQ(VecDestroy(&ctx->x2));
  CHKERRQ(VecDestroy(&ctx->y1));
  CHKERRQ(VecDestroy(&ctx->y2));
  CHKERRQ(PetscFree(ctx));
  ierr = SlepcFinalize();
  return ierr;
}

PetscErrorCode MatMult_Brussel(Mat A,Vec x,Vec y)
{
  PetscInt          n;
  const PetscScalar *px;
  PetscScalar       *py;
  CTX_BRUSSEL       *ctx;

  PetscFunctionBeginUser;
  CHKERRQ(MatShellGetContext(A,&ctx));
  CHKERRQ(MatGetLocalSize(ctx->T,&n,NULL));
  CHKERRQ(VecGetArrayRead(x,&px));
  CHKERRQ(VecGetArray(y,&py));
  CHKERRQ(VecPlaceArray(ctx->x1,px));
  CHKERRQ(VecPlaceArray(ctx->x2,px+n));
  CHKERRQ(VecPlaceArray(ctx->y1,py));
  CHKERRQ(VecPlaceArray(ctx->y2,py+n));

  CHKERRQ(MatMult(ctx->T,ctx->x1,ctx->y1));
  CHKERRQ(VecScale(ctx->y1,ctx->tau1));
  CHKERRQ(VecAXPY(ctx->y1,ctx->beta-1.0+ctx->sigma,ctx->x1));
  CHKERRQ(VecAXPY(ctx->y1,ctx->alpha*ctx->alpha,ctx->x2));

  CHKERRQ(MatMult(ctx->T,ctx->x2,ctx->y2));
  CHKERRQ(VecScale(ctx->y2,ctx->tau2));
  CHKERRQ(VecAXPY(ctx->y2,-ctx->beta,ctx->x1));
  CHKERRQ(VecAXPY(ctx->y2,-ctx->alpha*ctx->alpha+ctx->sigma,ctx->x2));

  CHKERRQ(VecRestoreArrayRead(x,&px));
  CHKERRQ(VecRestoreArray(y,&py));
  CHKERRQ(VecResetArray(ctx->x1));
  CHKERRQ(VecResetArray(ctx->x2));
  CHKERRQ(VecResetArray(ctx->y1));
  CHKERRQ(VecResetArray(ctx->y2));
  PetscFunctionReturn(0);
}

PetscErrorCode MatMultTranspose_Brussel(Mat A,Vec x,Vec y)
{
  PetscInt          n;
  const PetscScalar *px;
  PetscScalar       *py;
  CTX_BRUSSEL       *ctx;

  PetscFunctionBeginUser;
  CHKERRQ(MatShellGetContext(A,&ctx));
  CHKERRQ(MatGetLocalSize(ctx->T,&n,NULL));
  CHKERRQ(VecGetArrayRead(x,&px));
  CHKERRQ(VecGetArray(y,&py));
  CHKERRQ(VecPlaceArray(ctx->x1,px));
  CHKERRQ(VecPlaceArray(ctx->x2,px+n));
  CHKERRQ(VecPlaceArray(ctx->y1,py));
  CHKERRQ(VecPlaceArray(ctx->y2,py+n));

  CHKERRQ(MatMultTranspose(ctx->T,ctx->x1,ctx->y1));
  CHKERRQ(VecScale(ctx->y1,ctx->tau1));
  CHKERRQ(VecAXPY(ctx->y1,ctx->beta-1.0+ctx->sigma,ctx->x1));
  CHKERRQ(VecAXPY(ctx->y1,-ctx->beta,ctx->x2));

  CHKERRQ(MatMultTranspose(ctx->T,ctx->x2,ctx->y2));
  CHKERRQ(VecScale(ctx->y2,ctx->tau2));
  CHKERRQ(VecAXPY(ctx->y2,ctx->alpha*ctx->alpha,ctx->x1));
  CHKERRQ(VecAXPY(ctx->y2,-ctx->alpha*ctx->alpha+ctx->sigma,ctx->x2));

  CHKERRQ(VecRestoreArrayRead(x,&px));
  CHKERRQ(VecRestoreArray(y,&py));
  CHKERRQ(VecResetArray(ctx->x1));
  CHKERRQ(VecResetArray(ctx->x2));
  CHKERRQ(VecResetArray(ctx->y1));
  CHKERRQ(VecResetArray(ctx->y2));
  PetscFunctionReturn(0);
}

PetscErrorCode MatGetDiagonal_Brussel(Mat A,Vec diag)
{
  Vec            d1,d2;
  PetscInt       n;
  PetscScalar    *pd;
  MPI_Comm       comm;
  CTX_BRUSSEL    *ctx;

  PetscFunctionBeginUser;
  CHKERRQ(MatShellGetContext(A,&ctx));
  CHKERRQ(PetscObjectGetComm((PetscObject)A,&comm));
  CHKERRQ(MatGetLocalSize(ctx->T,&n,NULL));
  CHKERRQ(VecGetArray(diag,&pd));
  CHKERRQ(VecCreateMPIWithArray(comm,1,n,PETSC_DECIDE,pd,&d1));
  CHKERRQ(VecCreateMPIWithArray(comm,1,n,PETSC_DECIDE,pd+n,&d2));

  CHKERRQ(VecSet(d1,-2.0*ctx->tau1 + ctx->beta - 1.0 + ctx->sigma));
  CHKERRQ(VecSet(d2,-2.0*ctx->tau2 - ctx->alpha*ctx->alpha + ctx->sigma));

  CHKERRQ(VecDestroy(&d1));
  CHKERRQ(VecDestroy(&d2));
  CHKERRQ(VecRestoreArray(diag,&pd));
  PetscFunctionReturn(0);
}

/*TEST

   test:
      suffix: 1
      args: -n 50 -eps_nev 4 -eps_two_sided {{0 1}} -eps_type {{krylovschur lapack}} -terse
      requires: !single
      filter: grep -v method

   test:
      suffix: 2
      args: -eps_nev 8 -eps_max_it 300 -eps_target -28 -rg_type interval -rg_interval_endpoints -40,-20,-.1,.1 -terse
      requires: !single

   test:
      suffix: 3
      args: -n 50 -eps_nev 4 -eps_balance twoside -terse
      requires: double
      filter: grep -v method
      output_file: output/ex9_1.out

   test:
      suffix: 4
      args: -eps_smallest_imaginary -eps_ncv 24 -terse
      requires: !complex !single

   test:
      suffix: 4_complex
      args: -eps_smallest_imaginary -eps_ncv 24 -terse
      requires: complex !single

   test:
      suffix: 5
      args: -eps_nev 4 -eps_target_real -eps_target -3 -terse
      requires: !single

   test:
      suffix: 6
      args: -eps_nev 2 -eps_target_imaginary -eps_target 3i -terse
      requires: complex !single

   test:
      suffix: 7
      args: -n 40 -eps_nev 1 -eps_type arnoldi -eps_smallest_real -eps_refined -eps_ncv 40 -eps_max_it 300 -terse
      requires: double

   test:
      suffix: 8
      args: -eps_nev 2 -eps_target -30 -eps_type jd -st_matmode shell -eps_jd_fix 0.0001 -eps_jd_const_correction_tol 0 -terse
      requires: !single
      filter: sed -e "s/[+-]0\.0*i//g"

   test:
      suffix: 9
      args: -eps_largest_imaginary -eps_ncv 24 -terse
      requires: !single

TEST*/
