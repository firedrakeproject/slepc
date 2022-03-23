/*
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
   SLEPc - Scalable Library for Eigenvalue Problem Computations
   Copyright (c) 2002-2021, Universitat Politecnica de Valencia, Spain

   This file is part of SLEPc.
   SLEPc is distributed under a 2-clause BSD license (see LICENSE).
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
*/

static char help[] = "Illustrates the use of a region for filtering; the number of wanted eigenvalues is not known a priori.\n\n"
  "The problem is the Brusselator wave model as in ex9.c.\n"
  "The command line options are:\n"
  "  -n <n>, where <n> = block dimension of the 2x2 block matrix.\n"
  "  -L <L>, where <L> = bifurcation parameter.\n"
  "  -alpha <alpha>, -beta <beta>, -delta1 <delta1>,  -delta2 <delta2>,\n"
  "       where <alpha> <beta> <delta1> <delta2> = model parameters.\n\n";

#include <slepceps.h>

/*
   This example tries to compute all eigenvalues lying outside the real axis.
   This could be achieved by computing LARGEST_IMAGINARY eigenvalues, but
   here we take a different route: define a region of the complex plane where
   eigenvalues must be emphasized (eigenvalues outside the region are filtered
   out). In this case, we select the region as the complement of a thin stripe
   around the real axis.
 */

PetscErrorCode MatMult_Brussel(Mat,Vec,Vec);
PetscErrorCode MatGetDiagonal_Brussel(Mat,Vec);
PetscErrorCode MyStoppingTest(EPS,PetscInt,PetscInt,PetscInt,PetscInt,EPSConvergedReason*,void*);

typedef struct {
  Mat         T;
  Vec         x1,x2,y1,y2;
  PetscScalar alpha,beta,tau1,tau2,sigma;
  PetscInt    lastnconv;      /* last value of nconv; used in stopping test */
  PetscInt    nreps;          /* number of repetitions of nconv; used in stopping test */
} CTX_BRUSSEL;

int main(int argc,char **argv)
{
  Mat            A;               /* eigenvalue problem matrix */
  EPS            eps;             /* eigenproblem solver context */
  RG             rg;              /* region object */
  PetscScalar    delta1,delta2,L,h;
  PetscInt       N=30,n,i,Istart,Iend,mpd;
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
  CHKERRQ(MatShellSetOperation(A,MATOP_GET_DIAGONAL,(void(*)(void))MatGetDiagonal_Brussel));

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
                Create the eigensolver and configure the region
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

  CHKERRQ(EPSCreate(PETSC_COMM_WORLD,&eps));
  CHKERRQ(EPSSetOperators(eps,A,NULL));
  CHKERRQ(EPSSetProblemType(eps,EPS_NHEP));

  /*
     Define the region containing the eigenvalues of interest
  */
  CHKERRQ(EPSGetRG(eps,&rg));
  CHKERRQ(RGSetType(rg,RGINTERVAL));
  CHKERRQ(RGIntervalSetEndpoints(rg,-PETSC_INFINITY,PETSC_INFINITY,-0.01,0.01));
  CHKERRQ(RGSetComplement(rg,PETSC_TRUE));
  /* sort eigenvalue approximations wrt a target, otherwise convergence will be erratic */
  CHKERRQ(EPSSetTarget(eps,0.0));
  CHKERRQ(EPSSetWhichEigenpairs(eps,EPS_TARGET_MAGNITUDE));

  /*
     Set solver options. In particular, we must allocate sufficient
     storage for all eigenpairs that may converge (ncv). This is
     application-dependent.
  */
  mpd = 40;
  CHKERRQ(EPSSetDimensions(eps,2*mpd,3*mpd,mpd));
  CHKERRQ(EPSSetTolerances(eps,1e-7,2000));
  ctx->lastnconv = 0;
  ctx->nreps     = 0;
  CHKERRQ(EPSSetStoppingTestFunction(eps,MyStoppingTest,(void*)ctx,NULL));
  CHKERRQ(EPSSetFromOptions(eps));

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
                Solve the eigensystem and display solution
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

  CHKERRQ(EPSSolve(eps));

  /* show detailed info unless -terse option is given by user */
  CHKERRQ(PetscViewerASCIIGetStdout(PETSC_COMM_WORLD,&viewer));
  CHKERRQ(PetscViewerPushFormat(viewer,PETSC_VIEWER_ASCII_INFO_DETAIL));
  CHKERRQ(EPSConvergedReasonView(eps,viewer));
  CHKERRQ(PetscOptionsHasName(NULL,NULL,"-terse",&terse));
  if (!terse) CHKERRQ(EPSErrorView(eps,EPS_ERROR_RELATIVE,viewer));
  CHKERRQ(PetscViewerPopFormat(viewer));

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
  CHKERRQ(VecAXPY(ctx->y1,ctx->beta - 1.0 + ctx->sigma,ctx->x1));
  CHKERRQ(VecAXPY(ctx->y1,ctx->alpha * ctx->alpha,ctx->x2));

  CHKERRQ(MatMult(ctx->T,ctx->x2,ctx->y2));
  CHKERRQ(VecScale(ctx->y2,ctx->tau2));
  CHKERRQ(VecAXPY(ctx->y2,-ctx->beta,ctx->x1));
  CHKERRQ(VecAXPY(ctx->y2,-ctx->alpha * ctx->alpha + ctx->sigma,ctx->x2));

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

/*
    Function for user-defined stopping test.

    Ignores the value of nev. It only takes into account the number of
    eigenpairs that have converged in recent outer iterations (restarts);
    if no new eigenvalues have converged in the last few restarts,
    we stop the iteration, assuming that no more eigenvalues are present
    inside the region.
*/
PetscErrorCode MyStoppingTest(EPS eps,PetscInt its,PetscInt max_it,PetscInt nconv,PetscInt nev,EPSConvergedReason *reason,void *ptr)
{
  CTX_BRUSSEL    *ctx = (CTX_BRUSSEL*)ptr;

  PetscFunctionBeginUser;
  /* check usual termination conditions, but ignoring the case nconv>=nev */
  CHKERRQ(EPSStoppingBasic(eps,its,max_it,nconv,PETSC_MAX_INT,reason,NULL));
  if (*reason==EPS_CONVERGED_ITERATING) {
    /* check if nconv is the same as before */
    if (nconv==ctx->lastnconv) ctx->nreps++;
    else {
      ctx->lastnconv = nconv;
      ctx->nreps     = 0;
    }
    /* check if no eigenvalues converged in last 10 restarts */
    if (nconv && ctx->nreps>10) *reason = EPS_CONVERGED_USER;
  }
  PetscFunctionReturn(0);
}

/*TEST

   test:
      suffix: 1
      args: -n 100 -terse
      requires: !single

TEST*/
