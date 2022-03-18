/*
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
   SLEPc - Scalable Library for Eigenvalue Problem Computations
   Copyright (c) 2002-2021, Universitat Politecnica de Valencia, Spain

   This file is part of SLEPc.
   SLEPc is distributed under a 2-clause BSD license (see LICENSE).
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
*/

static char help[] = "Illustrates how to obtain invariant subspaces. "
  "Based on ex9.\n"
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

/* Matrix operations */
PetscErrorCode MatMult_Brussel(Mat,Vec,Vec);
PetscErrorCode MatGetDiagonal_Brussel(Mat,Vec);

typedef struct {
  Mat         T;
  Vec         x1,x2,y1,y2;
  PetscScalar alpha,beta,tau1,tau2,sigma;
} CTX_BRUSSEL;

int main(int argc,char **argv)
{
  EPS            eps;
  Mat            A;
  Vec            *Q,v;
  PetscScalar    delta1,delta2,L,h,kr,ki;
  PetscReal      errest,tol,re,im,lev;
  PetscInt       N=30,n,i,j,Istart,Iend,nev,nconv;
  CTX_BRUSSEL    *ctx;
  PetscBool      errok,trueres;
  PetscErrorCode ierr;

  ierr = SlepcInitialize(&argc,&argv,(char*)0,help);if (ierr) return ierr;
  CHKERRQ(PetscOptionsGetInt(NULL,NULL,"-n",&N,NULL));
  CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"\nBrusselator wave model, n=%" PetscInt_FMT "\n\n",N));

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        Generate the matrix
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

  CHKERRQ(PetscNew(&ctx));
  ctx->alpha = 2.0;
  ctx->beta  = 5.45;
  delta1     = 0.008;
  delta2     = 0.004;
  L          = 0.51302;

  CHKERRQ(PetscOptionsGetScalar(NULL,NULL,"-L",&L,NULL));
  CHKERRQ(PetscOptionsGetScalar(NULL,NULL,"-alpha",&ctx->alpha,NULL));
  CHKERRQ(PetscOptionsGetScalar(NULL,NULL,"-beta",&ctx->beta,NULL));
  CHKERRQ(PetscOptionsGetScalar(NULL,NULL,"-delta1",&delta1,NULL));
  CHKERRQ(PetscOptionsGetScalar(NULL,NULL,"-delta2",&delta2,NULL));

  /* Create matrix T */
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

  /* Fill the remaining information in the shell matrix context
     and create auxiliary vectors */
  h = 1.0 / (PetscReal)(N+1);
  ctx->tau1 = delta1 / ((h*L)*(h*L));
  ctx->tau2 = delta2 / ((h*L)*(h*L));
  ctx->sigma = 0.0;
  CHKERRQ(VecCreateMPIWithArray(PETSC_COMM_WORLD,1,n,PETSC_DECIDE,NULL,&ctx->x1));
  CHKERRQ(VecCreateMPIWithArray(PETSC_COMM_WORLD,1,n,PETSC_DECIDE,NULL,&ctx->x2));
  CHKERRQ(VecCreateMPIWithArray(PETSC_COMM_WORLD,1,n,PETSC_DECIDE,NULL,&ctx->y1));
  CHKERRQ(VecCreateMPIWithArray(PETSC_COMM_WORLD,1,n,PETSC_DECIDE,NULL,&ctx->y2));

  /* Create the shell matrix */
  CHKERRQ(MatCreateShell(PETSC_COMM_WORLD,2*n,2*n,2*N,2*N,(void*)ctx,&A));
  CHKERRQ(MatShellSetOperation(A,MATOP_MULT,(void(*)(void))MatMult_Brussel));
  CHKERRQ(MatShellSetOperation(A,MATOP_GET_DIAGONAL,(void(*)(void))MatGetDiagonal_Brussel));

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
                Create the eigensolver and solve the problem
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

  CHKERRQ(EPSCreate(PETSC_COMM_WORLD,&eps));
  CHKERRQ(EPSSetOperators(eps,A,NULL));
  CHKERRQ(EPSSetProblemType(eps,EPS_NHEP));
  CHKERRQ(EPSSetWhichEigenpairs(eps,EPS_LARGEST_REAL));
  CHKERRQ(EPSSetTrueResidual(eps,PETSC_FALSE));
  CHKERRQ(EPSSetFromOptions(eps));
  CHKERRQ(EPSSolve(eps));

  CHKERRQ(EPSGetTrueResidual(eps,&trueres));
  /*if (trueres) CHKERRQ(PetscPrintf(PETSC_COMM_WORLD," Computing true residuals explicitly\n\n"));*/

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
                    Display solution and clean up
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

  CHKERRQ(EPSGetDimensions(eps,&nev,NULL,NULL));
  CHKERRQ(EPSGetTolerances(eps,&tol,NULL));
  CHKERRQ(EPSGetConverged(eps,&nconv));
  if (nconv<nev) {
    CHKERRQ(PetscPrintf(PETSC_COMM_WORLD," Problem: less than %" PetscInt_FMT " eigenvalues converged\n\n",nev));
  } else {
    /* Check that all converged eigenpairs satisfy the requested tolerance
       (in this example we use the solver's error estimate instead of computing
       the residual norm explicitly) */
    errok = PETSC_TRUE;
    for (i=0;i<nev;i++) {
      CHKERRQ(EPSGetErrorEstimate(eps,i,&errest));
      CHKERRQ(EPSGetEigenpair(eps,i,&kr,&ki,NULL,NULL));
      errok = (errok && errest<5.0*SlepcAbsEigenvalue(kr,ki)*tol)? PETSC_TRUE: PETSC_FALSE;
    }
    if (!errok) {
      CHKERRQ(PetscPrintf(PETSC_COMM_WORLD," Problem: some of the first %" PetscInt_FMT " relative errors are higher than the tolerance\n\n",nev));
    } else {
      CHKERRQ(PetscPrintf(PETSC_COMM_WORLD," All requested eigenvalues computed up to the required tolerance:"));
      for (i=0;i<=(nev-1)/8;i++) {
        CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"\n     "));
        for (j=0;j<PetscMin(8,nev-8*i);j++) {
          CHKERRQ(EPSGetEigenpair(eps,8*i+j,&kr,&ki,NULL,NULL));
#if defined(PETSC_USE_COMPLEX)
          re = PetscRealPart(kr);
          im = PetscImaginaryPart(kr);
#else
          re = kr;
          im = ki;
#endif
          if (PetscAbs(re)/PetscAbs(im)<PETSC_SMALL) re = 0.0;
          if (PetscAbs(im)/PetscAbs(re)<PETSC_SMALL) im = 0.0;
          if (im!=0.0) {
            CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"%.5f%+.5fi",(double)re,(double)im));
          } else {
            CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"%.5f",(double)re));
          }
          if (8*i+j+1<nev) CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,", "));
        }
      }
      CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"\n\n"));
    }
  }

  /* Get an orthogonal basis of the invariant subspace and check it is indeed
     orthogonal (note that eigenvectors are not orthogonal in this case) */
  if (nconv>1) {
    CHKERRQ(MatCreateVecs(A,&v,NULL));
    CHKERRQ(VecDuplicateVecs(v,nconv,&Q));
    CHKERRQ(EPSGetInvariantSubspace(eps,Q));
    CHKERRQ(VecCheckOrthonormality(Q,nconv,NULL,nconv,NULL,NULL,&lev));
    if (lev<10*tol) {
      CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"Level of orthogonality below the tolerance\n"));
    } else {
      CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"Level of orthogonality: %g\n",(double)lev));
    }
    CHKERRQ(VecDestroyVecs(nconv,&Q));
    CHKERRQ(VecDestroy(&v));
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

/*TEST

   test:
      suffix: 1
      args: -eps_nev 4 -eps_true_residual {{0 1}}
      requires: !single

   test:
      suffix: 2
      args: -eps_nev 4 -eps_true_residual -eps_balance oneside -eps_tol 1e-7
      requires: !single

   test:
      suffix: 3
      args: -n 50 -eps_nev 4 -eps_ncv 16 -eps_type subspace -eps_largest_magnitude -bv_orthog_block {{gs tsqr chol tsqrchol svqb}}
      requires: !single

TEST*/
