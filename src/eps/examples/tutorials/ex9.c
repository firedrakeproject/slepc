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
PetscErrorCode MatShift_Brussel(PetscScalar*,Mat);
PetscErrorCode MatGetDiagonal_Brussel(Mat,Vec);

typedef struct {
  Mat         T;
  Vec         x1,x2,y1,y2;
  PetscScalar alpha,beta,tau1,tau2,sigma;
} CTX_BRUSSEL;

#undef __FUNCT__
#define __FUNCT__ "main"
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

  SlepcInitialize(&argc,&argv,(char*)0,help);

  ierr = PetscOptionsGetInt(NULL,NULL,"-n",&N,NULL);CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD,"\nBrusselator wave model, n=%D\n\n",N);CHKERRQ(ierr);

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        Generate the matrix
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

  /*
     Create shell matrix context and set default parameters
  */
  ierr = PetscNew(&ctx);CHKERRQ(ierr);
  ctx->alpha = 2.0;
  ctx->beta  = 5.45;
  delta1     = 0.008;
  delta2     = 0.004;
  L          = 0.51302;

  /*
     Look the command line for user-provided parameters
  */
  ierr = PetscOptionsGetScalar(NULL,NULL,"-L",&L,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetScalar(NULL,NULL,"-alpha",&ctx->alpha,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetScalar(NULL,NULL,"-beta",&ctx->beta,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetScalar(NULL,NULL,"-delta1",&delta1,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetScalar(NULL,NULL,"-delta2",&delta2,NULL);CHKERRQ(ierr);

  /*
     Create matrix T
  */
  ierr = MatCreate(PETSC_COMM_WORLD,&ctx->T);CHKERRQ(ierr);
  ierr = MatSetSizes(ctx->T,PETSC_DECIDE,PETSC_DECIDE,N,N);CHKERRQ(ierr);
  ierr = MatSetFromOptions(ctx->T);CHKERRQ(ierr);
  ierr = MatSetUp(ctx->T);CHKERRQ(ierr);

  ierr = MatGetOwnershipRange(ctx->T,&Istart,&Iend);CHKERRQ(ierr);
  for (i=Istart;i<Iend;i++) {
    if (i>0) { ierr = MatSetValue(ctx->T,i,i-1,1.0,INSERT_VALUES);CHKERRQ(ierr); }
    if (i<N-1) { ierr = MatSetValue(ctx->T,i,i+1,1.0,INSERT_VALUES);CHKERRQ(ierr); }
    ierr = MatSetValue(ctx->T,i,i,-2.0,INSERT_VALUES);CHKERRQ(ierr);
  }
  ierr = MatAssemblyBegin(ctx->T,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(ctx->T,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatGetLocalSize(ctx->T,&n,NULL);CHKERRQ(ierr);

  /*
     Fill the remaining information in the shell matrix context
     and create auxiliary vectors
  */
  h = 1.0 / (PetscReal)(N+1);
  ctx->tau1 = delta1 / ((h*L)*(h*L));
  ctx->tau2 = delta2 / ((h*L)*(h*L));
  ctx->sigma = 0.0;
  ierr = VecCreateMPIWithArray(PETSC_COMM_WORLD,1,n,PETSC_DECIDE,NULL,&ctx->x1);CHKERRQ(ierr);
  ierr = VecCreateMPIWithArray(PETSC_COMM_WORLD,1,n,PETSC_DECIDE,NULL,&ctx->x2);CHKERRQ(ierr);
  ierr = VecCreateMPIWithArray(PETSC_COMM_WORLD,1,n,PETSC_DECIDE,NULL,&ctx->y1);CHKERRQ(ierr);
  ierr = VecCreateMPIWithArray(PETSC_COMM_WORLD,1,n,PETSC_DECIDE,NULL,&ctx->y2);CHKERRQ(ierr);

  /*
     Create the shell matrix
  */
  ierr = MatCreateShell(PETSC_COMM_WORLD,2*n,2*n,2*N,2*N,(void*)ctx,&A);CHKERRQ(ierr);
  ierr = MatShellSetOperation(A,MATOP_MULT,(void(*)())MatMult_Brussel);CHKERRQ(ierr);
  ierr = MatShellSetOperation(A,MATOP_SHIFT,(void(*)())MatShift_Brussel);CHKERRQ(ierr);
  ierr = MatShellSetOperation(A,MATOP_GET_DIAGONAL,(void(*)())MatGetDiagonal_Brussel);CHKERRQ(ierr);

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
                Create the eigensolver and set various options
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

  /*
     Create eigensolver context
  */
  ierr = EPSCreate(PETSC_COMM_WORLD,&eps);CHKERRQ(ierr);

  /*
     Set operators. In this case, it is a standard eigenvalue problem
  */
  ierr = EPSSetOperators(eps,A,NULL);CHKERRQ(ierr);
  ierr = EPSSetProblemType(eps,EPS_NHEP);CHKERRQ(ierr);

  /*
     Ask for the rightmost eigenvalues
  */
  ierr = EPSSetWhichEigenpairs(eps,EPS_LARGEST_REAL);CHKERRQ(ierr);

  /*
     Set other solver options at runtime
  */
  ierr = EPSSetFromOptions(eps);CHKERRQ(ierr);

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
                      Solve the eigensystem
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

  ierr = EPSSolve(eps);CHKERRQ(ierr);

  /*
     Optional: Get some information from the solver and display it
  */
  ierr = EPSGetType(eps,&type);CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD," Solution method: %s\n\n",type);CHKERRQ(ierr);
  ierr = EPSGetDimensions(eps,&nev,NULL,NULL);CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD," Number of requested eigenvalues: %D\n",nev);CHKERRQ(ierr);

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
                    Display solution and clean up
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

  /* show detailed info unless -terse option is given by user */
  ierr = PetscOptionsHasName(NULL,NULL,"-terse",&terse);CHKERRQ(ierr);
  if (terse) {
    ierr = EPSErrorView(eps,EPS_ERROR_RELATIVE,NULL);CHKERRQ(ierr);
  } else {
    ierr = PetscViewerASCIIGetStdout(PETSC_COMM_WORLD,&viewer);CHKERRQ(ierr);
    ierr = PetscViewerPushFormat(viewer,PETSC_VIEWER_ASCII_INFO_DETAIL);CHKERRQ(ierr);
    ierr = EPSReasonView(eps,viewer);CHKERRQ(ierr);
    ierr = EPSErrorView(eps,EPS_ERROR_RELATIVE,viewer);CHKERRQ(ierr);
    ierr = PetscViewerPopFormat(viewer);CHKERRQ(ierr);
  }
  ierr = EPSDestroy(&eps);CHKERRQ(ierr);
  ierr = MatDestroy(&A);CHKERRQ(ierr);
  ierr = MatDestroy(&ctx->T);CHKERRQ(ierr);
  ierr = VecDestroy(&ctx->x1);CHKERRQ(ierr);
  ierr = VecDestroy(&ctx->x2);CHKERRQ(ierr);
  ierr = VecDestroy(&ctx->y1);CHKERRQ(ierr);
  ierr = VecDestroy(&ctx->y2);CHKERRQ(ierr);
  ierr = PetscFree(ctx);CHKERRQ(ierr);
  ierr = SlepcFinalize();
  return ierr;
}

#undef __FUNCT__
#define __FUNCT__ "MatMult_Brussel"
PetscErrorCode MatMult_Brussel(Mat A,Vec x,Vec y)
{
  PetscInt          n;
  const PetscScalar *px;
  PetscScalar       *py;
  CTX_BRUSSEL       *ctx;
  PetscErrorCode    ierr;

  PetscFunctionBeginUser;
  ierr = MatShellGetContext(A,(void**)&ctx);CHKERRQ(ierr);
  ierr = MatGetLocalSize(ctx->T,&n,NULL);CHKERRQ(ierr);
  ierr = VecGetArrayRead(x,&px);CHKERRQ(ierr);
  ierr = VecGetArray(y,&py);CHKERRQ(ierr);
  ierr = VecPlaceArray(ctx->x1,px);CHKERRQ(ierr);
  ierr = VecPlaceArray(ctx->x2,px+n);CHKERRQ(ierr);
  ierr = VecPlaceArray(ctx->y1,py);CHKERRQ(ierr);
  ierr = VecPlaceArray(ctx->y2,py+n);CHKERRQ(ierr);

  ierr = MatMult(ctx->T,ctx->x1,ctx->y1);CHKERRQ(ierr);
  ierr = VecScale(ctx->y1,ctx->tau1);CHKERRQ(ierr);
  ierr = VecAXPY(ctx->y1,ctx->beta - 1.0 + ctx->sigma,ctx->x1);CHKERRQ(ierr);
  ierr = VecAXPY(ctx->y1,ctx->alpha * ctx->alpha,ctx->x2);CHKERRQ(ierr);

  ierr = MatMult(ctx->T,ctx->x2,ctx->y2);CHKERRQ(ierr);
  ierr = VecScale(ctx->y2,ctx->tau2);CHKERRQ(ierr);
  ierr = VecAXPY(ctx->y2,-ctx->beta,ctx->x1);CHKERRQ(ierr);
  ierr = VecAXPY(ctx->y2,-ctx->alpha * ctx->alpha + ctx->sigma,ctx->x2);CHKERRQ(ierr);

  ierr = VecRestoreArrayRead(x,&px);CHKERRQ(ierr);
  ierr = VecRestoreArray(y,&py);CHKERRQ(ierr);
  ierr = VecResetArray(ctx->x1);CHKERRQ(ierr);
  ierr = VecResetArray(ctx->x2);CHKERRQ(ierr);
  ierr = VecResetArray(ctx->y1);CHKERRQ(ierr);
  ierr = VecResetArray(ctx->y2);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatShift_Brussel"
PetscErrorCode MatShift_Brussel(PetscScalar* a,Mat Y)
{
  CTX_BRUSSEL    *ctx;
  PetscErrorCode ierr;

  PetscFunctionBeginUser;
  ierr = MatShellGetContext(Y,(void**)&ctx);CHKERRQ(ierr);
  ctx->sigma += *a;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatGetDiagonal_Brussel"
PetscErrorCode MatGetDiagonal_Brussel(Mat A,Vec diag)
{
  Vec            d1,d2;
  PetscInt       n;
  PetscScalar    *pd;
  MPI_Comm       comm;
  CTX_BRUSSEL    *ctx;
  PetscErrorCode ierr;

  PetscFunctionBeginUser;
  ierr = MatShellGetContext(A,(void**)&ctx);CHKERRQ(ierr);
  ierr = PetscObjectGetComm((PetscObject)A,&comm);CHKERRQ(ierr);
  ierr = MatGetLocalSize(ctx->T,&n,NULL);CHKERRQ(ierr);
  ierr = VecGetArray(diag,&pd);CHKERRQ(ierr);
  ierr = VecCreateMPIWithArray(comm,1,n,PETSC_DECIDE,pd,&d1);CHKERRQ(ierr);
  ierr = VecCreateMPIWithArray(comm,1,n,PETSC_DECIDE,pd+n,&d2);CHKERRQ(ierr);

  ierr = VecSet(d1,-2.0*ctx->tau1 + ctx->beta - 1.0 + ctx->sigma);CHKERRQ(ierr);
  ierr = VecSet(d2,-2.0*ctx->tau2 - ctx->alpha*ctx->alpha + ctx->sigma);CHKERRQ(ierr);

  ierr = VecDestroy(&d1);CHKERRQ(ierr);
  ierr = VecDestroy(&d2);CHKERRQ(ierr);
  ierr = VecRestoreArray(diag,&pd);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

