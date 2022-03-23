/*
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
   SLEPc - Scalable Library for Eigenvalue Problem Computations
   Copyright (c) 2002-2021, Universitat Politecnica de Valencia, Spain

   This file is part of SLEPc.
   SLEPc is distributed under a 2-clause BSD license (see LICENSE).
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
*/

static char help[] = "Test the NLEIGS solver with shell matrices.\n\n"
  "This is based on ex27.\n"
  "The command line options are:\n"
  "  -n <n>, where <n> = matrix dimension.\n"
  "  -split <0/1>, to select the split form in the problem definition (enabled by default).\n";

/*
   Solve T(lambda)x=0 using NLEIGS solver
      with T(lambda) = -D+sqrt(lambda)*I
      where D is the Laplacian operator in 1 dimension
      and with the interpolation interval [.01,16]
*/

#include <slepcnep.h>

/* User-defined routines */
PetscErrorCode FormFunction(NEP,PetscScalar,Mat,Mat,void*);
PetscErrorCode ComputeSingularities(NEP,PetscInt*,PetscScalar*,void*);
PetscErrorCode MatMult_A0(Mat,Vec,Vec);
PetscErrorCode MatGetDiagonal_A0(Mat,Vec);
PetscErrorCode MatDuplicate_A0(Mat,MatDuplicateOption,Mat*);
PetscErrorCode MatMult_A1(Mat,Vec,Vec);
PetscErrorCode MatGetDiagonal_A1(Mat,Vec);
PetscErrorCode MatDuplicate_A1(Mat,MatDuplicateOption,Mat*);
PetscErrorCode MatMult_F(Mat,Vec,Vec);
PetscErrorCode MatGetDiagonal_F(Mat,Vec);
PetscErrorCode MatDuplicate_F(Mat,MatDuplicateOption,Mat*);
PetscErrorCode MatDestroy_F(Mat);

typedef struct {
  PetscScalar t;  /* square root of lambda */
} MatCtx;

int main(int argc,char **argv)
{
  NEP            nep;
  KSP            *ksp;
  PC             pc;
  Mat            F,A[2];
  NEPType        type;
  PetscInt       i,n=100,nev,its,nsolve;
  PetscReal      keep,tol=PETSC_SQRT_MACHINE_EPSILON/10;
  PetscErrorCode ierr;
  RG             rg;
  FN             f[2];
  PetscBool      terse,flg,lock,split=PETSC_TRUE;
  PetscScalar    coeffs;
  MatCtx         *ctx;

  ierr = SlepcInitialize(&argc,&argv,(char*)0,help);if (ierr) return ierr;
  CHKERRQ(PetscOptionsGetInt(NULL,NULL,"-n",&n,NULL));
  CHKERRQ(PetscOptionsGetBool(NULL,NULL,"-split",&split,NULL));
  CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"\nSquare root eigenproblem, n=%" PetscInt_FMT "%s\n\n",n,split?" (in split form)":""));

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Create NEP context, configure NLEIGS with appropriate options
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

  CHKERRQ(NEPCreate(PETSC_COMM_WORLD,&nep));
  CHKERRQ(NEPSetType(nep,NEPNLEIGS));
  CHKERRQ(NEPNLEIGSSetSingularitiesFunction(nep,ComputeSingularities,NULL));
  CHKERRQ(NEPGetRG(nep,&rg));
  CHKERRQ(RGSetType(rg,RGINTERVAL));
#if defined(PETSC_USE_COMPLEX)
  CHKERRQ(RGIntervalSetEndpoints(rg,0.01,16.0,-0.001,0.001));
#else
  CHKERRQ(RGIntervalSetEndpoints(rg,0.01,16.0,0,0));
#endif
  CHKERRQ(NEPSetTarget(nep,1.1));
  CHKERRQ(NEPNLEIGSGetKSPs(nep,&nsolve,&ksp));
  for (i=0;i<nsolve;i++) {
   CHKERRQ(KSPSetType(ksp[i],KSPBICG));
   CHKERRQ(KSPGetPC(ksp[i],&pc));
   CHKERRQ(PCSetType(pc,PCJACOBI));
   CHKERRQ(KSPSetTolerances(ksp[i],tol,PETSC_DEFAULT,PETSC_DEFAULT,PETSC_DEFAULT));
  }

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Define the nonlinear problem
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

  if (split) {
    /* Create matrix A0 (tridiagonal) */
    CHKERRQ(MatCreateShell(PETSC_COMM_WORLD,PETSC_DECIDE,PETSC_DECIDE,n,n,NULL,&A[0]));
    CHKERRQ(MatShellSetOperation(A[0],MATOP_MULT,(void(*)(void))MatMult_A0));
    CHKERRQ(MatShellSetOperation(A[0],MATOP_MULT_TRANSPOSE,(void(*)(void))MatMult_A0));
    CHKERRQ(MatShellSetOperation(A[0],MATOP_GET_DIAGONAL,(void(*)(void))MatGetDiagonal_A0));
    CHKERRQ(MatShellSetOperation(A[0],MATOP_DUPLICATE,(void(*)(void))MatDuplicate_A0));

    /* Create matrix A0 (identity) */
    CHKERRQ(MatCreateShell(PETSC_COMM_WORLD,PETSC_DECIDE,PETSC_DECIDE,n,n,NULL,&A[1]));
    CHKERRQ(MatShellSetOperation(A[1],MATOP_MULT,(void(*)(void))MatMult_A1));
    CHKERRQ(MatShellSetOperation(A[1],MATOP_MULT_TRANSPOSE,(void(*)(void))MatMult_A1));
    CHKERRQ(MatShellSetOperation(A[1],MATOP_GET_DIAGONAL,(void(*)(void))MatGetDiagonal_A1));
    CHKERRQ(MatShellSetOperation(A[1],MATOP_DUPLICATE,(void(*)(void))MatDuplicate_A1));

    /* Define functions for the split form */
    CHKERRQ(FNCreate(PETSC_COMM_WORLD,&f[0]));
    CHKERRQ(FNSetType(f[0],FNRATIONAL));
    coeffs = 1.0;
    CHKERRQ(FNRationalSetNumerator(f[0],1,&coeffs));
    CHKERRQ(FNCreate(PETSC_COMM_WORLD,&f[1]));
    CHKERRQ(FNSetType(f[1],FNSQRT));
    CHKERRQ(NEPSetSplitOperator(nep,2,A,f,SUBSET_NONZERO_PATTERN));
  } else {
    /* Callback form: create shell matrix for F=A0+sqrt(lambda)*A1  */
    CHKERRQ(PetscNew(&ctx));
    CHKERRQ(MatCreateShell(PETSC_COMM_WORLD,PETSC_DECIDE,PETSC_DECIDE,n,n,(void*)ctx,&F));
    CHKERRQ(MatShellSetOperation(F,MATOP_MULT,(void(*)(void))MatMult_F));
    CHKERRQ(MatShellSetOperation(F,MATOP_MULT_TRANSPOSE,(void(*)(void))MatMult_F));
    CHKERRQ(MatShellSetOperation(F,MATOP_GET_DIAGONAL,(void(*)(void))MatGetDiagonal_F));
    CHKERRQ(MatShellSetOperation(F,MATOP_DUPLICATE,(void(*)(void))MatDuplicate_F));
    CHKERRQ(MatShellSetOperation(F,MATOP_DESTROY,(void(*)(void))MatDestroy_F));
    /* Set Function evaluation routine */
    CHKERRQ(NEPSetFunction(nep,F,F,FormFunction,NULL));
  }

  /* Set solver parameters at runtime */
  CHKERRQ(NEPSetFromOptions(nep));

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
                      Solve the eigensystem
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  CHKERRQ(NEPSolve(nep));
  CHKERRQ(NEPGetType(nep,&type));
  CHKERRQ(PetscPrintf(PETSC_COMM_WORLD," Solution method: %s\n",type));
  CHKERRQ(NEPGetDimensions(nep,&nev,NULL,NULL));
  CHKERRQ(PetscPrintf(PETSC_COMM_WORLD," Number of requested eigenvalues: %" PetscInt_FMT "\n",nev));
  CHKERRQ(PetscObjectTypeCompare((PetscObject)nep,NEPNLEIGS,&flg));
  if (flg) {
    CHKERRQ(NEPNLEIGSGetRestart(nep,&keep));
    CHKERRQ(NEPNLEIGSGetLocking(nep,&lock));
    CHKERRQ(NEPNLEIGSGetInterpolation(nep,&tol,&its));
    CHKERRQ(PetscPrintf(PETSC_COMM_WORLD," Restart factor is %3.2f",(double)keep));
    if (lock) CHKERRQ(PetscPrintf(PETSC_COMM_WORLD," (locking activated)"));
    CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"\n Divided diferences with tol=%6.2g maxit=%" PetscInt_FMT "\n",(double)tol,its));
    CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"\n"));
  }

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
                    Display solution and clean up
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

  /* show detailed info unless -terse option is given by user */
  CHKERRQ(PetscOptionsHasName(NULL,NULL,"-terse",&terse));
  if (terse) CHKERRQ(NEPErrorView(nep,NEP_ERROR_RELATIVE,NULL));
  else {
    CHKERRQ(PetscViewerPushFormat(PETSC_VIEWER_STDOUT_WORLD,PETSC_VIEWER_ASCII_INFO_DETAIL));
    CHKERRQ(NEPConvergedReasonView(nep,PETSC_VIEWER_STDOUT_WORLD));
    CHKERRQ(NEPErrorView(nep,NEP_ERROR_RELATIVE,PETSC_VIEWER_STDOUT_WORLD));
    CHKERRQ(PetscViewerPopFormat(PETSC_VIEWER_STDOUT_WORLD));
  }
  CHKERRQ(NEPDestroy(&nep));
  if (split) {
    CHKERRQ(MatDestroy(&A[0]));
    CHKERRQ(MatDestroy(&A[1]));
    CHKERRQ(FNDestroy(&f[0]));
    CHKERRQ(FNDestroy(&f[1]));
  } else CHKERRQ(MatDestroy(&F));
  ierr = SlepcFinalize();
  return ierr;
}

/*
   FormFunction - Computes Function matrix  T(lambda)
*/
PetscErrorCode FormFunction(NEP nep,PetscScalar lambda,Mat fun,Mat B,void *ctx)
{
  MatCtx         *ctxF;

  PetscFunctionBeginUser;
  CHKERRQ(MatShellGetContext(fun,&ctxF));
  ctxF->t = PetscSqrtScalar(lambda);
  PetscFunctionReturn(0);
}

/*
   ComputeSingularities - Computes maxnp points (at most) in the complex plane where
   the function T(.) is not analytic.

   In this case, we discretize the singularity region (-inf,0)~(-10e+6,-10e-6)
*/
PetscErrorCode ComputeSingularities(NEP nep,PetscInt *maxnp,PetscScalar *xi,void *pt)
{
  PetscReal h;
  PetscInt  i;

  PetscFunctionBeginUser;
  h = 12.0/(*maxnp-1);
  xi[0] = -1e-6; xi[*maxnp-1] = -1e+6;
  for (i=1;i<*maxnp-1;i++) xi[i] = -PetscPowReal(10,-6+h*i);
  PetscFunctionReturn(0);
}

/* -------------------------------- A0 ----------------------------------- */

PetscErrorCode MatMult_A0(Mat A,Vec x,Vec y)
{
  PetscInt          i,n;
  PetscMPIInt       rank,size,next,prev;
  const PetscScalar *px;
  PetscScalar       *py,upper=0.0,lower=0.0;
  MPI_Comm          comm;

  PetscFunctionBeginUser;
  CHKERRQ(PetscObjectGetComm((PetscObject)A,&comm));
  CHKERRMPI(MPI_Comm_size(comm,&size));
  CHKERRMPI(MPI_Comm_rank(comm,&rank));
  next = rank==size-1? MPI_PROC_NULL: rank+1;
  prev = rank==0? MPI_PROC_NULL: rank-1;

  CHKERRQ(VecGetArrayRead(x,&px));
  CHKERRQ(VecGetArray(y,&py));
  CHKERRQ(VecGetLocalSize(x,&n));

  CHKERRMPI(MPI_Sendrecv(px,1,MPIU_SCALAR,prev,0,&lower,1,MPIU_SCALAR,next,0,comm,MPI_STATUS_IGNORE));
  CHKERRMPI(MPI_Sendrecv(px+n-1,1,MPIU_SCALAR,next,0,&upper,1,MPIU_SCALAR,prev,0,comm,MPI_STATUS_IGNORE));

  py[0] = upper-2.0*px[0]+px[1];
  for (i=1;i<n-1;i++) py[i] = px[i-1]-2.0*px[i]+px[i+1];
  py[n-1] = px[n-2]-2.0*px[n-1]+lower;
  CHKERRQ(VecRestoreArrayRead(x,&px));
  CHKERRQ(VecRestoreArray(y,&py));
  PetscFunctionReturn(0);
}

PetscErrorCode MatGetDiagonal_A0(Mat A,Vec diag)
{
  PetscFunctionBeginUser;
  CHKERRQ(VecSet(diag,-2.0));
  PetscFunctionReturn(0);
}

PetscErrorCode MatDuplicate_A0(Mat A,MatDuplicateOption op,Mat *B)
{
  PetscInt       m,n,M,N;
  MPI_Comm       comm;

  PetscFunctionBegin;
  CHKERRQ(MatGetSize(A,&M,&N));
  CHKERRQ(MatGetLocalSize(A,&m,&n));
  CHKERRQ(PetscObjectGetComm((PetscObject)A,&comm));
  CHKERRQ(MatCreateShell(comm,m,n,M,N,NULL,B));
  CHKERRQ(MatShellSetOperation(*B,MATOP_MULT,(void(*)(void))MatMult_A0));
  CHKERRQ(MatShellSetOperation(*B,MATOP_MULT_TRANSPOSE,(void(*)(void))MatMult_A0));
  CHKERRQ(MatShellSetOperation(*B,MATOP_GET_DIAGONAL,(void(*)(void))MatGetDiagonal_A0));
  CHKERRQ(MatShellSetOperation(*B,MATOP_DUPLICATE,(void(*)(void))MatDuplicate_A0));
  PetscFunctionReturn(0);
}

/* -------------------------------- A1 ----------------------------------- */

PetscErrorCode MatMult_A1(Mat A,Vec x,Vec y)
{
  PetscFunctionBeginUser;
  CHKERRQ(VecCopy(x,y));
  PetscFunctionReturn(0);
}

PetscErrorCode MatGetDiagonal_A1(Mat A,Vec diag)
{
  PetscFunctionBeginUser;
  CHKERRQ(VecSet(diag,1.0));
  PetscFunctionReturn(0);
}

PetscErrorCode MatDuplicate_A1(Mat A,MatDuplicateOption op,Mat *B)
{
  PetscInt       m,n,M,N;
  MPI_Comm       comm;

  PetscFunctionBegin;
  CHKERRQ(MatGetSize(A,&M,&N));
  CHKERRQ(MatGetLocalSize(A,&m,&n));
  CHKERRQ(PetscObjectGetComm((PetscObject)A,&comm));
  CHKERRQ(MatCreateShell(comm,m,n,M,N,NULL,B));
  CHKERRQ(MatShellSetOperation(*B,MATOP_MULT,(void(*)(void))MatMult_A1));
  CHKERRQ(MatShellSetOperation(*B,MATOP_MULT_TRANSPOSE,(void(*)(void))MatMult_A1));
  CHKERRQ(MatShellSetOperation(*B,MATOP_GET_DIAGONAL,(void(*)(void))MatGetDiagonal_A1));
  CHKERRQ(MatShellSetOperation(*B,MATOP_DUPLICATE,(void(*)(void))MatDuplicate_A1));
  PetscFunctionReturn(0);
}

/* -------------------------------- F ----------------------------------- */

PetscErrorCode MatMult_F(Mat A,Vec x,Vec y)
{
  PetscInt          i,n;
  PetscMPIInt       rank,size,next,prev;
  const PetscScalar *px;
  PetscScalar       *py,d,upper=0.0,lower=0.0;
  MatCtx            *ctx;
  MPI_Comm          comm;

  PetscFunctionBeginUser;
  CHKERRQ(PetscObjectGetComm((PetscObject)A,&comm));
  CHKERRMPI(MPI_Comm_size(comm,&size));
  CHKERRMPI(MPI_Comm_rank(comm,&rank));
  next = rank==size-1? MPI_PROC_NULL: rank+1;
  prev = rank==0? MPI_PROC_NULL: rank-1;

  CHKERRQ(MatShellGetContext(A,&ctx));
  CHKERRQ(VecGetArrayRead(x,&px));
  CHKERRQ(VecGetArray(y,&py));
  CHKERRQ(VecGetLocalSize(x,&n));

  CHKERRMPI(MPI_Sendrecv(px,1,MPIU_SCALAR,prev,0,&lower,1,MPIU_SCALAR,next,0,comm,MPI_STATUS_IGNORE));
  CHKERRMPI(MPI_Sendrecv(px+n-1,1,MPIU_SCALAR,next,0,&upper,1,MPIU_SCALAR,prev,0,comm,MPI_STATUS_IGNORE));

  d = -2.0+ctx->t;
  py[0] = upper+d*px[0]+px[1];
  for (i=1;i<n-1;i++) py[i] = px[i-1]+d*px[i]+px[i+1];
  py[n-1] = px[n-2]+d*px[n-1]+lower;
  CHKERRQ(VecRestoreArrayRead(x,&px));
  CHKERRQ(VecRestoreArray(y,&py));
  PetscFunctionReturn(0);
}

PetscErrorCode MatGetDiagonal_F(Mat A,Vec diag)
{
  MatCtx         *ctx;

  PetscFunctionBeginUser;
  CHKERRQ(MatShellGetContext(A,&ctx));
  CHKERRQ(VecSet(diag,-2.0+ctx->t));
  PetscFunctionReturn(0);
}

PetscErrorCode MatDuplicate_F(Mat A,MatDuplicateOption op,Mat *B)
{
  MatCtx         *actx,*bctx;
  PetscInt       m,n,M,N;
  MPI_Comm       comm;

  PetscFunctionBegin;
  CHKERRQ(MatShellGetContext(A,&actx));
  CHKERRQ(MatGetSize(A,&M,&N));
  CHKERRQ(MatGetLocalSize(A,&m,&n));
  CHKERRQ(PetscNew(&bctx));
  bctx->t = actx->t;
  CHKERRQ(PetscObjectGetComm((PetscObject)A,&comm));
  CHKERRQ(MatCreateShell(comm,m,n,M,N,(void*)bctx,B));
  CHKERRQ(MatShellSetOperation(*B,MATOP_MULT,(void(*)(void))MatMult_F));
  CHKERRQ(MatShellSetOperation(*B,MATOP_MULT_TRANSPOSE,(void(*)(void))MatMult_F));
  CHKERRQ(MatShellSetOperation(*B,MATOP_GET_DIAGONAL,(void(*)(void))MatGetDiagonal_F));
  CHKERRQ(MatShellSetOperation(*B,MATOP_DUPLICATE,(void(*)(void))MatDuplicate_F));
  CHKERRQ(MatShellSetOperation(*B,MATOP_DESTROY,(void(*)(void))MatDestroy_F));
  PetscFunctionReturn(0);
}

PetscErrorCode MatDestroy_F(Mat A)
{
  MatCtx         *ctx;

  PetscFunctionBegin;
  CHKERRQ(MatShellGetContext(A,&ctx));
  CHKERRQ(PetscFree(ctx));
  PetscFunctionReturn(0);
}

/*TEST

   testset:
      nsize: {{1 2}}
      args: -nep_nev 3 -nep_tol 1e-8 -terse
      filter: sed -e "s/[+-]0\.0*i//g"
      requires: !single
      test:
         suffix: 1
         args: -nep_nleigs_locking 0 -nep_nleigs_interpolation_degree 90 -nep_nleigs_interpolation_tol 1e-8 -nep_nleigs_restart 0.4
      test:
         suffix: 2
         args: -split 0

TEST*/
