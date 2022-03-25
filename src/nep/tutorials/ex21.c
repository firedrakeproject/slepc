/*
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
   SLEPc - Scalable Library for Eigenvalue Problem Computations
   Copyright (c) 2002-2021, Universitat Politecnica de Valencia, Spain

   This file is part of SLEPc.
   SLEPc is distributed under a 2-clause BSD license (see LICENSE).
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
*/

static char help[] = "Simple 1-D nonlinear eigenproblem (matrix-free version, sequential).\n\n"
  "The command line options are:\n"
  "  -n <n>, where <n> = number of grid subdivisions\n\n";

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
   Matrix operations and context
*/
PetscErrorCode MatMult_Fun(Mat,Vec,Vec);
PetscErrorCode MatGetDiagonal_Fun(Mat,Vec);
PetscErrorCode MatDestroy_Fun(Mat);
PetscErrorCode MatDuplicate_Fun(Mat,MatDuplicateOption,Mat*);
PetscErrorCode MatMult_Jac(Mat,Vec,Vec);
PetscErrorCode MatGetDiagonal_Jac(Mat,Vec);
PetscErrorCode MatDestroy_Jac(Mat);

typedef struct {
  PetscScalar lambda,kappa;
  PetscReal   h;
  PetscMPIInt next,prev;
} MatCtx;

/*
   User-defined application context
*/
typedef struct {
  PetscScalar kappa;   /* ratio between stiffness of spring and attached mass */
  PetscReal   h;       /* mesh spacing */
} ApplicationCtx;

int main(int argc,char **argv)
{
  NEP            nep;             /* nonlinear eigensolver context */
  Mat            F,J;             /* Function and Jacobian matrices */
  ApplicationCtx ctx;             /* user-defined context */
  MatCtx         *ctxF,*ctxJ;     /* contexts for shell matrices */
  PetscInt       n=128,nev;
  KSP            ksp;
  PC             pc;
  PetscMPIInt    rank,size;
  PetscBool      terse;

  CHKERRQ(SlepcInitialize(&argc,&argv,(char*)0,help));
  CHKERRMPI(MPI_Comm_size(PETSC_COMM_WORLD,&size));
  CHKERRMPI(MPI_Comm_rank(PETSC_COMM_WORLD,&rank));
  CHKERRQ(PetscOptionsGetInt(NULL,NULL,"-n",&n,NULL));
  CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"\n1-D Nonlinear Eigenproblem, n=%" PetscInt_FMT "\n\n",n));
  ctx.h = 1.0/(PetscReal)n;
  ctx.kappa = 1.0;

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Create nonlinear eigensolver context
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

  CHKERRQ(NEPCreate(PETSC_COMM_WORLD,&nep));

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Create matrix data structure; set Function evaluation routine
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

  CHKERRQ(PetscNew(&ctxF));
  ctxF->h = ctx.h;
  ctxF->kappa = ctx.kappa;
  ctxF->next = rank==size-1? MPI_PROC_NULL: rank+1;
  ctxF->prev = rank==0? MPI_PROC_NULL: rank-1;

  CHKERRQ(MatCreateShell(PETSC_COMM_WORLD,PETSC_DECIDE,PETSC_DECIDE,n,n,(void*)ctxF,&F));
  CHKERRQ(MatShellSetOperation(F,MATOP_MULT,(void(*)(void))MatMult_Fun));
  CHKERRQ(MatShellSetOperation(F,MATOP_GET_DIAGONAL,(void(*)(void))MatGetDiagonal_Fun));
  CHKERRQ(MatShellSetOperation(F,MATOP_DESTROY,(void(*)(void))MatDestroy_Fun));
  CHKERRQ(MatShellSetOperation(F,MATOP_DUPLICATE,(void(*)(void))MatDuplicate_Fun));

  /*
     Set Function matrix data structure and default Function evaluation
     routine
  */
  CHKERRQ(NEPSetFunction(nep,F,F,FormFunction,NULL));

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Create matrix data structure; set Jacobian evaluation routine
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

  CHKERRQ(PetscNew(&ctxJ));
  ctxJ->h = ctx.h;
  ctxJ->kappa = ctx.kappa;
  ctxJ->next = rank==size-1? MPI_PROC_NULL: rank+1;
  ctxJ->prev = rank==0? MPI_PROC_NULL: rank-1;

  CHKERRQ(MatCreateShell(PETSC_COMM_WORLD,PETSC_DECIDE,PETSC_DECIDE,n,n,(void*)ctxJ,&J));
  CHKERRQ(MatShellSetOperation(J,MATOP_MULT,(void(*)(void))MatMult_Jac));
  CHKERRQ(MatShellSetOperation(J,MATOP_GET_DIAGONAL,(void(*)(void))MatGetDiagonal_Jac));
  CHKERRQ(MatShellSetOperation(J,MATOP_DESTROY,(void(*)(void))MatDestroy_Jac));

  /*
     Set Jacobian matrix data structure and default Jacobian evaluation
     routine
  */
  CHKERRQ(NEPSetJacobian(nep,J,FormJacobian,NULL));

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Customize nonlinear solver; set runtime options
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

  CHKERRQ(NEPSetType(nep,NEPRII));
  CHKERRQ(NEPRIISetLagPreconditioner(nep,0));
  CHKERRQ(NEPRIIGetKSP(nep,&ksp));
  CHKERRQ(KSPSetType(ksp,KSPBCGS));
  CHKERRQ(KSPGetPC(ksp,&pc));
  CHKERRQ(PCSetType(pc,PCJACOBI));

  /*
     Set solver parameters at runtime
  */
  CHKERRQ(NEPSetFromOptions(nep));

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
                      Solve the eigensystem
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

  CHKERRQ(NEPSolve(nep));
  CHKERRQ(NEPGetDimensions(nep,&nev,NULL,NULL));
  CHKERRQ(PetscPrintf(PETSC_COMM_WORLD," Number of requested eigenvalues: %" PetscInt_FMT "\n",nev));

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
  CHKERRQ(MatDestroy(&F));
  CHKERRQ(MatDestroy(&J));
  CHKERRQ(SlepcFinalize());
  return 0;
}

/* ------------------------------------------------------------------- */
/*
   FormFunction - Computes Function matrix  T(lambda)

   Input Parameters:
.  nep    - the NEP context
.  lambda - real part of the scalar argument
.  ctx    - optional user-defined context, as set by NEPSetFunction()

   Output Parameters:
.  fun - Function matrix
.  B   - optionally different preconditioning matrix
*/
PetscErrorCode FormFunction(NEP nep,PetscScalar lambda,Mat fun,Mat B,void *ctx)
{
  MatCtx         *ctxF;

  PetscFunctionBeginUser;
  CHKERRQ(MatShellGetContext(fun,&ctxF));
  ctxF->lambda = lambda;
  PetscFunctionReturn(0);
}

/* ------------------------------------------------------------------- */
/*
   FormJacobian - Computes Jacobian matrix  T'(lambda)

   Input Parameters:
.  nep    - the NEP context
.  lambda - real part of the scalar argument
.  ctx    - optional user-defined context, as set by NEPSetJacobian()

   Output Parameters:
.  jac - Jacobian matrix
*/
PetscErrorCode FormJacobian(NEP nep,PetscScalar lambda,Mat jac,void *ctx)
{
  MatCtx         *ctxJ;

  PetscFunctionBeginUser;
  CHKERRQ(MatShellGetContext(jac,&ctxJ));
  ctxJ->lambda = lambda;
  PetscFunctionReturn(0);
}

/* ------------------------------------------------------------------- */
PetscErrorCode MatMult_Fun(Mat A,Vec x,Vec y)
{
  MatCtx            *ctx;
  PetscInt          i,n,N;
  const PetscScalar *px;
  PetscScalar       *py,c,d,de,oe,upper=0.0,lower=0.0;
  PetscReal         h;
  MPI_Comm          comm;

  PetscFunctionBeginUser;
  CHKERRQ(MatShellGetContext(A,&ctx));
  CHKERRQ(VecGetArrayRead(x,&px));
  CHKERRQ(VecGetArray(y,&py));
  CHKERRQ(VecGetSize(x,&N));
  CHKERRQ(VecGetLocalSize(x,&n));

  CHKERRQ(PetscObjectGetComm((PetscObject)A,&comm));
  CHKERRMPI(MPI_Sendrecv(px,1,MPIU_SCALAR,ctx->prev,0,&lower,1,MPIU_SCALAR,ctx->next,0,comm,MPI_STATUS_IGNORE));
  CHKERRMPI(MPI_Sendrecv(px+n-1,1,MPIU_SCALAR,ctx->next,0,&upper,1,MPIU_SCALAR,ctx->prev,0,comm,MPI_STATUS_IGNORE));

  h = ctx->h;
  c = ctx->kappa/(ctx->lambda-ctx->kappa);
  d = N;
  de = 2.0*(d-ctx->lambda*h/3.0);   /* diagonal entry */
  oe = -d-ctx->lambda*h/6.0;        /* offdiagonal entry */
  py[0] = oe*upper + de*px[0] + oe*px[1];
  for (i=1;i<n-1;i++) py[i] = oe*px[i-1] +de*px[i] + oe*px[i+1];
  if (ctx->next==MPI_PROC_NULL) de = d-ctx->lambda*h/3.0+ctx->lambda*c;   /* diagonal entry of last row */
  py[n-1] = oe*px[n-2] + de*px[n-1] + oe*lower;

  CHKERRQ(VecRestoreArrayRead(x,&px));
  CHKERRQ(VecRestoreArray(y,&py));
  PetscFunctionReturn(0);
}

/* ------------------------------------------------------------------- */
PetscErrorCode MatGetDiagonal_Fun(Mat A,Vec diag)
{
  MatCtx         *ctx;
  PetscInt       n,N;
  PetscScalar    *pd,c,d;
  PetscReal      h;

  PetscFunctionBeginUser;
  CHKERRQ(MatShellGetContext(A,&ctx));
  CHKERRQ(VecGetSize(diag,&N));
  CHKERRQ(VecGetLocalSize(diag,&n));
  h = ctx->h;
  c = ctx->kappa/(ctx->lambda-ctx->kappa);
  d = N;
  CHKERRQ(VecSet(diag,2.0*(d-ctx->lambda*h/3.0)));
  CHKERRQ(VecGetArray(diag,&pd));
  pd[n-1] = d-ctx->lambda*h/3.0+ctx->lambda*c;
  CHKERRQ(VecRestoreArray(diag,&pd));
  PetscFunctionReturn(0);
}

/* ------------------------------------------------------------------- */
PetscErrorCode MatDestroy_Fun(Mat A)
{
  MatCtx         *ctx;

  PetscFunctionBegin;
  CHKERRQ(MatShellGetContext(A,&ctx));
  CHKERRQ(PetscFree(ctx));
  PetscFunctionReturn(0);
}

/* ------------------------------------------------------------------- */
PetscErrorCode MatDuplicate_Fun(Mat A,MatDuplicateOption op,Mat *B)
{
  MatCtx         *actx,*bctx;
  PetscInt       m,n,M,N;
  MPI_Comm       comm;

  PetscFunctionBegin;
  CHKERRQ(MatShellGetContext(A,&actx));
  CHKERRQ(MatGetSize(A,&M,&N));
  CHKERRQ(MatGetLocalSize(A,&m,&n));

  CHKERRQ(PetscNew(&bctx));
  bctx->h      = actx->h;
  bctx->kappa  = actx->kappa;
  bctx->lambda = actx->lambda;
  bctx->next   = actx->next;
  bctx->prev   = actx->prev;

  CHKERRQ(PetscObjectGetComm((PetscObject)A,&comm));
  CHKERRQ(MatCreateShell(comm,m,n,M,N,(void*)bctx,B));
  CHKERRQ(MatShellSetOperation(*B,MATOP_MULT,(void(*)(void))MatMult_Fun));
  CHKERRQ(MatShellSetOperation(*B,MATOP_GET_DIAGONAL,(void(*)(void))MatGetDiagonal_Fun));
  CHKERRQ(MatShellSetOperation(*B,MATOP_DESTROY,(void(*)(void))MatDestroy_Fun));
  CHKERRQ(MatShellSetOperation(*B,MATOP_DUPLICATE,(void(*)(void))MatDuplicate_Fun));
  PetscFunctionReturn(0);
}

/* ------------------------------------------------------------------- */
PetscErrorCode MatMult_Jac(Mat A,Vec x,Vec y)
{
  MatCtx            *ctx;
  PetscInt          i,n;
  const PetscScalar *px;
  PetscScalar       *py,c,de,oe,upper=0.0,lower=0.0;
  PetscReal         h;
  MPI_Comm          comm;

  PetscFunctionBeginUser;
  CHKERRQ(MatShellGetContext(A,&ctx));
  CHKERRQ(VecGetArrayRead(x,&px));
  CHKERRQ(VecGetArray(y,&py));
  CHKERRQ(VecGetLocalSize(x,&n));

  CHKERRQ(PetscObjectGetComm((PetscObject)A,&comm));
  CHKERRMPI(MPI_Sendrecv(px,1,MPIU_SCALAR,ctx->prev,0,&lower,1,MPIU_SCALAR,ctx->next,0,comm,MPI_STATUS_IGNORE));
  CHKERRMPI(MPI_Sendrecv(px+n-1,1,MPIU_SCALAR,ctx->next,0,&upper,1,MPIU_SCALAR,ctx->prev,0,comm,MPI_STATUS_IGNORE));

  h = ctx->h;
  c = ctx->kappa/(ctx->lambda-ctx->kappa);
  de = -2.0*h/3.0;    /* diagonal entry */
  oe = -h/6.0;        /* offdiagonal entry */
  py[0] = oe*upper + de*px[0] + oe*px[1];
  for (i=1;i<n-1;i++) py[i] = oe*px[i-1] +de*px[i] + oe*px[i+1];
  if (ctx->next==MPI_PROC_NULL) de = -h/3.0-c*c;    /* diagonal entry of last row */
  py[n-1] = oe*px[n-2] + de*px[n-1] + oe*lower;

  CHKERRQ(VecRestoreArrayRead(x,&px));
  CHKERRQ(VecRestoreArray(y,&py));
  PetscFunctionReturn(0);
}

/* ------------------------------------------------------------------- */
PetscErrorCode MatGetDiagonal_Jac(Mat A,Vec diag)
{
  MatCtx         *ctx;
  PetscInt       n;
  PetscScalar    *pd,c;
  PetscReal      h;

  PetscFunctionBeginUser;
  CHKERRQ(MatShellGetContext(A,&ctx));
  CHKERRQ(VecGetLocalSize(diag,&n));
  h = ctx->h;
  c = ctx->kappa/(ctx->lambda-ctx->kappa);
  CHKERRQ(VecSet(diag,-2.0*h/3.0));
  CHKERRQ(VecGetArray(diag,&pd));
  pd[n-1] = -h/3.0-c*c;
  CHKERRQ(VecRestoreArray(diag,&pd));
  PetscFunctionReturn(0);
}

/* ------------------------------------------------------------------- */
PetscErrorCode MatDestroy_Jac(Mat A)
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
      args: -terse
      requires: !single
      output_file: output/ex21_1.out
      filter: sed -e "s/[+-]0\.0*i//g" -e "s/+0i//g"
      test:
         suffix: 1_rii
         args: -nep_type rii -nep_target 4
      test:
         suffix: 1_slp
         args: -nep_type slp -nep_slp_pc_type jacobi -nep_slp_ksp_type bcgs -nep_target 10

TEST*/
