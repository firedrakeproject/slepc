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
PetscErrorCode FormInitialGuess(Vec);
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
PetscErrorCode MatDestroy_Jac(Mat);

typedef struct {
  PetscScalar lambda,kappa;
  PetscReal   h;
} MatCtx;

/*
   User-defined application context
*/
typedef struct {
  PetscScalar kappa;   /* ratio between stiffness of spring and attached mass */
  PetscReal   h;       /* mesh spacing */
} ApplicationCtx;

#undef __FUNCT__
#define __FUNCT__ "main"
int main(int argc,char **argv)
{
  NEP            nep;             /* nonlinear eigensolver context */
  Mat            F,J;             /* Function and Jacobian matrices */
  ApplicationCtx ctx;             /* user-defined context */
  MatCtx         *ctxF,*ctxJ;     /* contexts for shell matrices */
  NEPType        type;
  PetscInt       n=128,nev;
  KSP            ksp;
  PC             pc;
  PetscMPIInt    size;
  PetscBool      terse;
  PetscErrorCode ierr;

  SlepcInitialize(&argc,&argv,(char*)0,help);
  ierr = MPI_Comm_size(PETSC_COMM_WORLD,&size);CHKERRQ(ierr);
  if (size != 1) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,"This is a uniprocessor example only!");
  ierr = PetscOptionsGetInt(NULL,NULL,"-n",&n,NULL);CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD,"\n1-D Nonlinear Eigenproblem, n=%D\n\n",n);CHKERRQ(ierr);
  ctx.h = 1.0/(PetscReal)n;
  ctx.kappa = 1.0;

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Create nonlinear eigensolver context
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

  ierr = NEPCreate(PETSC_COMM_WORLD,&nep);CHKERRQ(ierr);

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Create matrix data structure; set Function evaluation routine
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

  ierr = PetscNew(&ctxF);CHKERRQ(ierr);
  ctxF->h = ctx.h;
  ctxF->kappa = ctx.kappa;

  ierr = MatCreateShell(PETSC_COMM_WORLD,n,n,n,n,(void*)ctxF,&F);CHKERRQ(ierr);
  ierr = MatShellSetOperation(F,MATOP_MULT,(void(*)())MatMult_Fun);CHKERRQ(ierr);
  ierr = MatShellSetOperation(F,MATOP_GET_DIAGONAL,(void(*)())MatGetDiagonal_Fun);CHKERRQ(ierr);
  ierr = MatShellSetOperation(F,MATOP_DESTROY,(void(*)())MatDestroy_Fun);CHKERRQ(ierr);
  ierr = MatShellSetOperation(F,MATOP_DUPLICATE,(void(*)())MatDuplicate_Fun);CHKERRQ(ierr);

  /*
     Set Function matrix data structure and default Function evaluation
     routine
  */
  ierr = NEPSetFunction(nep,F,F,FormFunction,NULL);CHKERRQ(ierr);

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Create matrix data structure; set Jacobian evaluation routine
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

  ierr = PetscNew(&ctxJ);CHKERRQ(ierr);
  ctxJ->h = ctx.h;
  ctxJ->kappa = ctx.kappa;

  ierr = MatCreateShell(PETSC_COMM_WORLD,n,n,n,n,(void*)ctxJ,&J);CHKERRQ(ierr);
  ierr = MatShellSetOperation(J,MATOP_MULT,(void(*)())MatMult_Jac);CHKERRQ(ierr);
  ierr = MatShellSetOperation(J,MATOP_DESTROY,(void(*)())MatDestroy_Jac);CHKERRQ(ierr);

  /*
     Set Jacobian matrix data structure and default Jacobian evaluation
     routine
  */
  ierr = NEPSetJacobian(nep,J,FormJacobian,NULL);CHKERRQ(ierr);

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Customize nonlinear solver; set runtime options
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

  ierr = NEPSetType(nep,NEPRII);CHKERRQ(ierr);
  ierr = NEPRIISetLagPreconditioner(nep,0);CHKERRQ(ierr);
  ierr = NEPRIIGetKSP(nep,&ksp);CHKERRQ(ierr);
  ierr = KSPSetType(ksp,KSPBCGS);CHKERRQ(ierr);
  ierr = KSPGetPC(ksp,&pc);CHKERRQ(ierr);
  ierr = PCSetType(pc,PCJACOBI);CHKERRQ(ierr);

  /*
     Set solver parameters at runtime
  */
  ierr = NEPSetFromOptions(nep);CHKERRQ(ierr);

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
                      Solve the eigensystem
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

  ierr = NEPSolve(nep);CHKERRQ(ierr);
  ierr = NEPGetType(nep,&type);CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD," Solution method: %s\n",type);CHKERRQ(ierr);
  ierr = NEPGetDimensions(nep,&nev,NULL,NULL);CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD," Number of requested eigenvalues: %D\n",nev);CHKERRQ(ierr);

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
                    Display solution and clean up
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

  /* show detailed info unless -terse option is given by user */
  ierr = PetscOptionsHasName(NULL,NULL,"-terse",&terse);CHKERRQ(ierr);
  if (terse) {
    ierr = NEPErrorView(nep,NEP_ERROR_RELATIVE,NULL);CHKERRQ(ierr);
  } else {
    ierr = PetscViewerPushFormat(PETSC_VIEWER_STDOUT_WORLD,PETSC_VIEWER_ASCII_INFO_DETAIL);CHKERRQ(ierr);
    ierr = NEPReasonView(nep,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
    ierr = NEPErrorView(nep,NEP_ERROR_RELATIVE,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
    ierr = PetscViewerPopFormat(PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
  }
  ierr = NEPDestroy(&nep);CHKERRQ(ierr);
  ierr = MatDestroy(&F);CHKERRQ(ierr);
  ierr = MatDestroy(&J);CHKERRQ(ierr);
  ierr = SlepcFinalize();
  return ierr;
}

/* ------------------------------------------------------------------- */
#undef __FUNCT__
#define __FUNCT__ "FormInitialGuess"
/*
   FormInitialGuess - Computes initial guess.

   Input/Output Parameter:
.  x - the solution vector
*/
PetscErrorCode FormInitialGuess(Vec x)
{
  PetscErrorCode ierr;

  PetscFunctionBeginUser;
  ierr = VecSet(x,1.0);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/* ------------------------------------------------------------------- */
#undef __FUNCT__
#define __FUNCT__ "FormFunction"
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
  PetscErrorCode ierr;
  MatCtx         *ctxF;

  PetscFunctionBeginUser;
  ierr = MatShellGetContext(fun,(void**)&ctxF);CHKERRQ(ierr);
  ctxF->lambda = lambda;
  PetscFunctionReturn(0);
}

/* ------------------------------------------------------------------- */
#undef __FUNCT__
#define __FUNCT__ "FormJacobian"
/*
   FormJacobian - Computes Jacobian matrix  T'(lambda)

   Input Parameters:
.  nep    - the NEP context
.  lambda - real part of the scalar argument
.  ctx    - optional user-defined context, as set by NEPSetJacobian()

   Output Parameters:
.  jac - Jacobian matrix
.  B   - optionally different preconditioning matrix
*/
PetscErrorCode FormJacobian(NEP nep,PetscScalar lambda,Mat jac,void *ctx)
{
  PetscErrorCode ierr;
  MatCtx         *ctxJ;

  PetscFunctionBeginUser;
  ierr = MatShellGetContext(jac,(void**)&ctxJ);CHKERRQ(ierr);
  ctxJ->lambda = lambda;
  PetscFunctionReturn(0);
}

/* ------------------------------------------------------------------- */
#undef __FUNCT__
#define __FUNCT__ "MatMult_Fun"
PetscErrorCode MatMult_Fun(Mat A,Vec x,Vec y)
{
  PetscErrorCode    ierr;
  MatCtx            *ctx;
  PetscInt          i,n;
  const PetscScalar *px;
  PetscScalar       *py,c,d,de,oe;
  PetscReal         h;

  PetscFunctionBeginUser;
  ierr = MatShellGetContext(A,(void**)&ctx);CHKERRQ(ierr);
  ierr = VecGetArrayRead(x,&px);CHKERRQ(ierr);
  ierr = VecGetArray(y,&py);CHKERRQ(ierr);

  ierr = VecGetSize(x,&n);CHKERRQ(ierr);
  h = ctx->h;
  c = ctx->kappa/(ctx->lambda-ctx->kappa);
  d = n;
  de = 2.0*(d-ctx->lambda*h/3.0);   /* diagonal entry */
  oe = -d-ctx->lambda*h/6.0;        /* offdiagonal entry */
  py[0] = de*px[0] + oe*px[1];
  for (i=1;i<n-1;i++) py[i] = oe*px[i-1] +de*px[i] + oe*px[i+1];
  de = d-ctx->lambda*h/3.0+ctx->lambda*c;   /* diagonal entry of last row */
  py[n-1] = oe*px[n-2] + de*px[n-1];

  ierr = VecRestoreArrayRead(x,&px);CHKERRQ(ierr);
  ierr = VecRestoreArray(y,&py);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/* ------------------------------------------------------------------- */
#undef __FUNCT__
#define __FUNCT__ "MatGetDiagonal_Fun"
PetscErrorCode MatGetDiagonal_Fun(Mat A,Vec diag)
{
  PetscErrorCode    ierr;
  MatCtx            *ctx;
  PetscInt          n;
  PetscScalar       *pd,c,d;
  PetscReal         h;

  PetscFunctionBeginUser;
  ierr = MatShellGetContext(A,(void**)&ctx);CHKERRQ(ierr);
  ierr = VecGetSize(diag,&n);CHKERRQ(ierr);
  h = ctx->h;
  c = ctx->kappa/(ctx->lambda-ctx->kappa);
  d = n;
  ierr = VecSet(diag,2.0*(d-ctx->lambda*h/3.0));CHKERRQ(ierr);
  ierr = VecGetArray(diag,&pd);CHKERRQ(ierr);
  pd[n-1] = d-ctx->lambda*h/3.0+ctx->lambda*c;
  ierr = VecRestoreArray(diag,&pd);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/* ------------------------------------------------------------------- */
#undef __FUNCT__
#define __FUNCT__ "MatDestroy_Fun"
PetscErrorCode MatDestroy_Fun(Mat A)
{
  MatCtx         *ctx;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = MatShellGetContext(A,(void**)&ctx);CHKERRQ(ierr);
  ierr = PetscFree(ctx);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/* ------------------------------------------------------------------- */
#undef __FUNCT__
#define __FUNCT__ "MatDuplicate_Fun"
PetscErrorCode MatDuplicate_Fun(Mat A,MatDuplicateOption op,Mat *B)
{
  MatCtx         *actx,*bctx;
  PetscInt       n;
  MPI_Comm       comm;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = MatShellGetContext(A,(void**)&actx);CHKERRQ(ierr);
  ierr = MatGetSize(A,&n,NULL);CHKERRQ(ierr);

  ierr = PetscNew(&bctx);CHKERRQ(ierr);
  bctx->h      = actx->h;
  bctx->kappa  = actx->kappa;
  bctx->lambda = actx->lambda;

  ierr = PetscObjectGetComm((PetscObject)A,&comm);CHKERRQ(ierr);
  ierr = MatCreateShell(comm,n,n,n,n,(void*)bctx,B);CHKERRQ(ierr);
  ierr = MatShellSetOperation(*B,MATOP_MULT,(void(*)())MatMult_Fun);CHKERRQ(ierr);
  ierr = MatShellSetOperation(*B,MATOP_GET_DIAGONAL,(void(*)())MatGetDiagonal_Fun);CHKERRQ(ierr);
  ierr = MatShellSetOperation(*B,MATOP_DESTROY,(void(*)())MatDestroy_Fun);CHKERRQ(ierr);
  ierr = MatShellSetOperation(*B,MATOP_DUPLICATE,(void(*)())MatDuplicate_Fun);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/* ------------------------------------------------------------------- */
#undef __FUNCT__
#define __FUNCT__ "MatMult_Jac"
PetscErrorCode MatMult_Jac(Mat A,Vec x,Vec y)
{
  PetscErrorCode    ierr;
  MatCtx            *ctx;
  PetscInt          i,n;
  const PetscScalar *px;
  PetscScalar       *py,c,de,oe;
  PetscReal         h;

  PetscFunctionBeginUser;
  ierr = MatShellGetContext(A,(void**)&ctx);CHKERRQ(ierr);
  ierr = VecGetArrayRead(x,&px);CHKERRQ(ierr);
  ierr = VecGetArray(y,&py);CHKERRQ(ierr);

  ierr = VecGetSize(x,&n);CHKERRQ(ierr);
  h = ctx->h;
  c = ctx->kappa/(ctx->lambda-ctx->kappa);
  de = -2.0*h/3.0;    /* diagonal entry */
  oe = -h/6.0;        /* offdiagonal entry */
  py[0] = de*px[0] + oe*px[1];
  for (i=1;i<n-1;i++) py[i] = oe*px[i-1] +de*px[i] + oe*px[i+1];
  de = -h/3.0-c*c;    /* diagonal entry of last row */
  py[n-1] = oe*px[n-2] + de*px[n-1];

  ierr = VecRestoreArrayRead(x,&px);CHKERRQ(ierr);
  ierr = VecRestoreArray(y,&py);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/* ------------------------------------------------------------------- */
#undef __FUNCT__
#define __FUNCT__ "MatDestroy_Jac"
PetscErrorCode MatDestroy_Jac(Mat A)
{
  MatCtx         *ctx;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = MatShellGetContext(A,(void**)&ctx);CHKERRQ(ierr);
  ierr = PetscFree(ctx);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

