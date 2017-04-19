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

static char help[] = "Illustrates the use of nonlinear inverse iteration for A(x)*x=lambda*B(x)*x.\n\n"
  "The problem is the same as in ex13.c. The command line options are:\n"
  "  -n <n>, where <n> = number of grid subdivisions in x dimension.\n"
  "  -m <m>, where <m> = number of grid subdivisions in y dimension.\n"
  "  -nulldim <k>, where <k> = dimension of the nullspace of B.\n\n";

#include <slepceps.h>

PetscErrorCode FormJacobianA(SNES,Vec,Mat,Mat,void*);
PetscErrorCode FormJacobianB(SNES,Vec,Mat,Mat,void*);
PetscErrorCode FormFunctionA(SNES,Vec,Vec,void*);
PetscErrorCode FormFunctionB(SNES,Vec,Vec,void*);

typedef struct {
  PetscInt n;         /* number of intervals in X direction */
  PetscInt m;         /* number of intervals in Y direction */
  PetscInt nulldim;   /* dimension of the nullspace of B */
  Mat      mat;       /* auxiliary matrix used in FormFunction */
} AppCtx;

int main(int argc,char **argv)
{
  Mat            A,B;         /* matrices */
  EPS            eps;         /* eigenproblem solver context */
  ST             st;          /* spectral transformation context */
  EPSType        type;
  PetscInt       N,n=10,m,nev,nulldim=0;
  PetscBool      flag,terse;
  AppCtx         user;
  PetscContainer container;
  PetscErrorCode ierr;

  ierr = SlepcInitialize(&argc,&argv,(char*)0,help);if (ierr) return ierr;

  ierr = PetscOptionsGetInt(NULL,NULL,"-n",&n,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetInt(NULL,NULL,"-m",&m,&flag);CHKERRQ(ierr);
  if (!flag) m=n;
  N = n*m;
  ierr = PetscOptionsGetInt(NULL,NULL,"-nulldim",&nulldim,NULL);CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD,"\nNonlinear Eigenproblem, N=%D (%Dx%D grid), null(B)=%D\n\n",N,n,m,nulldim);CHKERRQ(ierr);
  user.n = n; user.m = m; user.nulldim = nulldim;

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Create the matrices that define the eigensystem, A(x)*x=k*B(x)*x
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

  ierr = MatCreate(PETSC_COMM_WORLD,&A);CHKERRQ(ierr);
  ierr = MatSetSizes(A,PETSC_DECIDE,PETSC_DECIDE,N,N);CHKERRQ(ierr);
  ierr = MatSetFromOptions(A);CHKERRQ(ierr);
  ierr = MatSetUp(A);CHKERRQ(ierr);

  ierr = MatCreate(PETSC_COMM_WORLD,&B);CHKERRQ(ierr);
  ierr = MatSetSizes(B,PETSC_DECIDE,PETSC_DECIDE,N,N);CHKERRQ(ierr);
  ierr = MatSetFromOptions(B);CHKERRQ(ierr);
  ierr = MatSetUp(B);CHKERRQ(ierr);

  ierr = MatCreate(PETSC_COMM_WORLD,&user.mat);CHKERRQ(ierr);
  ierr = MatSetSizes(user.mat,PETSC_DECIDE,PETSC_DECIDE,N,N);CHKERRQ(ierr);
  ierr = MatSetFromOptions(user.mat);CHKERRQ(ierr);
  ierr = MatSetUp(user.mat);CHKERRQ(ierr);

  ierr = FormJacobianA(NULL,NULL,A,A,&user);CHKERRQ(ierr);
  ierr = FormJacobianB(NULL,NULL,B,B,&user);CHKERRQ(ierr);

  /*
     Compose callback functions and context that will be needed by the solver
  */
  ierr = PetscObjectComposeFunction((PetscObject)A,"formFunction",FormFunctionA);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)A,"formJacobian",FormJacobianA);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)B,"formFunction",FormFunctionB);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)B,"formJacobian",FormJacobianB);CHKERRQ(ierr);
  ierr = PetscContainerCreate(PETSC_COMM_WORLD,&container);CHKERRQ(ierr);
  ierr = PetscContainerSetPointer(container,&user);CHKERRQ(ierr);
  ierr = PetscObjectCompose((PetscObject)A,"formFunctionCtx",(PetscObject)container);CHKERRQ(ierr);
  ierr = PetscObjectCompose((PetscObject)A,"formJacobianCtx",(PetscObject)container);CHKERRQ(ierr);
  ierr = PetscObjectCompose((PetscObject)B,"formFunctionCtx",(PetscObject)container);CHKERRQ(ierr);
  ierr = PetscObjectCompose((PetscObject)B,"formJacobianCtx",(PetscObject)container);CHKERRQ(ierr);
  ierr = PetscContainerDestroy(&container);CHKERRQ(ierr);

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
                Create the eigensolver and set various options
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

  ierr = EPSCreate(PETSC_COMM_WORLD,&eps);CHKERRQ(ierr);
  ierr = EPSSetOperators(eps,A,B);CHKERRQ(ierr);
  ierr = EPSSetProblemType(eps,EPS_GHEP);CHKERRQ(ierr);
  /*
     Use nonlinear inverse iteration
  */
  ierr = EPSSetType(eps,EPSPOWER);CHKERRQ(ierr);
  ierr = EPSPowerSetNonlinear(eps,PETSC_TRUE);CHKERRQ(ierr);

  /*
     Nonlinear inverse iteration requires shift-and-invert with target=0
  */
  ierr = EPSSetTarget(eps,0.0);CHKERRQ(ierr);
  ierr = EPSSetWhichEigenpairs(eps,EPS_TARGET_MAGNITUDE);CHKERRQ(ierr);
  ierr = EPSGetST(eps,&st);CHKERRQ(ierr);
  ierr = STSetType(st,STSINVERT);CHKERRQ(ierr);

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
    ierr = PetscViewerPushFormat(PETSC_VIEWER_STDOUT_WORLD,PETSC_VIEWER_ASCII_INFO_DETAIL);CHKERRQ(ierr);
    ierr = EPSReasonView(eps,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
    ierr = EPSErrorView(eps,EPS_ERROR_RELATIVE,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
    ierr = PetscViewerPopFormat(PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
  }
  ierr = EPSDestroy(&eps);CHKERRQ(ierr);
  ierr = MatDestroy(&A);CHKERRQ(ierr);
  ierr = MatDestroy(&B);CHKERRQ(ierr);
  ierr = MatDestroy(&user.mat);CHKERRQ(ierr);
  ierr = SlepcFinalize();
  return ierr;
}

PetscErrorCode FormFunctionA(SNES snes,Vec x,Vec y,void* ctx)
{
  PetscErrorCode ierr;
  AppCtx         *user = (AppCtx*)ctx;

  PetscFunctionBegin;
  ierr = MatZeroEntries(user->mat);CHKERRQ(ierr);
  ierr = FormJacobianA(snes,x,user->mat,user->mat,user);CHKERRQ(ierr);
  ierr = MatMult(user->mat,x,y);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode FormJacobianA(SNES snes,Vec x,Mat J,Mat A,void* ctx)
{
  PetscErrorCode ierr;
  PetscInt       Istart,Iend,II,i,j;
  AppCtx         *user = (AppCtx*)ctx;

  PetscFunctionBegin;
  ierr = MatZeroEntries(A);CHKERRQ(ierr);
  ierr = MatGetOwnershipRange(A,&Istart,&Iend);CHKERRQ(ierr);
  for (II=Istart;II<Iend;II++) {
    i = II/user->n; j = II-i*user->n;
    if (i>0) { ierr = MatSetValue(A,II,II-user->n,-1.0,INSERT_VALUES);CHKERRQ(ierr); }
    if (i<user->m-1) { ierr = MatSetValue(A,II,II+user->n,-1.0,INSERT_VALUES);CHKERRQ(ierr); }
    if (j>0) { ierr = MatSetValue(A,II,II-1,-1.0,INSERT_VALUES);CHKERRQ(ierr); }
    if (j<user->n-1) { ierr = MatSetValue(A,II,II+1,-1.0,INSERT_VALUES);CHKERRQ(ierr); }
    ierr = MatSetValue(A,II,II,4.0,INSERT_VALUES);CHKERRQ(ierr);
  }
  ierr = MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  if (J!=A) {
   ierr = MatAssemblyBegin(J,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
   ierr = MatAssemblyEnd(J,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

PetscErrorCode FormFunctionB(SNES snes,Vec x,Vec y,void* ctx)
{
  PetscErrorCode ierr;
  AppCtx         *user = (AppCtx*)ctx;

  PetscFunctionBegin;
  ierr = MatZeroEntries(user->mat);CHKERRQ(ierr);
  ierr = FormJacobianB(snes,x,user->mat,user->mat,user);CHKERRQ(ierr);
  ierr = MatMult(user->mat,x,y);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode FormJacobianB(SNES snes,Vec x,Mat J,Mat B,void* ctx)
{
  PetscErrorCode ierr;
  PetscInt       Istart,Iend,II;
  AppCtx         *user = (AppCtx*)ctx;

  PetscFunctionBegin;
  ierr = MatZeroEntries(B);CHKERRQ(ierr);
  ierr = MatGetOwnershipRange(B,&Istart,&Iend);CHKERRQ(ierr);
  for (II=Istart;II<Iend;II++) {
    if (II>=user->nulldim) { ierr = MatSetValue(B,II,II,4.0,INSERT_VALUES);CHKERRQ(ierr); }
  }
  ierr = MatAssemblyBegin(B,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(B,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  if (J!=B) {
   ierr = MatAssemblyBegin(J,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
   ierr = MatAssemblyEnd(J,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

