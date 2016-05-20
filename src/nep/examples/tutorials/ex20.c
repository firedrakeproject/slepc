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

static char help[] = "Simple 1-D nonlinear eigenproblem.\n\n"
  "The command line options are:\n"
  "  -n <n>, where <n> = number of grid subdivisions.\n"
  "  -draw_sol, to draw the computed solution.\n\n";

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
PetscErrorCode CheckSolution(PetscScalar,Vec,PetscReal*,void*);
PetscErrorCode FixSign(Vec);

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
  Vec            x;               /* eigenvector */
  PetscScalar    lambda;          /* eigenvalue */
  Mat            F,J;             /* Function and Jacobian matrices */
  ApplicationCtx ctx;             /* user-defined context */
  NEPType        type;
  PetscInt       n=128,nev,i,its,maxit,nconv;
  PetscReal      re,im,tol,norm,error;
  PetscBool      draw_sol;
  PetscErrorCode ierr;

  SlepcInitialize(&argc,&argv,(char*)0,help);
  ierr = PetscOptionsGetInt(NULL,NULL,"-n",&n,NULL);CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD,"\n1-D Nonlinear Eigenproblem, n=%D\n\n",n);CHKERRQ(ierr);
  ctx.h = 1.0/(PetscReal)n;
  ctx.kappa = 1.0;
  ierr = PetscOptionsHasName(NULL,NULL,"-draw_sol",&draw_sol);CHKERRQ(ierr);

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Create nonlinear eigensolver context
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

  ierr = NEPCreate(PETSC_COMM_WORLD,&nep);CHKERRQ(ierr);

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Create matrix data structure; set Function evaluation routine
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

  ierr = MatCreate(PETSC_COMM_WORLD,&F);CHKERRQ(ierr);
  ierr = MatSetSizes(F,PETSC_DECIDE,PETSC_DECIDE,n,n);CHKERRQ(ierr);
  ierr = MatSetFromOptions(F);CHKERRQ(ierr);
  ierr = MatSeqAIJSetPreallocation(F,3,NULL);CHKERRQ(ierr);
  ierr = MatMPIAIJSetPreallocation(F,3,NULL,1,NULL);CHKERRQ(ierr);
  ierr = MatSetUp(F);CHKERRQ(ierr);

  /*
     Set Function matrix data structure and default Function evaluation
     routine
  */
  ierr = NEPSetFunction(nep,F,F,FormFunction,&ctx);CHKERRQ(ierr);

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Create matrix data structure; set Jacobian evaluation routine
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

  ierr = MatCreate(PETSC_COMM_WORLD,&J);CHKERRQ(ierr);
  ierr = MatSetSizes(J,PETSC_DECIDE,PETSC_DECIDE,n,n);CHKERRQ(ierr);
  ierr = MatSetFromOptions(J);CHKERRQ(ierr);
  ierr = MatSeqAIJSetPreallocation(J,3,NULL);CHKERRQ(ierr);
  ierr = MatMPIAIJSetPreallocation(F,3,NULL,1,NULL);CHKERRQ(ierr);
  ierr = MatSetUp(J);CHKERRQ(ierr);

  /*
     Set Jacobian matrix data structure and default Jacobian evaluation
     routine
  */
  ierr = NEPSetJacobian(nep,J,FormJacobian,&ctx);CHKERRQ(ierr);

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Customize nonlinear solver; set runtime options
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

  ierr = NEPSetTolerances(nep,1e-9,PETSC_DEFAULT);CHKERRQ(ierr);
  ierr = NEPSetDimensions(nep,1,PETSC_DEFAULT,PETSC_DEFAULT);CHKERRQ(ierr);

  /*
     Set solver parameters at runtime
  */
  ierr = NEPSetFromOptions(nep);CHKERRQ(ierr);

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
                      Initialize application
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

  /*
     Evaluate initial guess
  */
  ierr = MatCreateVecs(F,&x,NULL);CHKERRQ(ierr);
  ierr = FormInitialGuess(x);CHKERRQ(ierr);
  ierr = NEPSetInitialSpace(nep,1,&x);CHKERRQ(ierr);

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
                      Solve the eigensystem
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

  ierr = NEPSolve(nep);CHKERRQ(ierr);
  ierr = NEPGetIterationNumber(nep,&its);CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD," Number of NEP iterations = %D\n\n",its);CHKERRQ(ierr);

  /*
     Optional: Get some information from the solver and display it
  */
  ierr = NEPGetType(nep,&type);CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD," Solution method: %s\n",type);CHKERRQ(ierr);
  ierr = NEPGetDimensions(nep,&nev,NULL,NULL);CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD," Number of requested eigenvalues: %D\n",nev);CHKERRQ(ierr);
  ierr = NEPGetTolerances(nep,&tol,&maxit);CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD," Stopping condition: tol=%g, maxit=%D\n",(double)tol,maxit);CHKERRQ(ierr);

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
                    Display solution and clean up
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

  /*
     Get number of converged approximate eigenpairs
  */
  ierr = NEPGetConverged(nep,&nconv);CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD," Number of converged approximate eigenpairs: %D\n\n",nconv);CHKERRQ(ierr);

  if (nconv>0) {
    /*
       Display eigenvalues and relative errors
    */
    ierr = PetscPrintf(PETSC_COMM_WORLD,
         "           k              ||T(k)x||           error\n"
         "   ----------------- ------------------ ------------------\n");CHKERRQ(ierr);
    for (i=0;i<nconv;i++) {
      /*
        Get converged eigenpairs (in this example they are always real)
      */
      ierr = NEPGetEigenpair(nep,i,&lambda,NULL,x,NULL);CHKERRQ(ierr);
      ierr = FixSign(x);CHKERRQ(ierr);
      /*
         Compute residual norm and error
      */
      ierr = NEPComputeError(nep,i,NEP_ERROR_RELATIVE,&norm);CHKERRQ(ierr);
      ierr = CheckSolution(lambda,x,&error,&ctx);CHKERRQ(ierr);

#if defined(PETSC_USE_COMPLEX)
      re = PetscRealPart(lambda);
      im = PetscImaginaryPart(lambda);
#else
      re = lambda;
      im = 0.0;
#endif
      if (im!=0.0) {
        ierr = PetscPrintf(PETSC_COMM_WORLD," %9f%+9fi %12g     %12g\n",(double)re,(double)im,(double)norm,(double)error);CHKERRQ(ierr);
      } else {
        ierr = PetscPrintf(PETSC_COMM_WORLD,"   %12f         %12g     %12g\n",(double)re,(double)norm,(double)error);CHKERRQ(ierr);
      }
      if (draw_sol) {
        ierr = PetscViewerDrawSetPause(PETSC_VIEWER_DRAW_WORLD,-1);CHKERRQ(ierr);
        ierr = VecView(x,PETSC_VIEWER_DRAW_WORLD);CHKERRQ(ierr);
      }
    }
    ierr = PetscPrintf(PETSC_COMM_WORLD,"\n");CHKERRQ(ierr);
  }

  ierr = NEPDestroy(&nep);CHKERRQ(ierr);
  ierr = MatDestroy(&F);CHKERRQ(ierr);
  ierr = MatDestroy(&J);CHKERRQ(ierr);
  ierr = VecDestroy(&x);CHKERRQ(ierr);
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
.  lambda - the scalar argument
.  ctx    - optional user-defined context, as set by NEPSetFunction()

   Output Parameters:
.  fun - Function matrix
.  B   - optionally different preconditioning matrix
*/
PetscErrorCode FormFunction(NEP nep,PetscScalar lambda,Mat fun,Mat B,void *ctx)
{
  PetscErrorCode ierr;
  ApplicationCtx *user = (ApplicationCtx*)ctx;
  PetscScalar    A[3],c,d;
  PetscReal      h;
  PetscInt       i,n,j[3],Istart,Iend;
  PetscBool      FirstBlock=PETSC_FALSE,LastBlock=PETSC_FALSE;

  PetscFunctionBeginUser;
  /*
     Compute Function entries and insert into matrix
  */
  ierr = MatGetSize(fun,&n,NULL);CHKERRQ(ierr);
  ierr = MatGetOwnershipRange(fun,&Istart,&Iend);CHKERRQ(ierr);
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
    ierr = MatSetValues(fun,1,&i,3,j,A,INSERT_VALUES);CHKERRQ(ierr);
  }

  /*
     Boundary points
  */
  if (FirstBlock) {
    i = 0;
    j[0] = 0; j[1] = 1;
    A[0] = 2.0*(d-lambda*h/3.0); A[1] = -d-lambda*h/6.0;
    ierr = MatSetValues(fun,1,&i,2,j,A,INSERT_VALUES);CHKERRQ(ierr);
  }

  if (LastBlock) {
    i = n-1;
    j[0] = n-2; j[1] = n-1;
    A[0] = -d-lambda*h/6.0; A[1] = d-lambda*h/3.0+lambda*c;
    ierr = MatSetValues(fun,1,&i,2,j,A,INSERT_VALUES);CHKERRQ(ierr);
  }

  /*
     Assemble matrix
  */
  ierr = MatAssemblyBegin(B,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(B,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  if (fun != B) {
    ierr = MatAssemblyBegin(fun,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
    ierr = MatAssemblyEnd(fun,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

/* ------------------------------------------------------------------- */
#undef __FUNCT__
#define __FUNCT__ "FormJacobian"
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
  PetscErrorCode ierr;
  ApplicationCtx *user = (ApplicationCtx*)ctx;
  PetscScalar    A[3],c;
  PetscReal      h;
  PetscInt       i,n,j[3],Istart,Iend;
  PetscBool      FirstBlock=PETSC_FALSE,LastBlock=PETSC_FALSE;

  PetscFunctionBeginUser;
  /*
     Compute Jacobian entries and insert into matrix
  */
  ierr = MatGetSize(jac,&n,NULL);CHKERRQ(ierr);
  ierr = MatGetOwnershipRange(jac,&Istart,&Iend);CHKERRQ(ierr);
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
    ierr = MatSetValues(jac,1,&i,3,j,A,INSERT_VALUES);CHKERRQ(ierr);
  }

  /*
     Boundary points
  */
  if (FirstBlock) {
    i = 0;
    j[0] = 0; j[1] = 1;
    A[0] = -2.0*h/3.0; A[1] = -h/6.0;
    ierr = MatSetValues(jac,1,&i,2,j,A,INSERT_VALUES);CHKERRQ(ierr);
  }

  if (LastBlock) {
    i = n-1;
    j[0] = n-2; j[1] = n-1;
    A[0] = -h/6.0; A[1] = -h/3.0-c*c;
    ierr = MatSetValues(jac,1,&i,2,j,A,INSERT_VALUES);CHKERRQ(ierr);
  }

  /*
     Assemble matrix
  */
  ierr = MatAssemblyBegin(jac,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(jac,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/* ------------------------------------------------------------------- */
#undef __FUNCT__
#define __FUNCT__ "CheckSolution"
/*
   CheckSolution - Given a computed solution (lambda,x) check if it
   satisfies the analytic solution.

   Input Parameters:
+  lambda - the computed eigenvalue
-  y      - the computed eigenvector

   Output Parameter:
.  error - norm of difference between the computed and exact eigenvector
*/
PetscErrorCode CheckSolution(PetscScalar lambda,Vec y,PetscReal *error,void *ctx)
{
  PetscErrorCode ierr;
  PetscScalar    nu,*uu;
  PetscInt       i,n,Istart,Iend;
  PetscReal      x;
  Vec            u;
  ApplicationCtx *user = (ApplicationCtx*)ctx;

  PetscFunctionBeginUser;
  nu = PetscSqrtScalar(lambda);
  ierr = VecDuplicate(y,&u);CHKERRQ(ierr);
  ierr = VecGetSize(u,&n);CHKERRQ(ierr);
  ierr = VecGetOwnershipRange(y,&Istart,&Iend);CHKERRQ(ierr);
  ierr = VecGetArray(u,&uu);CHKERRQ(ierr);
  for (i=Istart;i<Iend;i++) {
    x = (i+1)*user->h;
    uu[i-Istart] = PetscSinReal(nu*x);
  }
  ierr = VecRestoreArray(u,&uu);CHKERRQ(ierr);
  ierr = VecNormalize(u,NULL);CHKERRQ(ierr);
  ierr = VecAXPY(u,-1.0,y);CHKERRQ(ierr);
  ierr = VecNorm(u,NORM_2,error);CHKERRQ(ierr);
  ierr = VecDestroy(&u);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/* ------------------------------------------------------------------- */
#undef __FUNCT__
#define __FUNCT__ "FixSign"
/*
   FixSign - Force the eigenfunction to be real and positive, since
   some eigensolvers may return the eigenvector multiplied by a
   complex number of modulus one.

   Input/Output Parameter:
.  x - the computed vector
*/
PetscErrorCode FixSign(Vec x)
{
  PetscErrorCode    ierr;
  PetscMPIInt       rank;
  PetscScalar       sign;
  const PetscScalar *xx;

  PetscFunctionBeginUser;
  ierr = MPI_Comm_rank(PETSC_COMM_WORLD,&rank);CHKERRQ(ierr);
  if (!rank) {
    ierr = VecGetArrayRead(x,&xx);CHKERRQ(ierr);
    sign = *xx/PetscAbsScalar(*xx);
    ierr = VecRestoreArrayRead(x,&xx);CHKERRQ(ierr);
  }
  ierr = MPI_Bcast(&sign,1,MPIU_SCALAR,0,PETSC_COMM_WORLD);CHKERRQ(ierr);
  ierr = VecScale(x,1.0/sign);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

