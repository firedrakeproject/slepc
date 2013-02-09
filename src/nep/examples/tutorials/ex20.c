/*
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
   SLEPc - Scalable Library for Eigenvalue Problem Computations
   Copyright (c) 2002-2012, Universitat Politecnica de Valencia, Spain

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

static char help[] = "Simple 1-D nonlinear eigenproblem, sequential.\n\n"
  "The command line options are:\n"
  "  -n <n>, where <n> = number of grid subdivisions.\n\n";

#include <slepcnep.h>

/*
   User-defined routines
*/
extern PetscErrorCode FormInitialGuess(Vec);
extern PetscErrorCode FormFunction(NEP,PetscScalar,PetscScalar,Mat*,Mat*,MatStructure*,void*);
extern PetscErrorCode FormJacobian(NEP,PetscScalar,PetscScalar,Mat*,Mat*,MatStructure*,void*);

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
  Vec            x,r,u;           /* vectors */
  Mat            F,J;             /* Function and Jacobian matrices */
  ApplicationCtx ctx;             /* user-defined context */
  NEPType        type;
  PetscInt       n=128,nev,i,its,maxit,maxf,nconv;
  PetscMPIInt    size;
  PetscScalar    kr,ki;
  PetscReal      re,im,abstol,rtol,stol,norm;
  PetscErrorCode ierr;

  SlepcInitialize(&argc,&argv,(char*)0,help);
  ierr = MPI_Comm_size(PETSC_COMM_WORLD,&size);CHKERRQ(ierr);
  if (size != 1) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,"This is a uniprocessor example only!");
  ierr = PetscOptionsGetInt(NULL,"-n",&n,NULL);CHKERRQ(ierr);
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

  ierr = MatCreate(PETSC_COMM_WORLD,&F);CHKERRQ(ierr);
  ierr = MatSetSizes(F,PETSC_DECIDE,PETSC_DECIDE,n,n);CHKERRQ(ierr);
  ierr = MatSetFromOptions(F);CHKERRQ(ierr);
  ierr = MatSeqAIJSetPreallocation(F,3,NULL);CHKERRQ(ierr);

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

  /*
     Set Jacobian matrix data structure and default Jacobian evaluation
     routine
  */
  ierr = NEPSetJacobian(nep,J,J,FormJacobian,&ctx);CHKERRQ(ierr);

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
     Customize nonlinear solver; set runtime options
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

  /*
     Set solver parameters at runtime
  */
  ierr = NEPSetFromOptions(nep);CHKERRQ(ierr);

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
                      Initialize application
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

  /*
     Compute analytic solution
  */

  /*
     Evaluate initial guess
  */
  ierr = MatGetVecs(F,&x,&r);CHKERRQ(ierr);
  ierr = VecDuplicate(x,&u);CHKERRQ(ierr);
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
  ierr = NEPGetTolerances(nep,&abstol,&rtol,&stol,&maxit,&maxf);CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD," Stopping condition: atol=%G, rtol=%G, stol=%G, maxit=%D, maxf=%D\n",abstol,rtol,stol,maxit,maxf);CHKERRQ(ierr);

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
         "           k              ||T(k)x|| \n"
         "   ----------------- ------------------\n");CHKERRQ(ierr);
    for (i=0;i<nconv;i++) {
      /* 
        Get converged eigenpairs: i-th eigenvalue is stored in kr (real part) and
        ki (imaginary part)
      */
      //ierr = NEPGetEigenpairs(nep,i,&kr,&ki,NULL,NULL);CHKERRQ(ierr);
      kr=0.0;ki=0.0;
      /*
         Compute the relative errors associated to both right and left eigenvectors
      */
      //ierr = EPSComputeRelativeError(eps,i,&norm);CHKERRQ(ierr);
      norm=0.0;

#if defined(PETSC_USE_COMPLEX)
      re = PetscRealPart(kr);
      im = PetscImaginaryPart(kr);
#else
      re = kr;
      im = ki;
#endif 
      if (im!=0.0) {
        ierr = PetscPrintf(PETSC_COMM_WORLD," %9F%+9F j %12G\n",re,im,norm);CHKERRQ(ierr);
      } else {
        ierr = PetscPrintf(PETSC_COMM_WORLD,"   %12F       %12G\n",re,norm);CHKERRQ(ierr);
      }
    }
    ierr = PetscPrintf(PETSC_COMM_WORLD,"\n");CHKERRQ(ierr);
  }

  /*
     Check the error
  */


  ierr = NEPDestroy(&nep);CHKERRQ(ierr);
  ierr = MatDestroy(&F);CHKERRQ(ierr);
  ierr = MatDestroy(&J);CHKERRQ(ierr);
  ierr = VecDestroy(&x);CHKERRQ(ierr);
  ierr = VecDestroy(&r);CHKERRQ(ierr);
  ierr = VecDestroy(&u);CHKERRQ(ierr);
  ierr = SlepcFinalize();
  return 0;
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
.  nep - the NEP context
.  wr  - real part of the scalar argument
.  wi  - imaginary part of the scalar argument
.  ctx - optional user-defined context, as set by NEPSetJacobian()

   Output Parameters:
.  fun - Function matrix
.  B   - optionally different preconditioning matrix
.  flg - flag indicating matrix structure

   Note:
   lambda can be represented as wr+wi*PETSC_i or as wr (in case of a configuration
   with complex scalars). See NEPGetEigenvalues() for details.
*/
PetscErrorCode FormFunction(NEP nep,PetscScalar wr,PetscScalar wi,Mat *fun,Mat *B,MatStructure *flg,void *ctx)
{
  PetscErrorCode ierr;
  ApplicationCtx *user = (ApplicationCtx*)ctx;
  PetscScalar    lambda,A[3];
  PetscReal      h,c;
  PetscInt       i,n,j[3];

  PetscFunctionBeginUser;
  /* in this example the eigenvalue is always real */
  lambda = wr;
  if (wi!=0.0) SETERRQ(PETSC_COMM_SELF,1,"Non-real scalar parameter generated!");

  /*
     Compute Function entries and insert into matrix
  */
  ierr = MatGetSize(*fun,&n,NULL);CHKERRQ(ierr);
  h = user->h;
  c = user->kappa/(lambda-user->kappa);

  /*
     Interior grid points
  */
  for (i=1;i<n-1;i++) {
    j[0] = i-1; j[1] = i; j[2] = i+1;
    A[0] = A[2] = -n-lambda*h/6.0; A[1] = 2.0*(n-lambda*h/3.0);
    ierr = MatSetValues(*fun,1,&i,3,j,A,INSERT_VALUES);CHKERRQ(ierr);
  }

  /*
     Boundary points
  */
  i = 0;
  j[0] = 0; j[1] = 1;
  A[0] = 2.0*(n-lambda*h/3.0); A[1] = -n-lambda*h/6.0;
  ierr = MatSetValues(*fun,1,&i,2,j,A,INSERT_VALUES);CHKERRQ(ierr);

  i = n-1;
  j[0] = n-2; j[1] = n-1;
  A[0] = -n-lambda*h/6.0; A[1] = n-lambda*h/3.0+lambda*c;
  ierr = MatSetValues(*fun,1,&i,2,j,A,INSERT_VALUES);CHKERRQ(ierr);

  /*
     Assemble matrix
  */
  ierr = MatAssemblyBegin(*B,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(*B,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  if (*fun != *B) {
    ierr = MatAssemblyBegin(*fun,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
    ierr = MatAssemblyEnd(*fun,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

/* ------------------------------------------------------------------- */
#undef __FUNCT__
#define __FUNCT__ "FormJacobian"
/*
   FormJacobian - Computes Jacobian matrix  T'(lambda)

   Input Parameters:
.  nep - the NEP context
.  wr  - real part of the scalar argument
.  wi  - imaginary part of the scalar argument
.  ctx - optional user-defined context, as set by NEPSetJacobian()

   Output Parameters:
.  jac - Jacobian matrix
.  B   - optionally different preconditioning matrix
.  flg - flag indicating matrix structure

   Note:
   lambda can be represented as wr+wi*PETSC_i or as wr (in case of a configuration
   with complex scalars). See NEPGetEigenvalues() for details.
*/
PetscErrorCode FormJacobian(NEP nep,PetscScalar wr,PetscScalar wi,Mat *jac,Mat *B,MatStructure *flg,void *ctx)
{
  PetscErrorCode ierr;
  ApplicationCtx *user = (ApplicationCtx*)ctx;
  PetscScalar    lambda,A[3];
  PetscReal      h,c;
  PetscInt       i,n,j[3];

  PetscFunctionBeginUser;
  /* in this example the eigenvalue is always real */
  lambda = wr;
  if (wi!=0.0) SETERRQ(PETSC_COMM_SELF,1,"Non-real scalar parameter generated!");

  /*
     Compute Jacobian entries and insert into matrix
  */
  ierr = MatGetSize(*jac,&n,NULL);CHKERRQ(ierr);
  h = user->h;
  c = user->kappa/(lambda-user->kappa);

  /*
     Interior grid points
  */
  for (i=1;i<n-1;i++) {
    j[0] = i-1; j[1] = i; j[2] = i+1;
    A[0] = A[2] = -h/6.0; A[1] = -2.0*h/3.0;
    ierr = MatSetValues(*jac,1,&i,3,j,A,INSERT_VALUES);CHKERRQ(ierr);
  }

  /*
     Boundary points
  */
  i = 0;
  j[0] = 0; j[1] = 1;
  A[0] = -2.0*h/3.0; A[1] = -h/6.0;
  ierr = MatSetValues(*jac,1,&i,2,j,A,INSERT_VALUES);CHKERRQ(ierr);

  i = n-1;
  j[0] = n-2; j[1] = n-1;
  A[0] = -h/6.0; A[1] = -h/3.0-c*c;
  ierr = MatSetValues(*jac,1,&i,2,j,A,INSERT_VALUES);CHKERRQ(ierr);

  /*
     Assemble matrix
  */
  ierr = MatAssemblyBegin(*B,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(*B,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  if (*jac != *B) {
    ierr = MatAssemblyBegin(*jac,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
    ierr = MatAssemblyEnd(*jac,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

