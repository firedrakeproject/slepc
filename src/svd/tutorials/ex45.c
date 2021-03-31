/*
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
   SLEPc - Scalable Library for Eigenvalue Problem Computations
   Copyright (c) 2002-2021, Universitat Politecnica de Valencia, Spain

   This file is part of SLEPc.
   SLEPc is distributed under a 2-clause BSD license (see LICENSE).
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
*/

static char help[] = "Computes a partial generalized singular value decomposition (GSVD).\n"
  "The command line options are:\n"
  "  -m <m>, where <m> = number of rows of A.\n"
  "  -n <n>, where <n> = number of columns of A.\n"
  "  -p <p>, where <p> = number of rows of B.\n\n";

#include <slepcsvd.h>

int main(int argc,char **argv)
{
  Mat            A,B;             /* operator matrices */
  Vec            u,v,x;           /* singular vectors */
  SVD            svd;             /* singular value problem solver context */
  SVDType        type;
  Vec            uv,aux[2];
  PetscReal      error,tol,sigma;
  PetscInt       m=100,n,p=14,i,j,Istart,Iend,nsv,maxit,its,nconv;
  PetscBool      flg;
  PetscErrorCode ierr;

  ierr = SlepcInitialize(&argc,&argv,(char*)0,help);if (ierr) return ierr;

  ierr = PetscOptionsGetInt(NULL,NULL,"-m",&m,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetInt(NULL,NULL,"-n",&n,&flg);CHKERRQ(ierr);
  if (!flg) n = m;
  ierr = PetscOptionsGetInt(NULL,NULL,"-p",&p,&flg);CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD,"\nGeneralized singular value decomposition, (%D+%D)x%D\n\n",m,p,n);CHKERRQ(ierr);

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
                          Build the matrices
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

  ierr = MatCreate(PETSC_COMM_WORLD,&A);CHKERRQ(ierr);
  ierr = MatSetSizes(A,PETSC_DECIDE,PETSC_DECIDE,m,n);CHKERRQ(ierr);
  ierr = MatSetFromOptions(A);CHKERRQ(ierr);
  ierr = MatSetUp(A);CHKERRQ(ierr);

  ierr = MatGetOwnershipRange(A,&Istart,&Iend);CHKERRQ(ierr);
  for (i=Istart;i<Iend;i++) {
    if (i>0 && i-1<n) { ierr = MatSetValue(A,i,i-1,-1.0,INSERT_VALUES);CHKERRQ(ierr); }
    if (i+1<n) { ierr = MatSetValue(A,i,i+1,-1.0,INSERT_VALUES);CHKERRQ(ierr); }
    if (i<n) { ierr = MatSetValue(A,i,i,2.0,INSERT_VALUES);CHKERRQ(ierr); }
    if (i>n) { ierr = MatSetValue(A,i,n-1,1.0,INSERT_VALUES);CHKERRQ(ierr); }
  }
  ierr = MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);

  ierr = MatCreate(PETSC_COMM_WORLD,&B);CHKERRQ(ierr);
  ierr = MatSetSizes(B,PETSC_DECIDE,PETSC_DECIDE,p,n);CHKERRQ(ierr);
  ierr = MatSetFromOptions(B);CHKERRQ(ierr);
  ierr = MatSetUp(B);CHKERRQ(ierr);

  ierr = MatGetOwnershipRange(B,&Istart,&Iend);CHKERRQ(ierr);
  for (i=Istart;i<Iend;i++) {
    for (j=0;j<=PetscMin(i,n-1);j++) {
      ierr = MatSetValue(B,i,j,1.0,INSERT_VALUES);CHKERRQ(ierr);
    }
  }
  ierr = MatAssemblyBegin(B,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(B,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
          Create the singular value solver and set various options
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

  /*
     Create singular value solver context
  */
  ierr = SVDCreate(PETSC_COMM_WORLD,&svd);CHKERRQ(ierr);

  /*
     Set operators and problem type
  */
  ierr = SVDSetOperators(svd,A,B);CHKERRQ(ierr);
  ierr = SVDSetProblemType(svd,SVD_GENERALIZED);CHKERRQ(ierr);

  /*
     Set solver parameters at runtime
  */
  ierr = SVDSetFromOptions(svd);CHKERRQ(ierr);

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
                      Solve the singular value system
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

  ierr = SVDSolve(svd);CHKERRQ(ierr);
  ierr = SVDGetIterationNumber(svd,&its);CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD," Number of iterations of the method: %D\n",its);CHKERRQ(ierr);

  /*
     Optional: Get some information from the solver and display it
  */
  ierr = SVDGetType(svd,&type);CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD," Solution method: %s\n\n",type);CHKERRQ(ierr);
  ierr = SVDGetDimensions(svd,&nsv,NULL,NULL);CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD," Number of requested singular values: %D\n",nsv);CHKERRQ(ierr);
  ierr = SVDGetTolerances(svd,&tol,&maxit);CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD," Stopping condition: tol=%.4g, maxit=%D\n",(double)tol,maxit);CHKERRQ(ierr);

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
                    Display solution and clean up
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

  /*
     Get number of converged singular triplets
  */
  ierr = SVDGetConverged(svd,&nconv);CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD," Number of converged approximate singular triplets: %D\n\n",nconv);CHKERRQ(ierr);

  if (nconv>0) {
    /*
       Create vectors. The interface returns u and v as stacked on top of each other
       [u;v] so need to create a special vector (VecNest) to extract them
    */
    ierr = MatCreateVecs(A,&x,&u);CHKERRQ(ierr);
    ierr = MatCreateVecs(B,NULL,&v);CHKERRQ(ierr);
    aux[0] = u;
    aux[1] = v;
    ierr = VecCreateNest(PETSC_COMM_WORLD,2,NULL,aux,&uv);CHKERRQ(ierr);

    /*
       Display singular values and relative errors
    */
    ierr = PetscPrintf(PETSC_COMM_WORLD,
         "          sigma           relative error\n"
         "  --------------------- ------------------\n");CHKERRQ(ierr);
    for (i=0;i<nconv;i++) {
      /*
         Get converged singular triplets: i-th singular value is stored in sigma
      */
      ierr = SVDGetSingularTriplet(svd,i,&sigma,uv,x);CHKERRQ(ierr);
      /* at this point, u and v can be used normally as individual vectors */

      /*
         Compute the error associated to each singular triplet
      */
      ierr = SVDComputeError(svd,i,SVD_ERROR_RELATIVE,&error);CHKERRQ(ierr);

      ierr = PetscPrintf(PETSC_COMM_WORLD,"       % 6f      ",(double)sigma);CHKERRQ(ierr);
      ierr = PetscPrintf(PETSC_COMM_WORLD,"   % 12g\n",(double)error);CHKERRQ(ierr);
    }
    ierr = PetscPrintf(PETSC_COMM_WORLD,"\n");CHKERRQ(ierr);
    ierr = VecDestroy(&x);CHKERRQ(ierr);
    ierr = VecDestroy(&u);CHKERRQ(ierr);
    ierr = VecDestroy(&v);CHKERRQ(ierr);
    ierr = VecDestroy(&uv);CHKERRQ(ierr);
  }

  /*
     Free work space
  */
  ierr = SVDDestroy(&svd);CHKERRQ(ierr);
  ierr = MatDestroy(&A);CHKERRQ(ierr);
  ierr = MatDestroy(&B);CHKERRQ(ierr);
  ierr = SlepcFinalize();
  return ierr;
}

/*TEST

   test:
      args: -m 20 -n 10 -p 6 -svd_type lapack
      suffix: 1
      filter: sed -e "s/[0-9]\.[0-9]*e[+-]\([0-9]*\)/removed/g"
      requires: double

   test:
      args: -m 15 -n 20 -p 20 -svd_type lapack
      suffix: 2
      filter: sed -e "s/[0-9]\.[0-9]*e[+-]\([0-9]*\)/removed/g"
      requires: double

TEST*/
