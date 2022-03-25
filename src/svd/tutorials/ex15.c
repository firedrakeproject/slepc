/*
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
   SLEPc - Scalable Library for Eigenvalue Problem Computations
   Copyright (c) 2002-2021, Universitat Politecnica de Valencia, Spain

   This file is part of SLEPc.
   SLEPc is distributed under a 2-clause BSD license (see LICENSE).
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
*/

static char help[] = "Singular value decomposition of the Lauchli matrix.\n"
  "The command line options are:\n"
  "  -n <n>, where <n> = matrix dimension.\n"
  "  -mu <mu>, where <mu> = subdiagonal value.\n\n";

#include <slepcsvd.h>

int main(int argc,char **argv)
{
  Mat            A;               /* operator matrix */
  Vec            u,v;             /* left and right singular vectors */
  SVD            svd;             /* singular value problem solver context */
  SVDType        type;
  PetscReal      error,tol,sigma,mu=PETSC_SQRT_MACHINE_EPSILON;
  PetscInt       n=100,i,j,Istart,Iend,nsv,maxit,its,nconv;

  CHKERRQ(SlepcInitialize(&argc,&argv,(char*)0,help));

  CHKERRQ(PetscOptionsGetInt(NULL,NULL,"-n",&n,NULL));
  CHKERRQ(PetscOptionsGetReal(NULL,NULL,"-mu",&mu,NULL));
  CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"\nLauchli singular value decomposition, (%" PetscInt_FMT " x %" PetscInt_FMT ") mu=%g\n\n",n+1,n,(double)mu));

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
                          Build the Lauchli matrix
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

  CHKERRQ(MatCreate(PETSC_COMM_WORLD,&A));
  CHKERRQ(MatSetSizes(A,PETSC_DECIDE,PETSC_DECIDE,n+1,n));
  CHKERRQ(MatSetFromOptions(A));
  CHKERRQ(MatSetUp(A));

  CHKERRQ(MatGetOwnershipRange(A,&Istart,&Iend));
  for (i=Istart;i<Iend;i++) {
    if (i == 0) {
      for (j=0;j<n;j++) CHKERRQ(MatSetValue(A,0,j,1.0,INSERT_VALUES));
    } else CHKERRQ(MatSetValue(A,i,i-1,mu,INSERT_VALUES));
  }

  CHKERRQ(MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY));
  CHKERRQ(MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY));
  CHKERRQ(MatCreateVecs(A,&v,&u));

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
          Create the singular value solver and set various options
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

  /*
     Create singular value solver context
  */
  CHKERRQ(SVDCreate(PETSC_COMM_WORLD,&svd));

  /*
     Set operators and problem type
  */
  CHKERRQ(SVDSetOperators(svd,A,NULL));
  CHKERRQ(SVDSetProblemType(svd,SVD_STANDARD));

  /*
     Use thick-restart Lanczos as default solver
  */
  CHKERRQ(SVDSetType(svd,SVDTRLANCZOS));

  /*
     Set solver parameters at runtime
  */
  CHKERRQ(SVDSetFromOptions(svd));

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
                      Solve the singular value system
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

  CHKERRQ(SVDSolve(svd));
  CHKERRQ(SVDGetIterationNumber(svd,&its));
  CHKERRQ(PetscPrintf(PETSC_COMM_WORLD," Number of iterations of the method: %" PetscInt_FMT "\n",its));

  /*
     Optional: Get some information from the solver and display it
  */
  CHKERRQ(SVDGetType(svd,&type));
  CHKERRQ(PetscPrintf(PETSC_COMM_WORLD," Solution method: %s\n\n",type));
  CHKERRQ(SVDGetDimensions(svd,&nsv,NULL,NULL));
  CHKERRQ(PetscPrintf(PETSC_COMM_WORLD," Number of requested singular values: %" PetscInt_FMT "\n",nsv));
  CHKERRQ(SVDGetTolerances(svd,&tol,&maxit));
  CHKERRQ(PetscPrintf(PETSC_COMM_WORLD," Stopping condition: tol=%.4g, maxit=%" PetscInt_FMT "\n",(double)tol,maxit));

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
                    Display solution and clean up
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

  /*
     Get number of converged singular triplets
  */
  CHKERRQ(SVDGetConverged(svd,&nconv));
  CHKERRQ(PetscPrintf(PETSC_COMM_WORLD," Number of converged approximate singular triplets: %" PetscInt_FMT "\n\n",nconv));

  if (nconv>0) {
    /*
       Display singular values and relative errors
    */
    CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,
         "          sigma           relative error\n"
         "  --------------------- ------------------\n"));
    for (i=0;i<nconv;i++) {
      /*
         Get converged singular triplets: i-th singular value is stored in sigma
      */
      CHKERRQ(SVDGetSingularTriplet(svd,i,&sigma,u,v));

      /*
         Compute the error associated to each singular triplet
      */
      CHKERRQ(SVDComputeError(svd,i,SVD_ERROR_RELATIVE,&error));

      CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"       % 6f      ",(double)sigma));
      CHKERRQ(PetscPrintf(PETSC_COMM_WORLD," % 12g\n",(double)error));
    }
    CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"\n"));
  }

  /*
     Free work space
  */
  CHKERRQ(SVDDestroy(&svd));
  CHKERRQ(MatDestroy(&A));
  CHKERRQ(VecDestroy(&u));
  CHKERRQ(VecDestroy(&v));
  CHKERRQ(SlepcFinalize());
  return 0;
}

/*TEST

   testset:
      filter: sed -e "s/[0-9]\.[0-9]*e[+-]\([0-9]*\)/removed/g"
      requires: double
      test:
         suffix: 1
      test:
         suffix: 1_scalapack
         nsize: {{1 2}}
         args: -svd_type scalapack
         requires: scalapack

TEST*/
