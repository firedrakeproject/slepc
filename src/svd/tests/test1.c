/*
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
   SLEPc - Scalable Library for Eigenvalue Problem Computations
   Copyright (c) 2002-2021, Universitat Politecnica de Valencia, Spain

   This file is part of SLEPc.
   SLEPc is distributed under a 2-clause BSD license (see LICENSE).
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
*/

static char help[] = "Test the solution of a SVD without calling SVDSetFromOptions (based on ex8.c).\n\n"
  "The command line options are:\n"
  "  -n <n>, where <n> = matrix dimension.\n"
  "  -type <svd_type> = svd type to test.\n\n";

#include <slepcsvd.h>

/*
   This example computes the singular values of an nxn Grcar matrix,
   which is a nonsymmetric Toeplitz matrix:

              |  1  1  1  1               |
              | -1  1  1  1  1            |
              |    -1  1  1  1  1         |
              |       .  .  .  .  .       |
          A = |          .  .  .  .  .    |
              |            -1  1  1  1  1 |
              |               -1  1  1  1 |
              |                  -1  1  1 |
              |                     -1  1 |

 */

int main(int argc,char **argv)
{
  Mat            A;               /* Grcar matrix */
  SVD            svd;             /* singular value solver context */
  PetscInt       N=30,Istart,Iend,i,col[5],nconv1,nconv2;
  PetscScalar    value[] = { -1, 1, 1, 1, 1 };
  PetscReal      sigma_1,sigma_n;
  char           svdtype[30] = "cross",epstype[30] = "";
  PetscBool      flg;
  EPS            eps;
  PetscErrorCode ierr;

  ierr = SlepcInitialize(&argc,&argv,(char*)0,help);if (ierr) return ierr;

  CHKERRQ(PetscOptionsGetInt(NULL,NULL,"-n",&N,NULL));
  CHKERRQ(PetscOptionsGetString(NULL,NULL,"-type",svdtype,sizeof(svdtype),NULL));
  CHKERRQ(PetscOptionsGetString(NULL,NULL,"-epstype",epstype,sizeof(epstype),&flg));
  CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"\nEstimate the condition number of a Grcar matrix, n=%" PetscInt_FMT,N));
  CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"\n\n"));

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        Generate the matrix
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

  CHKERRQ(MatCreate(PETSC_COMM_WORLD,&A));
  CHKERRQ(MatSetSizes(A,PETSC_DECIDE,PETSC_DECIDE,N,N));
  CHKERRQ(MatSetFromOptions(A));
  CHKERRQ(MatSetUp(A));

  CHKERRQ(MatGetOwnershipRange(A,&Istart,&Iend));
  for (i=Istart;i<Iend;i++) {
    col[0]=i-1; col[1]=i; col[2]=i+1; col[3]=i+2; col[4]=i+3;
    if (i==0) {
      CHKERRQ(MatSetValues(A,1,&i,4,col+1,value+1,INSERT_VALUES));
    } else {
      CHKERRQ(MatSetValues(A,1,&i,PetscMin(5,N-i+1),col,value,INSERT_VALUES));
    }
  }

  CHKERRQ(MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY));
  CHKERRQ(MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY));

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
         Create the singular value solver and set the solution method
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

  /*
     Create singular value context
  */
  CHKERRQ(SVDCreate(PETSC_COMM_WORLD,&svd));

  /*
     Set operator
  */
  CHKERRQ(SVDSetOperators(svd,A,NULL));

  /*
     Set solver parameters at runtime
  */
  CHKERRQ(SVDSetType(svd,svdtype));
  if (flg) {
    CHKERRQ(PetscObjectTypeCompare((PetscObject)svd,SVDCROSS,&flg));
    if (flg) {
      CHKERRQ(SVDCrossGetEPS(svd,&eps));
      CHKERRQ(EPSSetType(eps,epstype));
    }
    CHKERRQ(PetscObjectTypeCompare((PetscObject)svd,SVDCYCLIC,&flg));
    if (flg) {
      CHKERRQ(SVDCyclicGetEPS(svd,&eps));
      CHKERRQ(EPSSetType(eps,epstype));
    }
  }
  CHKERRQ(SVDSetDimensions(svd,1,PETSC_DEFAULT,PETSC_DEFAULT));
  CHKERRQ(SVDSetTolerances(svd,1e-6,1000));

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
                      Compute the singular values
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

  /*
     First request the largest singular value
  */
  CHKERRQ(SVDSetWhichSingularTriplets(svd,SVD_LARGEST));
  CHKERRQ(SVDSolve(svd));
  /*
     Get number of converged singular values
  */
  CHKERRQ(SVDGetConverged(svd,&nconv1));
  /*
     Get converged singular values: largest singular value is stored in sigma_1.
     In this example, we are not interested in the singular vectors
  */
  if (nconv1 > 0) {
    CHKERRQ(SVDGetSingularTriplet(svd,0,&sigma_1,NULL,NULL));
  } else {
    CHKERRQ(PetscPrintf(PETSC_COMM_WORLD," Unable to compute large singular value!\n\n"));
  }

  /*
     Request the smallest singular value
  */
  CHKERRQ(SVDSetWhichSingularTriplets(svd,SVD_SMALLEST));
  CHKERRQ(SVDSolve(svd));
  /*
     Get number of converged triplets
  */
  CHKERRQ(SVDGetConverged(svd,&nconv2));
  /*
     Get converged singular values: smallest singular value is stored in sigma_n.
     As before, we are not interested in the singular vectors
  */
  if (nconv2 > 0) {
    CHKERRQ(SVDGetSingularTriplet(svd,0,&sigma_n,NULL,NULL));
  } else {
    CHKERRQ(PetscPrintf(PETSC_COMM_WORLD," Unable to compute small singular value!\n\n"));
  }

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
                    Display solution and clean up
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  if (nconv1 > 0 && nconv2 > 0) {
    CHKERRQ(PetscPrintf(PETSC_COMM_WORLD," Computed singular values: sigma_1=%.4f, sigma_n=%.4f\n",(double)sigma_1,(double)sigma_n));
    CHKERRQ(PetscPrintf(PETSC_COMM_WORLD," Estimated condition number: sigma_1/sigma_n=%.4f\n\n",(double)(sigma_1/sigma_n)));
  }

  /*
     Free work space
  */
  CHKERRQ(SVDDestroy(&svd));
  CHKERRQ(MatDestroy(&A));
  ierr = SlepcFinalize();
  return ierr;
}

/*TEST

   test:
      suffix: 1
      args: -type {{lanczos trlanczos cross cyclic lapack}}

   test:
      suffix: 1_cross_gd
      args: -type cross -epstype gd
      output_file: output/test1_1.out

   test:
      suffix: 1_cyclic_gd
      args: -type cyclic -epstype gd
      output_file: output/test1_1.out
      requires: !single

   test:
      suffix: 1_primme
      args: -type primme
      requires: primme
      output_file: output/test1_1.out

TEST*/
