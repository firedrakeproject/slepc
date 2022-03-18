/*
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
   SLEPc - Scalable Library for Eigenvalue Problem Computations
   Copyright (c) 2002-2021, Universitat Politecnica de Valencia, Spain

   This file is part of SLEPc.
   SLEPc is distributed under a 2-clause BSD license (see LICENSE).
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
*/

static char help[] = "Tests multiple calls to SVDSolve with equal matrix size (GSVD).\n\n"
  "The command line options are:\n"
  "  -m <m>, where <m> = number of rows of A.\n"
  "  -n <n>, where <n> = number of columns of A.\n"
  "  -p <p>, where <p> = number of rows of B.\n\n";

#include <slepcsvd.h>

/*
   This example solves two GSVD problems for the bidiagonal matrices

              |  1  2                     |       |  1                        |
              |     1  2                  |       |  2  1                     |
              |        1  2               |       |     2  1                  |
         A1 = |          .  .             |  A2 = |       .  .                |
              |             .  .          |       |          .  .             |
              |                1  2       |       |             2  1          |
              |                   1  2    |       |                2  1       |

   with B = tril(ones(p,n))
 */

int main(int argc,char **argv)
{
  Mat            A1,A2,B;
  SVD            svd;
  PetscInt       m=15,n=20,p=21,Istart,Iend,i,j,d,col[2];
  PetscScalar    valsa[] = { 1, 2 }, valsb[] = { 2, 1 };
  PetscErrorCode ierr;

  ierr = SlepcInitialize(&argc,&argv,(char*)0,help);if (ierr) return ierr;
  CHKERRQ(PetscOptionsGetInt(NULL,NULL,"-m",&m,NULL));
  CHKERRQ(PetscOptionsGetInt(NULL,NULL,"-n",&n,NULL));
  CHKERRQ(PetscOptionsGetInt(NULL,NULL,"-p",&p,NULL));
  CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"\nGeneralized singular value decomposition, (%" PetscInt_FMT "+%" PetscInt_FMT ")x%" PetscInt_FMT "\n\n",m,p,n));

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
                     Generate the matrices
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

  CHKERRQ(MatCreate(PETSC_COMM_WORLD,&A1));
  CHKERRQ(MatSetSizes(A1,PETSC_DECIDE,PETSC_DECIDE,m,n));
  CHKERRQ(MatSetFromOptions(A1));
  CHKERRQ(MatSetUp(A1));
  CHKERRQ(MatGetOwnershipRange(A1,&Istart,&Iend));
  for (i=Istart;i<Iend;i++) {
    col[0]=i; col[1]=i+1;
    if (i<n-1) {
      CHKERRQ(MatSetValues(A1,1,&i,2,col,valsa,INSERT_VALUES));
    } else if (i==n-1) {
      CHKERRQ(MatSetValue(A1,i,col[0],valsa[0],INSERT_VALUES));
    }
  }
  CHKERRQ(MatAssemblyBegin(A1,MAT_FINAL_ASSEMBLY));
  CHKERRQ(MatAssemblyEnd(A1,MAT_FINAL_ASSEMBLY));

  CHKERRQ(MatCreate(PETSC_COMM_WORLD,&A2));
  CHKERRQ(MatSetSizes(A2,PETSC_DECIDE,PETSC_DECIDE,m,n));
  CHKERRQ(MatSetFromOptions(A2));
  CHKERRQ(MatSetUp(A2));
  CHKERRQ(MatGetOwnershipRange(A2,&Istart,&Iend));
  for (i=Istart;i<Iend;i++) {
    col[0]=i-1; col[1]=i;
    if (i==0) {
      CHKERRQ(MatSetValue(A2,i,col[1],valsb[1],INSERT_VALUES));
    } else if (i<n) {
      CHKERRQ(MatSetValues(A2,1,&i,2,col,valsb,INSERT_VALUES));
    } else if (i==n) {
      CHKERRQ(MatSetValue(A2,i,col[0],valsb[0],INSERT_VALUES));
    }
  }
  CHKERRQ(MatAssemblyBegin(A2,MAT_FINAL_ASSEMBLY));
  CHKERRQ(MatAssemblyEnd(A2,MAT_FINAL_ASSEMBLY));

  CHKERRQ(MatCreate(PETSC_COMM_WORLD,&B));
  CHKERRQ(MatSetSizes(B,PETSC_DECIDE,PETSC_DECIDE,p,n));
  CHKERRQ(MatSetFromOptions(B));
  CHKERRQ(MatSetUp(B));
  CHKERRQ(MatGetOwnershipRange(B,&Istart,&Iend));
  d = PetscMax(0,n-p);
  for (i=Istart;i<Iend;i++) {
    for (j=0;j<=PetscMin(i,n-1);j++) {
      CHKERRQ(MatSetValue(B,i,j+d,1.0,INSERT_VALUES));
    }
  }
  CHKERRQ(MatAssemblyBegin(B,MAT_FINAL_ASSEMBLY));
  CHKERRQ(MatAssemblyEnd(B,MAT_FINAL_ASSEMBLY));

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
         Create the singular value solver, set options and solve
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

  CHKERRQ(SVDCreate(PETSC_COMM_WORLD,&svd));
  CHKERRQ(SVDSetOperators(svd,A1,B));
  CHKERRQ(SVDSetFromOptions(svd));
  CHKERRQ(SVDSolve(svd));
  CHKERRQ(SVDErrorView(svd,SVD_ERROR_NORM,NULL));

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
                       Solve second problem
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

  CHKERRQ(SVDSetOperators(svd,A2,B));
  CHKERRQ(SVDSolve(svd));
  CHKERRQ(SVDErrorView(svd,SVD_ERROR_NORM,NULL));

  /* Free work space */
  CHKERRQ(SVDDestroy(&svd));
  CHKERRQ(MatDestroy(&A1));
  CHKERRQ(MatDestroy(&A2));
  CHKERRQ(MatDestroy(&B));
  ierr = SlepcFinalize();
  return ierr;
}

/*TEST

   testset:
      args: -svd_nsv 3
      requires: !single
      output_file: output/test16_1.out
      test:
         suffix: 1_lapack
         args: -svd_type lapack
      test:
         suffix: 1_cross
         args: -svd_type cross -svd_cross_explicitmatrix {{0 1}}
      test:
         suffix: 1_cyclic
         args: -svd_type cyclic -svd_cyclic_explicitmatrix {{0 1}}
      test:
         suffix: 1_trlanczos
         args: -svd_type trlanczos -svd_trlanczos_gbidiag {{single upper lower}}

TEST*/
