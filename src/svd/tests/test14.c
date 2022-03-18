/*
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
   SLEPc - Scalable Library for Eigenvalue Problem Computations
   Copyright (c) 2002-2021, Universitat Politecnica de Valencia, Spain

   This file is part of SLEPc.
   SLEPc is distributed under a 2-clause BSD license (see LICENSE).
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
*/

static char help[] = "Tests multiple calls to SVDSolve with equal matrix size.\n\n"
  "The command line options are:\n"
  "  -m <m>, where <m> = matrix rows.\n"
  "  -n <n>, where <n> = matrix columns (defaults to m+2).\n\n";

#include <slepcsvd.h>

/*
   This example computes the singular values of two rectangular bidiagonal matrices

              |  1  2                     |       |  1                        |
              |     1  2                  |       |  2  1                     |
              |        1  2               |       |     2  1                  |
          A = |          .  .             |   B = |       .  .                |
              |             .  .          |       |          .  .             |
              |                1  2       |       |             2  1          |
              |                   1  2    |       |                2  1       |
 */

int main(int argc,char **argv)
{
  Mat            A,B;
  SVD            svd;
  PetscInt       m=20,n,Istart,Iend,i,col[2];
  PetscScalar    valsa[] = { 1, 2 }, valsb[] = { 2, 1 };
  PetscBool      flg;
  PetscErrorCode ierr;

  ierr = SlepcInitialize(&argc,&argv,(char*)0,help);if (ierr) return ierr;
  CHKERRQ(PetscOptionsGetInt(NULL,NULL,"-m",&m,NULL));
  CHKERRQ(PetscOptionsGetInt(NULL,NULL,"-n",&n,&flg));
  if (!flg) n=m+2;
  CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"\nRectangular bidiagonal matrix, m=%" PetscInt_FMT " n=%" PetscInt_FMT "\n\n",m,n));

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
                     Generate the matrices
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

  CHKERRQ(MatCreate(PETSC_COMM_WORLD,&A));
  CHKERRQ(MatSetSizes(A,PETSC_DECIDE,PETSC_DECIDE,m,n));
  CHKERRQ(MatSetFromOptions(A));
  CHKERRQ(MatSetUp(A));
  CHKERRQ(MatGetOwnershipRange(A,&Istart,&Iend));
  for (i=Istart;i<Iend;i++) {
    col[0]=i; col[1]=i+1;
    if (i<n-1) {
      CHKERRQ(MatSetValues(A,1,&i,2,col,valsa,INSERT_VALUES));
    } else if (i==n-1) {
      CHKERRQ(MatSetValue(A,i,col[0],valsa[0],INSERT_VALUES));
    }
  }
  CHKERRQ(MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY));
  CHKERRQ(MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY));

  CHKERRQ(MatCreate(PETSC_COMM_WORLD,&B));
  CHKERRQ(MatSetSizes(B,PETSC_DECIDE,PETSC_DECIDE,m,n));
  CHKERRQ(MatSetFromOptions(B));
  CHKERRQ(MatSetUp(B));
  CHKERRQ(MatGetOwnershipRange(B,&Istart,&Iend));
  for (i=Istart;i<Iend;i++) {
    col[0]=i-1; col[1]=i;
    if (i==0) {
      CHKERRQ(MatSetValue(B,i,col[1],valsb[1],INSERT_VALUES));
    } else if (i<n) {
      CHKERRQ(MatSetValues(B,1,&i,2,col,valsb,INSERT_VALUES));
    } else if (i==n) {
      CHKERRQ(MatSetValue(B,i,col[0],valsb[0],INSERT_VALUES));
    }
  }
  CHKERRQ(MatAssemblyBegin(B,MAT_FINAL_ASSEMBLY));
  CHKERRQ(MatAssemblyEnd(B,MAT_FINAL_ASSEMBLY));

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
         Create the singular value solver, set options and solve
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

  CHKERRQ(SVDCreate(PETSC_COMM_WORLD,&svd));
  CHKERRQ(SVDSetOperators(svd,A,NULL));
  CHKERRQ(SVDSetTolerances(svd,PETSC_DEFAULT,1000));
  CHKERRQ(SVDSetFromOptions(svd));
  CHKERRQ(SVDSolve(svd));
  CHKERRQ(SVDErrorView(svd,SVD_ERROR_RELATIVE,NULL));

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
                       Solve with second matrix
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

  CHKERRQ(SVDSetOperators(svd,B,NULL));
  CHKERRQ(SVDSolve(svd));
  CHKERRQ(SVDErrorView(svd,SVD_ERROR_RELATIVE,NULL));

  /* Free work space */
  CHKERRQ(SVDDestroy(&svd));
  CHKERRQ(MatDestroy(&A));
  CHKERRQ(MatDestroy(&B));
  ierr = SlepcFinalize();
  return ierr;
}

/*TEST

   testset:
      args: -svd_nsv 3
      requires: !single
      output_file: output/test14_1.out
      test:
         suffix: 1
         args: -svd_type {{lanczos trlanczos lapack}}
      test:
         suffix: 1_cross
         args: -svd_type cross -svd_cross_explicitmatrix {{0 1}}
      test:
         suffix: 1_cyclic
         args: -svd_type cyclic -svd_cyclic_explicitmatrix {{0 1}}

   testset:
      args: -n 18 -svd_nsv 3
      requires: !single
      output_file: output/test14_2.out
      test:
         suffix: 2
         args: -svd_type {{lanczos trlanczos lapack}}
      test:
         suffix: 2_cross
         args: -svd_type cross -svd_cross_explicitmatrix {{0 1}}
      test:
         suffix: 2_cyclic
         args: -svd_type cyclic -svd_cyclic_explicitmatrix {{0 1}}

TEST*/
