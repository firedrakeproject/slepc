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
  ierr = PetscOptionsGetInt(NULL,NULL,"-m",&m,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetInt(NULL,NULL,"-n",&n,&flg);CHKERRQ(ierr);
  if (!flg) n=m+2;
  ierr = PetscPrintf(PETSC_COMM_WORLD,"\nRectangular bidiagonal matrix, m=%D n=%D\n\n",m,n);CHKERRQ(ierr);

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
                     Generate the matrices
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

  ierr = MatCreate(PETSC_COMM_WORLD,&A);CHKERRQ(ierr);
  ierr = MatSetSizes(A,PETSC_DECIDE,PETSC_DECIDE,m,n);CHKERRQ(ierr);
  ierr = MatSetFromOptions(A);CHKERRQ(ierr);
  ierr = MatSetUp(A);CHKERRQ(ierr);
  ierr = MatGetOwnershipRange(A,&Istart,&Iend);CHKERRQ(ierr);
  for (i=Istart;i<Iend;i++) {
    col[0]=i; col[1]=i+1;
    if (i<n-1) {
      ierr = MatSetValues(A,1,&i,2,col,valsa,INSERT_VALUES);CHKERRQ(ierr);
    } else if (i==n-1) {
      ierr = MatSetValue(A,i,col[0],valsa[0],INSERT_VALUES);CHKERRQ(ierr);
    }
  }
  ierr = MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);

  ierr = MatCreate(PETSC_COMM_WORLD,&B);CHKERRQ(ierr);
  ierr = MatSetSizes(B,PETSC_DECIDE,PETSC_DECIDE,m,n);CHKERRQ(ierr);
  ierr = MatSetFromOptions(B);CHKERRQ(ierr);
  ierr = MatSetUp(B);CHKERRQ(ierr);
  ierr = MatGetOwnershipRange(B,&Istart,&Iend);CHKERRQ(ierr);
  for (i=Istart;i<Iend;i++) {
    col[0]=i; col[1]=i+1;
    if (i==0) {
      ierr = MatSetValue(B,i,col[0],valsb[1],INSERT_VALUES);CHKERRQ(ierr);
    } else if (i<n-1) {
      ierr = MatSetValues(B,1,&i,2,col,valsb,INSERT_VALUES);CHKERRQ(ierr);
    } else if (i==n-1) {
      ierr = MatSetValue(B,i,col[0],valsb[0],INSERT_VALUES);CHKERRQ(ierr);
    }
  }
  ierr = MatAssemblyBegin(B,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(B,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
         Create the singular value solver, set options and solve
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

  ierr = SVDCreate(PETSC_COMM_WORLD,&svd);CHKERRQ(ierr);
  ierr = SVDSetOperators(svd,A,NULL);CHKERRQ(ierr);
  ierr = SVDSetTolerances(svd,PETSC_DEFAULT,1000);CHKERRQ(ierr);
  ierr = SVDSetFromOptions(svd);CHKERRQ(ierr);
  ierr = SVDSolve(svd);CHKERRQ(ierr);
  ierr = SVDErrorView(svd,SVD_ERROR_RELATIVE,NULL);CHKERRQ(ierr);

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
                       Solve with second matrix
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

  ierr = SVDSetOperators(svd,B,NULL);CHKERRQ(ierr);
  ierr = SVDSolve(svd);CHKERRQ(ierr);
  ierr = SVDErrorView(svd,SVD_ERROR_RELATIVE,NULL);CHKERRQ(ierr);

  /* Free work space */
  ierr = SVDDestroy(&svd);CHKERRQ(ierr);
  ierr = MatDestroy(&A);CHKERRQ(ierr);
  ierr = MatDestroy(&B);CHKERRQ(ierr);
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
