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
  ierr = PetscOptionsGetInt(NULL,NULL,"-m",&m,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetInt(NULL,NULL,"-n",&n,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetInt(NULL,NULL,"-p",&p,NULL);CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD,"\nGeneralized singular value decomposition, (%D+%D)x%D\n\n",m,p,n);CHKERRQ(ierr);

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
                     Generate the matrices
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

  ierr = MatCreate(PETSC_COMM_WORLD,&A1);CHKERRQ(ierr);
  ierr = MatSetSizes(A1,PETSC_DECIDE,PETSC_DECIDE,m,n);CHKERRQ(ierr);
  ierr = MatSetFromOptions(A1);CHKERRQ(ierr);
  ierr = MatSetUp(A1);CHKERRQ(ierr);
  ierr = MatGetOwnershipRange(A1,&Istart,&Iend);CHKERRQ(ierr);
  for (i=Istart;i<Iend;i++) {
    col[0]=i; col[1]=i+1;
    if (i<n-1) {
      ierr = MatSetValues(A1,1,&i,2,col,valsa,INSERT_VALUES);CHKERRQ(ierr);
    } else if (i==n-1) {
      ierr = MatSetValue(A1,i,col[0],valsa[0],INSERT_VALUES);CHKERRQ(ierr);
    }
  }
  ierr = MatAssemblyBegin(A1,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(A1,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);

  ierr = MatCreate(PETSC_COMM_WORLD,&A2);CHKERRQ(ierr);
  ierr = MatSetSizes(A2,PETSC_DECIDE,PETSC_DECIDE,m,n);CHKERRQ(ierr);
  ierr = MatSetFromOptions(A2);CHKERRQ(ierr);
  ierr = MatSetUp(A2);CHKERRQ(ierr);
  ierr = MatGetOwnershipRange(A2,&Istart,&Iend);CHKERRQ(ierr);
  for (i=Istart;i<Iend;i++) {
    col[0]=i-1; col[1]=i;
    if (i==0) {
      ierr = MatSetValue(A2,i,col[1],valsb[1],INSERT_VALUES);CHKERRQ(ierr);
    } else if (i<n) {
      ierr = MatSetValues(A2,1,&i,2,col,valsb,INSERT_VALUES);CHKERRQ(ierr);
    } else if (i==n) {
      ierr = MatSetValue(A2,i,col[0],valsb[0],INSERT_VALUES);CHKERRQ(ierr);
    }
  }
  ierr = MatAssemblyBegin(A2,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(A2,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);

  ierr = MatCreate(PETSC_COMM_WORLD,&B);CHKERRQ(ierr);
  ierr = MatSetSizes(B,PETSC_DECIDE,PETSC_DECIDE,p,n);CHKERRQ(ierr);
  ierr = MatSetFromOptions(B);CHKERRQ(ierr);
  ierr = MatSetUp(B);CHKERRQ(ierr);
  ierr = MatGetOwnershipRange(B,&Istart,&Iend);CHKERRQ(ierr);
  d = PetscMax(0,n-p);
  for (i=Istart;i<Iend;i++) {
    for (j=0;j<=PetscMin(i,n-1);j++) {
      ierr = MatSetValue(B,i,j+d,1.0,INSERT_VALUES);CHKERRQ(ierr);
    }
  }
  ierr = MatAssemblyBegin(B,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(B,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
         Create the singular value solver, set options and solve
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

  ierr = SVDCreate(PETSC_COMM_WORLD,&svd);CHKERRQ(ierr);
  ierr = SVDSetOperators(svd,A1,B);CHKERRQ(ierr);
  ierr = SVDSetFromOptions(svd);CHKERRQ(ierr);
  ierr = SVDSolve(svd);CHKERRQ(ierr);
  ierr = SVDErrorView(svd,SVD_ERROR_RELATIVE,NULL);CHKERRQ(ierr);

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
                       Solve second problem
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

  ierr = SVDSetOperators(svd,A2,B);CHKERRQ(ierr);
  ierr = SVDSolve(svd);CHKERRQ(ierr);
  ierr = SVDErrorView(svd,SVD_ERROR_RELATIVE,NULL);CHKERRQ(ierr);

  /* Free work space */
  ierr = SVDDestroy(&svd);CHKERRQ(ierr);
  ierr = MatDestroy(&A1);CHKERRQ(ierr);
  ierr = MatDestroy(&A2);CHKERRQ(ierr);
  ierr = MatDestroy(&B);CHKERRQ(ierr);
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
