/*
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
   SLEPc - Scalable Library for Eigenvalue Problem Computations
   Copyright (c) 2002-2021, Universitat Politecnica de Valencia, Spain

   This file is part of SLEPc.
   SLEPc is distributed under a 2-clause BSD license (see LICENSE).
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
*/

static char help[] = "Tests multiple calls to SVDSolve with different matrix size.\n\n"
  "The command line options are:\n"
  "  -n <n>, where <n> = matrix dimension.\n\n";

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
  Mat            A,B;
  SVD            svd;
  PetscInt       N=30,Istart,Iend,i,col[5];
  PetscScalar    value[] = { -1, 1, 1, 1, 1 };
  PetscErrorCode ierr;

  ierr = SlepcInitialize(&argc,&argv,(char*)0,help);if (ierr) return ierr;
  CHKERRQ(PetscOptionsGetInt(NULL,NULL,"-n",&N,NULL));
  CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"\nSingular values of a Grcar matrix, n=%" PetscInt_FMT "\n\n",N));

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        Generate the matrix of size N
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
         Create the singular value solver, set options and solve
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

  CHKERRQ(SVDCreate(PETSC_COMM_WORLD,&svd));
  CHKERRQ(SVDSetOperators(svd,A,NULL));
  CHKERRQ(SVDSetTolerances(svd,1e-6,1000));
  CHKERRQ(SVDSetFromOptions(svd));
  CHKERRQ(SVDSolve(svd));
  CHKERRQ(SVDErrorView(svd,SVD_ERROR_RELATIVE,NULL));

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        Generate the matrix of size 2*N
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

  N *= 2;
  CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"\nSingular values of a Grcar matrix, n=%" PetscInt_FMT "\n\n",N));

  CHKERRQ(MatCreate(PETSC_COMM_WORLD,&B));
  CHKERRQ(MatSetSizes(B,PETSC_DECIDE,PETSC_DECIDE,N,N));
  CHKERRQ(MatSetFromOptions(B));
  CHKERRQ(MatSetUp(B));
  CHKERRQ(MatGetOwnershipRange(B,&Istart,&Iend));
  for (i=Istart;i<Iend;i++) {
    col[0]=i-1; col[1]=i; col[2]=i+1; col[3]=i+2; col[4]=i+3;
    if (i==0) {
      CHKERRQ(MatSetValues(B,1,&i,4,col+1,value+1,INSERT_VALUES));
    } else {
      CHKERRQ(MatSetValues(B,1,&i,PetscMin(5,N-i+1),col,value,INSERT_VALUES));
    }
  }
  CHKERRQ(MatAssemblyBegin(B,MAT_FINAL_ASSEMBLY));
  CHKERRQ(MatAssemblyEnd(B,MAT_FINAL_ASSEMBLY));

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
       Solve again, calling SVDReset() since matrix size has changed
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

  CHKERRQ(SVDReset(svd));  /* if this is omitted, it will be called in SVDSetOperators() */
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

   test:
      suffix: 1
      args: -svd_type {{lanczos trlanczos cross cyclic lapack randomized}} -svd_nsv 3
      requires: double

TEST*/
