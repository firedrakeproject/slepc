/*
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
   SLEPc - Scalable Library for Eigenvalue Problem Computations
   Copyright (c) 2002-, Universitat Politecnica de Valencia, Spain

   This file is part of SLEPc.
   SLEPc is distributed under a 2-clause BSD license (see LICENSE).
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
*/

static char help[] = "Tests multiple calls to SVDSolve changing ncv.\n\n"
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
  Mat            A;
  SVD            svd;
  PetscInt       N=30,Istart,Iend,i,col[5],nsv,ncv;
  PetscScalar    value[] = { -1, 1, 1, 1, 1 };

  PetscFunctionBeginUser;
  PetscCall(SlepcInitialize(&argc,&argv,(char*)0,help));
  PetscCall(PetscOptionsGetInt(NULL,NULL,"-n",&N,NULL));
  PetscCall(PetscPrintf(PETSC_COMM_WORLD,"\nSingular values of a Grcar matrix, n=%" PetscInt_FMT,N));
  PetscCall(PetscPrintf(PETSC_COMM_WORLD,"\n\n"));

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        Generate the matrix
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

  PetscCall(MatCreate(PETSC_COMM_WORLD,&A));
  PetscCall(MatSetSizes(A,PETSC_DECIDE,PETSC_DECIDE,N,N));
  PetscCall(MatSetFromOptions(A));
  PetscCall(MatSetUp(A));

  PetscCall(MatGetOwnershipRange(A,&Istart,&Iend));
  for (i=Istart;i<Iend;i++) {
    col[0]=i-1; col[1]=i; col[2]=i+1; col[3]=i+2; col[4]=i+3;
    if (i==0) PetscCall(MatSetValues(A,1,&i,4,col+1,value+1,INSERT_VALUES));
    else PetscCall(MatSetValues(A,1,&i,PetscMin(5,N-i+1),col,value,INSERT_VALUES));
  }

  PetscCall(MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY));

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
         Create the singular value solver and set the solution method
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

  PetscCall(SVDCreate(PETSC_COMM_WORLD,&svd));
  PetscCall(SVDSetOperators(svd,A,NULL));
  PetscCall(SVDSetTolerances(svd,1e-6,1000));
  PetscCall(SVDSetWhichSingularTriplets(svd,SVD_LARGEST));
  PetscCall(SVDSetFromOptions(svd));

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
                      Compute the singular values
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

  /* First solve */
  PetscCall(SVDSolve(svd));
  PetscCall(PetscPrintf(PETSC_COMM_WORLD," - - - First solve, default subspace dimension - - -\n"));
  PetscCall(SVDErrorView(svd,SVD_ERROR_RELATIVE,NULL));

  /* Second solve */
  PetscCall(SVDGetDimensions(svd,&nsv,&ncv,NULL));
  PetscCall(SVDSetDimensions(svd,nsv,ncv+2,PETSC_DEFAULT));
  PetscCall(SVDSolve(svd));
  PetscCall(PetscPrintf(PETSC_COMM_WORLD," - - - Second solve, subspace of increased size - - -\n"));
  PetscCall(SVDErrorView(svd,SVD_ERROR_RELATIVE,NULL));

  /* Free work space */
  PetscCall(SVDDestroy(&svd));
  PetscCall(MatDestroy(&A));
  PetscCall(SlepcFinalize());
  return 0;
}

/*TEST

   test:
      suffix: 1
      args: -svd_type {{lanczos trlanczos cross cyclic lapack randomized}} -svd_nsv 3 -svd_ncv 12
      requires: !single

TEST*/
