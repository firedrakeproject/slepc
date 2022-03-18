/*
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
   SLEPc - Scalable Library for Eigenvalue Problem Computations
   Copyright (c) 2002-2021, Universitat Politecnica de Valencia, Spain

   This file is part of SLEPc.
   SLEPc is distributed under a 2-clause BSD license (see LICENSE).
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
*/

static char help[] = "Tests a user-defined convergence test (based on ex8.c).\n\n"
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

/*
  MyConvergedRel - Convergence test relative to the norm of A (given in ctx).
*/
PetscErrorCode MyConvergedRel(SVD svd,PetscReal sigma,PetscReal res,PetscReal *errest,void *ctx)
{
  PetscReal norm = *(PetscReal*)ctx;

  PetscFunctionBegin;
  *errest = res/norm;
  PetscFunctionReturn(0);
}

int main(int argc,char **argv)
{
  Mat            A;               /* Grcar matrix */
  SVD            svd;             /* singular value solver context */
  PetscInt       N=30,Istart,Iend,i,col[5],nconv1,nconv2;
  PetscScalar    value[] = { -1, 1, 1, 1, 1 };
  PetscReal      sigma_1,sigma_n;
  PetscErrorCode ierr;

  ierr = SlepcInitialize(&argc,&argv,(char*)0,help);if (ierr) return ierr;

  CHKERRQ(PetscOptionsGetInt(NULL,NULL,"-n",&N,NULL));
  CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"\nEstimate the condition number of a Grcar matrix, n=%" PetscInt_FMT "\n\n",N));

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
      CHKERRQ(MatSetValues(A,1,&i,PetscMin(4,N-i),col+1,value+1,INSERT_VALUES));
    } else {
      CHKERRQ(MatSetValues(A,1,&i,PetscMin(5,N-i+1),col,value,INSERT_VALUES));
    }
  }
  CHKERRQ(MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY));
  CHKERRQ(MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY));

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
             Create the SVD solver and set the solution method
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

  CHKERRQ(SVDCreate(PETSC_COMM_WORLD,&svd));
  CHKERRQ(SVDSetOperators(svd,A,NULL));
  CHKERRQ(SVDSetType(svd,SVDTRLANCZOS));
  CHKERRQ(SVDSetFromOptions(svd));

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
                      Solve the singular value problem
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

  CHKERRQ(SVDSetWhichSingularTriplets(svd,SVD_LARGEST));
  CHKERRQ(SVDSolve(svd));
  CHKERRQ(SVDGetConverged(svd,&nconv1));
  if (nconv1 > 0) {
    CHKERRQ(SVDGetSingularTriplet(svd,0,&sigma_1,NULL,NULL));
  } else {
    CHKERRQ(PetscPrintf(PETSC_COMM_WORLD," Unable to compute large singular value!\n\n"));
  }

  /* compute smallest singular value relative to the matrix norm */
  CHKERRQ(SVDSetConvergenceTestFunction(svd,MyConvergedRel,&sigma_1,NULL));
  CHKERRQ(SVDSetWhichSingularTriplets(svd,SVD_SMALLEST));
  CHKERRQ(SVDSolve(svd));
  CHKERRQ(SVDGetConverged(svd,&nconv2));
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

  CHKERRQ(SVDDestroy(&svd));
  CHKERRQ(MatDestroy(&A));
  ierr = SlepcFinalize();
  return ierr;
}

/*TEST

   test:
      suffix: 1

TEST*/
