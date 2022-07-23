/*
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
   SLEPc - Scalable Library for Eigenvalue Problem Computations
   Copyright (c) 2002-, Universitat Politecnica de Valencia, Spain

   This file is part of SLEPc.
   SLEPc is distributed under a 2-clause BSD license (see LICENSE).
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
*/

static char help[] = "Test GSVD with user-provided initial vectors.\n\n"
  "The command line options are:\n"
  "  -m <m>, where <m> = number of rows of A.\n"
  "  -n <n>, where <n> = number of columns of A.\n"
  "  -p <p>, where <p> = number of rows of B.\n\n";

#include <slepcsvd.h>

/*
   This example solves a GSVD problem for the bidiagonal matrices

              |  1  2                     |       |  2                        |
              |     1  2                  |       | -1  2                     |
              |        1  2               |       |    -1  2                  |
          A = |          .  .             |   B = |       .  .                |
              |             .  .          |       |          .  .             |
              |                1  2       |       |            -1  2          |
              |                   1  2    |       |               -1  2       |
 */

int main(int argc,char **argv)
{
  Mat            A,B;
  SVD            svd;
  Vec            v0,w0;           /* initial vectors */
  VecType        vtype;
  PetscInt       m=22,n=20,p=22,Istart,Iend,i,col[2];
  PetscScalar    valsa[] = { 1, 2 }, valsb[] = { -1, 2 };

  PetscFunctionBeginUser;
  PetscCall(SlepcInitialize(&argc,&argv,(char*)0,help));
  PetscCall(PetscOptionsGetInt(NULL,NULL,"-m",&m,NULL));
  PetscCall(PetscOptionsGetInt(NULL,NULL,"-n",&n,NULL));
  PetscCall(PetscOptionsGetInt(NULL,NULL,"-p",&p,NULL));
  PetscCall(PetscPrintf(PETSC_COMM_WORLD,"\nGeneralized singular value decomposition, (%" PetscInt_FMT "+%" PetscInt_FMT ")x%" PetscInt_FMT "\n\n",m,p,n));

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
                     Generate the matrices
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

  PetscCall(MatCreate(PETSC_COMM_WORLD,&A));
  PetscCall(MatSetSizes(A,PETSC_DECIDE,PETSC_DECIDE,m,n));
  PetscCall(MatSetFromOptions(A));
  PetscCall(MatSetUp(A));
  PetscCall(MatGetOwnershipRange(A,&Istart,&Iend));
  for (i=Istart;i<Iend;i++) {
    col[0]=i; col[1]=i+1;
    if (i<n-1) PetscCall(MatSetValues(A,1,&i,2,col,valsa,INSERT_VALUES));
    else if (i==n-1) PetscCall(MatSetValue(A,i,col[0],valsa[0],INSERT_VALUES));
  }
  PetscCall(MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY));

  PetscCall(MatCreate(PETSC_COMM_WORLD,&B));
  PetscCall(MatSetSizes(B,PETSC_DECIDE,PETSC_DECIDE,m,n));
  PetscCall(MatSetFromOptions(B));
  PetscCall(MatSetUp(B));
  PetscCall(MatGetOwnershipRange(B,&Istart,&Iend));
  for (i=Istart;i<Iend;i++) {
    col[0]=i-1; col[1]=i;
    if (i==0) PetscCall(MatSetValue(B,i,col[1],valsb[1],INSERT_VALUES));
    else if (i<n) PetscCall(MatSetValues(B,1,&i,2,col,valsb,INSERT_VALUES));
    else if (i==n) PetscCall(MatSetValue(B,i,col[0],valsb[0],INSERT_VALUES));
  }
  PetscCall(MatAssemblyBegin(B,MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(B,MAT_FINAL_ASSEMBLY));

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
         Create the singular value solver, set options and solve
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

  PetscCall(SVDCreate(PETSC_COMM_WORLD,&svd));
  PetscCall(SVDSetOperators(svd,A,B));
  PetscCall(SVDSetFromOptions(svd));

  /*
     Set the initial vectors. This is optional, if not done the initial
     vectors are set to random values
  */
  PetscCall(MatCreateVecs(A,&v0,NULL));        /* right initial vector, length n */
  PetscCall(VecCreate(PETSC_COMM_WORLD,&w0));  /* left initial vector, length m+p */
  PetscCall(VecSetSizes(w0,PETSC_DECIDE,m+p));
  PetscCall(VecGetType(v0,&vtype));
  PetscCall(VecSetType(w0,vtype));
  PetscCall(VecSet(v0,1.0));
  PetscCall(VecSet(w0,1.0));
  PetscCall(SVDSetInitialSpaces(svd,1,&v0,1,&w0));

  /*
     Compute solution
  */
  PetscCall(SVDSolve(svd));
  PetscCall(SVDErrorView(svd,SVD_ERROR_NORM,NULL));

  /* Free work space */
  PetscCall(VecDestroy(&v0));
  PetscCall(VecDestroy(&w0));
  PetscCall(SVDDestroy(&svd));
  PetscCall(MatDestroy(&A));
  PetscCall(MatDestroy(&B));
  PetscCall(SlepcFinalize());
  return 0;
}

/*TEST

   testset:
      args: -svd_nsv 3
      requires: !single
      output_file: output/test18_1.out
      test:
         suffix: 1
         args: -svd_type {{lapack cross cyclic}}
      test:
         suffix: 1_trlanczos
         args: -svd_type trlanczos -svd_trlanczos_gbidiag {{single upper lower}}
         requires: !__float128

   test:
      suffix: 2
      args: -svd_nsv 3 -svd_type trlanczos -svd_monitor_conditioning
      requires: double

TEST*/
