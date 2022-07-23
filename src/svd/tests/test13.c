/*
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
   SLEPc - Scalable Library for Eigenvalue Problem Computations
   Copyright (c) 2002-, Universitat Politecnica de Valencia, Spain

   This file is part of SLEPc.
   SLEPc is distributed under a 2-clause BSD license (see LICENSE).
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
*/

static char help[] = "Test interface to external package PRIMME.\n\n"
  "The command line options are:\n"
  "  -m <m>, where <m> = matrix rows.\n"
  "  -n <n>, where <n> = matrix columns (defaults to m+2).\n\n";

#include <slepcsvd.h>

/*
   This example computes the singular values of a rectangular bidiagonal matrix

              |  1  2                     |
              |     1  2                  |
              |        1  2               |
          A = |          .  .             |
              |             .  .          |
              |                1  2       |
              |                   1  2    |
 */

int main(int argc,char **argv)
{
  Mat             A;
  SVD             svd;
  PetscInt        m=20,n,Istart,Iend,i,k=6,col[2],bs;
  PetscScalar     value[] = { 1, 2 };
  PetscBool       flg;
  SVDPRIMMEMethod meth;

  PetscFunctionBeginUser;
  PetscCall(SlepcInitialize(&argc,&argv,(char*)0,help));
  PetscCall(PetscOptionsGetInt(NULL,NULL,"-m",&m,NULL));
  PetscCall(PetscOptionsGetInt(NULL,NULL,"-n",&n,&flg));
  if (!flg) n=m+2;
  PetscCall(PetscOptionsGetInt(NULL,NULL,"-k",&k,NULL));
  PetscCall(PetscPrintf(PETSC_COMM_WORLD,"\nRectangular bidiagonal matrix, m=%" PetscInt_FMT " n=%" PetscInt_FMT "\n\n",m,n));

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        Generate the matrix
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

  PetscCall(MatCreate(PETSC_COMM_WORLD,&A));
  PetscCall(MatSetSizes(A,PETSC_DECIDE,PETSC_DECIDE,m,n));
  PetscCall(MatSetFromOptions(A));
  PetscCall(MatSetUp(A));
  PetscCall(MatGetOwnershipRange(A,&Istart,&Iend));
  for (i=Istart;i<Iend;i++) {
    col[0]=i; col[1]=i+1;
    if (i<n-1) PetscCall(MatSetValues(A,1,&i,2,col,value,INSERT_VALUES));
    else if (i==n-1) PetscCall(MatSetValue(A,i,col[0],value[0],INSERT_VALUES));
  }
  PetscCall(MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY));

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        Compute singular values
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

  PetscCall(SVDCreate(PETSC_COMM_WORLD,&svd));
  PetscCall(SVDSetOperators(svd,A,NULL));
  PetscCall(SVDSetType(svd,SVDPRIMME));
  PetscCall(SVDPRIMMESetBlockSize(svd,4));
  PetscCall(SVDPRIMMESetMethod(svd,SVD_PRIMME_HYBRID));
  PetscCall(SVDSetFromOptions(svd));

  PetscCall(SVDSolve(svd));
  PetscCall(SVDPRIMMEGetBlockSize(svd,&bs));
  PetscCall(SVDPRIMMEGetMethod(svd,&meth));
  PetscCall(PetscPrintf(PETSC_COMM_WORLD," PRIMME: using block size %" PetscInt_FMT ", method %s\n",bs,SVDPRIMMEMethods[meth]));

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
                    Display solution and clean up
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  PetscCall(SVDErrorView(svd,SVD_ERROR_RELATIVE,PETSC_VIEWER_STDOUT_WORLD));
  PetscCall(SVDDestroy(&svd));
  PetscCall(MatDestroy(&A));
  PetscCall(SlepcFinalize());
  return 0;
}

/*TEST

   build:
      requires: primme

   test:
      suffix: 1
      nsize: 2
      args: -svd_nsv 4
      requires: primme
      filter: sed -e "s/2.99255/2.99254/"

TEST*/
