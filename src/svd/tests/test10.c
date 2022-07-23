/*
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
   SLEPc - Scalable Library for Eigenvalue Problem Computations
   Copyright (c) 2002-, Universitat Politecnica de Valencia, Spain

   This file is part of SLEPc.
   SLEPc is distributed under a 2-clause BSD license (see LICENSE).
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
*/

static char help[] = "Lanczos SVD. Also illustrates the use of SVDSetBV().\n\n"
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
  Mat            A;
  SVD            svd;
  PetscInt       m=20,n,Istart,Iend,i,k=6,col[2];
  PetscScalar    value[] = { 1, 2 };
  PetscBool      flg,oneside=PETSC_FALSE;
  const char     *prefix;
  BV             U,V;
  Vec            u,v;

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
  PetscCall(MatCreateVecs(A,&v,&u));

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
         Create standalone BV objects to illustrate use of SVDSetBV()
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

  PetscCall(BVCreate(PETSC_COMM_WORLD,&U));
  PetscCall(PetscObjectSetName((PetscObject)U,"U"));
  PetscCall(BVSetSizesFromVec(U,u,k));
  PetscCall(BVSetFromOptions(U));
  PetscCall(BVCreate(PETSC_COMM_WORLD,&V));
  PetscCall(PetscObjectSetName((PetscObject)V,"V"));
  PetscCall(BVSetSizesFromVec(V,v,k));
  PetscCall(BVSetFromOptions(V));

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        Compute singular values
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

  PetscCall(SVDCreate(PETSC_COMM_WORLD,&svd));
  PetscCall(SVDSetBV(svd,V,U));
  PetscCall(SVDSetOptionsPrefix(svd,"check_"));
  PetscCall(SVDAppendOptionsPrefix(svd,"myprefix_"));
  PetscCall(SVDGetOptionsPrefix(svd,&prefix));
  PetscCall(PetscPrintf(PETSC_COMM_WORLD,"SVD prefix is currently: %s\n\n",prefix));
  PetscCall(PetscObjectSetName((PetscObject)svd,"SVD_solver"));

  PetscCall(SVDSetOperators(svd,A,NULL));
  PetscCall(SVDSetType(svd,SVDLANCZOS));
  PetscCall(SVDSetFromOptions(svd));

  PetscCall(PetscObjectTypeCompare((PetscObject)svd,SVDLANCZOS,&flg));
  if (flg) {
    PetscCall(SVDLanczosGetOneSide(svd,&oneside));
    PetscCall(PetscPrintf(PETSC_COMM_WORLD,"Running Lanczos %s\n\n",oneside?"(onesided)":""));
  }
  PetscCall(PetscObjectTypeCompare((PetscObject)svd,SVDTRLANCZOS,&flg));
  if (flg) {
    PetscCall(SVDTRLanczosGetOneSide(svd,&oneside));
    PetscCall(PetscPrintf(PETSC_COMM_WORLD,"Running thick-restart Lanczos %s\n\n",oneside?"(onesided)":""));
  }

  PetscCall(SVDSolve(svd));

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
                    Display solution and clean up
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  PetscCall(SVDErrorView(svd,SVD_ERROR_RELATIVE,PETSC_VIEWER_STDOUT_WORLD));
  PetscCall(BVDestroy(&U));
  PetscCall(BVDestroy(&V));
  PetscCall(VecDestroy(&u));
  PetscCall(VecDestroy(&v));
  PetscCall(SVDDestroy(&svd));
  PetscCall(MatDestroy(&A));
  PetscCall(SlepcFinalize());
  return 0;
}

/*TEST

   testset:
      args: -check_myprefix_svd_nsv 3
      requires: double
      test:
         suffix: 1
         args: -check_myprefix_svd_view_vectors ::ascii_info
      test:
         suffix: 2
         args: -check_myprefix_svd_type trlanczos -check_myprefix_svd_monitor -check_myprefix_svd_view_values ::ascii_matlab
         filter: sed -e "s/[0-9]\.[0-9]*e[+-]\([0-9]*\)/removed/g"
      test:
         suffix: 3
         args: -m 22 -n 20

TEST*/
