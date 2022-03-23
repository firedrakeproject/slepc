/*
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
   SLEPc - Scalable Library for Eigenvalue Problem Computations
   Copyright (c) 2002-2021, Universitat Politecnica de Valencia, Spain

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
  PetscErrorCode ierr;

  ierr = SlepcInitialize(&argc,&argv,(char*)0,help);if (ierr) return ierr;
  CHKERRQ(PetscOptionsGetInt(NULL,NULL,"-m",&m,NULL));
  CHKERRQ(PetscOptionsGetInt(NULL,NULL,"-n",&n,&flg));
  if (!flg) n=m+2;
  CHKERRQ(PetscOptionsGetInt(NULL,NULL,"-k",&k,NULL));
  CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"\nRectangular bidiagonal matrix, m=%" PetscInt_FMT " n=%" PetscInt_FMT "\n\n",m,n));

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        Generate the matrix
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

  CHKERRQ(MatCreate(PETSC_COMM_WORLD,&A));
  CHKERRQ(MatSetSizes(A,PETSC_DECIDE,PETSC_DECIDE,m,n));
  CHKERRQ(MatSetFromOptions(A));
  CHKERRQ(MatSetUp(A));
  CHKERRQ(MatGetOwnershipRange(A,&Istart,&Iend));
  for (i=Istart;i<Iend;i++) {
    col[0]=i; col[1]=i+1;
    if (i<n-1) CHKERRQ(MatSetValues(A,1,&i,2,col,value,INSERT_VALUES));
    else if (i==n-1) CHKERRQ(MatSetValue(A,i,col[0],value[0],INSERT_VALUES));
  }
  CHKERRQ(MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY));
  CHKERRQ(MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY));
  CHKERRQ(MatCreateVecs(A,&v,&u));

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
         Create standalone BV objects to illustrate use of SVDSetBV()
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

  CHKERRQ(BVCreate(PETSC_COMM_WORLD,&U));
  CHKERRQ(PetscObjectSetName((PetscObject)U,"U"));
  CHKERRQ(BVSetSizesFromVec(U,u,k));
  CHKERRQ(BVSetFromOptions(U));
  CHKERRQ(BVCreate(PETSC_COMM_WORLD,&V));
  CHKERRQ(PetscObjectSetName((PetscObject)V,"V"));
  CHKERRQ(BVSetSizesFromVec(V,v,k));
  CHKERRQ(BVSetFromOptions(V));

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        Compute singular values
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

  CHKERRQ(SVDCreate(PETSC_COMM_WORLD,&svd));
  CHKERRQ(SVDSetBV(svd,V,U));
  CHKERRQ(SVDSetOptionsPrefix(svd,"check_"));
  CHKERRQ(SVDAppendOptionsPrefix(svd,"myprefix_"));
  CHKERRQ(SVDGetOptionsPrefix(svd,&prefix));
  CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"SVD prefix is currently: %s\n\n",prefix));
  CHKERRQ(PetscObjectSetName((PetscObject)svd,"SVD_solver"));

  CHKERRQ(SVDSetOperators(svd,A,NULL));
  CHKERRQ(SVDSetType(svd,SVDLANCZOS));
  CHKERRQ(SVDSetFromOptions(svd));

  CHKERRQ(PetscObjectTypeCompare((PetscObject)svd,SVDLANCZOS,&flg));
  if (flg) {
    CHKERRQ(SVDLanczosGetOneSide(svd,&oneside));
    CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"Running Lanczos %s\n\n",oneside?"(onesided)":""));
  }
  CHKERRQ(PetscObjectTypeCompare((PetscObject)svd,SVDTRLANCZOS,&flg));
  if (flg) {
    CHKERRQ(SVDTRLanczosGetOneSide(svd,&oneside));
    CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"Running thick-restart Lanczos %s\n\n",oneside?"(onesided)":""));
  }

  CHKERRQ(SVDSolve(svd));

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
                    Display solution and clean up
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  CHKERRQ(SVDErrorView(svd,SVD_ERROR_RELATIVE,PETSC_VIEWER_STDOUT_WORLD));
  CHKERRQ(BVDestroy(&U));
  CHKERRQ(BVDestroy(&V));
  CHKERRQ(VecDestroy(&u));
  CHKERRQ(VecDestroy(&v));
  CHKERRQ(SVDDestroy(&svd));
  CHKERRQ(MatDestroy(&A));
  ierr = SlepcFinalize();
  return ierr;
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
