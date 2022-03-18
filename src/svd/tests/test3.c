/*
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
   SLEPc - Scalable Library for Eigenvalue Problem Computations
   Copyright (c) 2002-2021, Universitat Politecnica de Valencia, Spain

   This file is part of SLEPc.
   SLEPc is distributed under a 2-clause BSD license (see LICENSE).
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
*/

static char help[] = "Test SVD with user-provided initial vectors.\n\n"
  "The command line options are:\n"
  "  -n <n>, where <n> = row dimension.\n"
  "  -m <m>, where <m> = column dimension.\n\n";

#include <slepcsvd.h>

/*
   This example computes the singular values of a rectangular nxm Grcar matrix:

              |  1  1  1  1               |
              | -1  1  1  1  1            |
              |    -1  1  1  1  1         |
          A = |       .  .  .  .  .       |
              |          .  .  .  .  .    |
              |            -1  1  1  1  1 |
              |               -1  1  1  1 |

 */

int main(int argc,char **argv)
{
  Mat            A;               /* Grcar matrix */
  SVD            svd;             /* singular value solver context */
  Vec            v0,w0;           /* initial vectors */
  Vec            *U,*V;
  PetscInt       N=35,M=30,Istart,Iend,i,col[5],nconv;
  PetscScalar    value[] = { -1, 1, 1, 1, 1 };
  PetscReal      lev1=0.0,lev2=0.0,tol=PETSC_SMALL;
  PetscBool      skiporth=PETSC_FALSE;
  PetscErrorCode ierr;

  ierr = SlepcInitialize(&argc,&argv,(char*)0,help);if (ierr) return ierr;
  CHKERRQ(PetscOptionsGetInt(NULL,NULL,"-n",&N,NULL));
  CHKERRQ(PetscOptionsGetInt(NULL,NULL,"-m",&M,NULL));
  CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"\nSVD of a rectangular Grcar matrix, %" PetscInt_FMT "x%" PetscInt_FMT "\n\n",N,M));
  CHKERRQ(PetscOptionsGetBool(NULL,NULL,"-skiporth",&skiporth,NULL));

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        Generate the matrix
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

  CHKERRQ(MatCreate(PETSC_COMM_WORLD,&A));
  CHKERRQ(MatSetSizes(A,PETSC_DECIDE,PETSC_DECIDE,N,M));
  CHKERRQ(MatSetFromOptions(A));
  CHKERRQ(MatSetUp(A));

  CHKERRQ(MatGetOwnershipRange(A,&Istart,&Iend));
  for (i=Istart;i<Iend;i++) {
    col[0]=i-1; col[1]=i; col[2]=i+1; col[3]=i+2; col[4]=i+3;
    if (i==0) {
      CHKERRQ(MatSetValues(A,1,&i,PetscMin(4,M-i+1),col+1,value+1,INSERT_VALUES));
    } else {
      CHKERRQ(MatSetValues(A,1,&i,PetscMin(5,M-i+1),col,value,INSERT_VALUES));
    }
  }
  CHKERRQ(MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY));
  CHKERRQ(MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY));

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
             Create the SVD context and solve the problem
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

  CHKERRQ(SVDCreate(PETSC_COMM_WORLD,&svd));
  CHKERRQ(SVDSetOperators(svd,A,NULL));
  CHKERRQ(SVDSetFromOptions(svd));

  /*
     Set the initial vectors. This is optional, if not done the initial
     vectors are set to random values
  */
  CHKERRQ(MatCreateVecs(A,&v0,&w0));
  CHKERRQ(VecSet(v0,1.0));
  CHKERRQ(VecSet(w0,1.0));
  CHKERRQ(SVDSetInitialSpaces(svd,1,&v0,1,&w0));

  /*
     Compute solution
  */
  CHKERRQ(SVDSolve(svd));
  CHKERRQ(SVDErrorView(svd,SVD_ERROR_RELATIVE,NULL));

  /*
     Check orthonormality of computed singular vectors
  */
  CHKERRQ(SVDGetConverged(svd,&nconv));
  if (nconv>1) {
    CHKERRQ(VecDuplicateVecs(w0,nconv,&U));
    CHKERRQ(VecDuplicateVecs(v0,nconv,&V));
    for (i=0;i<nconv;i++) {
      CHKERRQ(SVDGetSingularTriplet(svd,i,NULL,U[i],V[i]));
    }
    if (!skiporth) {
      CHKERRQ(VecCheckOrthonormality(U,nconv,NULL,nconv,NULL,NULL,&lev1));
      CHKERRQ(VecCheckOrthonormality(V,nconv,NULL,nconv,NULL,NULL,&lev2));
    }
    if (lev1+lev2<20*tol) {
      CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"Level of orthogonality below the tolerance\n"));
    } else {
      CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"Level of orthogonality: %g (U) %g (V)\n",(double)lev1,(double)lev2));
    }
    CHKERRQ(VecDestroyVecs(nconv,&U));
    CHKERRQ(VecDestroyVecs(nconv,&V));
  }

  /*
     Free work space
  */
  CHKERRQ(VecDestroy(&v0));
  CHKERRQ(VecDestroy(&w0));
  CHKERRQ(SVDDestroy(&svd));
  CHKERRQ(MatDestroy(&A));
  ierr = SlepcFinalize();
  return ierr;
}

/*TEST

   testset:
      args: -svd_nsv 4
      output_file: output/test3_1.out
      filter: sed -e "s/22176/22175/" | sed -e "s/21798/21797/" | sed -e "s/16826/16825/" | sed -e "s/15129/15128/"
      test:
         suffix: 1_lanczos
         args: -svd_type lanczos
      test:
         suffix: 1_lanczos_one
         args: -svd_type lanczos -svd_lanczos_oneside
      test:
         suffix: 1_trlanczos
         args: -svd_type trlanczos -svd_trlanczos_locking {{0 1}}
      test:
         suffix: 1_trlanczos_one
         args: -svd_type trlanczos -svd_trlanczos_oneside
      test:
         suffix: 1_trlanczos_one_mgs
         args: -svd_type trlanczos -svd_trlanczos_oneside -bv_orthog_type mgs
      test:
         suffix: 1_trlanczos_one_always
         args: -svd_type trlanczos -svd_trlanczos_oneside -bv_orthog_refine always
      test:
         suffix: 1_cross
         args: -svd_type cross
      test:
         suffix: 1_cross_exp
         args: -svd_type cross -svd_cross_explicitmatrix
      test:
         suffix: 1_cyclic
         args: -svd_type cyclic
         requires: !__float128
      test:
         suffix: 1_cyclic_exp
         args: -svd_type cyclic -svd_cyclic_explicitmatrix
         requires: !__float128
      test:
         suffix: 1_lapack
         args: -svd_type lapack
      test:
         suffix: 1_randomized
         args: -svd_type randomized
      test:
         suffix: 1_primme
         args: -svd_type primme
         requires: primme

   testset:
      args: -svd_implicittranspose -svd_nsv 4 -svd_tol 1e-5
      output_file: output/test3_1.out
      filter: sed -e "s/22176/22175/" | sed -e "s/21798/21797/" | sed -e "s/16826/16825/" | sed -e "s/15129/15128/"
      test:
         suffix: 2_lanczos
         args: -svd_type lanczos -svd_conv_norm
      test:
         suffix: 2_lanczos_one
         args: -svd_type lanczos -svd_lanczos_oneside
      test:
         suffix: 2_trlanczos
         args: -svd_type trlanczos
      test:
         suffix: 2_trlanczos_one
         args: -svd_type trlanczos -svd_trlanczos_oneside
      test:
         suffix: 2_trlanczos_one_mgs
         args: -svd_type trlanczos -svd_trlanczos_oneside -bv_orthog_type mgs
      test:
         suffix: 2_trlanczos_one_always
         args: -svd_type trlanczos -svd_trlanczos_oneside -bv_orthog_refine always
      test:
         suffix: 2_cross
         args: -svd_type cross
      test:
         suffix: 2_cross_exp
         args: -svd_type cross -svd_cross_explicitmatrix
         requires: !complex
      test:
         suffix: 2_cyclic
         args: -svd_type cyclic -svd_tol 1e-8
         requires: double
      test:
         suffix: 2_lapack
         args: -svd_type lapack
      test:
         suffix: 2_randomized
         args: -svd_type randomized

   testset:
      args: -svd_nsv 4 -mat_type aijcusparse
      requires: cuda
      output_file: output/test3_1.out
      filter: sed -e "s/22176/22175/" | sed -e "s/21798/21797/" | sed -e "s/16826/16825/" | sed -e "s/15129/15128/"
      test:
         suffix: 3_cuda_lanczos
         args: -svd_type lanczos
      test:
         suffix: 3_cuda_lanczos_one
         args: -svd_type lanczos -svd_lanczos_oneside
      test:
         suffix: 3_cuda_trlanczos
         args: -svd_type trlanczos
      test:
         suffix: 3_cuda_trlanczos_one
         args: -svd_type trlanczos -svd_trlanczos_oneside
      test:
         suffix: 3_cuda_trlanczos_one_mgs
         args: -svd_type trlanczos -svd_trlanczos_oneside -bv_orthog_type mgs
      test:
         suffix: 3_cuda_trlanczos_one_always
         args: -svd_type trlanczos -svd_trlanczos_oneside -bv_orthog_refine always
      test:
         suffix: 3_cuda_cross
         args: -svd_type cross
      test:
         suffix: 3_cuda_cyclic
         args: -svd_type cyclic
      test:
         suffix: 3_cuda_cyclic_exp
         args: -svd_type cyclic -svd_cyclic_explicitmatrix
      test:
         suffix: 3_cuda_randomized
         args: -svd_type randomized

   test:
      suffix: 4
      args: -svd_type lapack -svd_nsv 4
      output_file: output/test3_1.out
      nsize: 2

   test:
      suffix: 5
      args: -svd_nsv 4 -svd_view_values draw -svd_monitor draw::draw_lg
      requires: x
      output_file: output/test3_1.out

TEST*/
