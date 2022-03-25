/*
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
   SLEPc - Scalable Library for Eigenvalue Problem Computations
   Copyright (c) 2002-2021, Universitat Politecnica de Valencia, Spain

   This file is part of SLEPc.
   SLEPc is distributed under a 2-clause BSD license (see LICENSE).
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
*/

static char help[] = "Test an SVD problem with more columns than rows.\n\n"
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
  Mat                  A,B;
  SVD                  svd;
  SVDConv              conv;
  SVDStop              stop;
  SVDWhich             which;
  SVDProblemType       ptype;
  SVDConvergedReason   reason;
  PetscInt             m=20,n,Istart,Iend,i,col[2],its;
  PetscScalar          value[] = { 1, 2 };
  PetscBool            flg,tmode;
  PetscViewerAndFormat *vf;
  const char           *ctest[] = { "absolute", "relative to the singular value", "user-defined" };
  const char           *stest[] = { "basic", "user-defined" };

  CHKERRQ(SlepcInitialize(&argc,&argv,(char*)0,help));

  CHKERRQ(PetscOptionsGetInt(NULL,NULL,"-m",&m,NULL));
  CHKERRQ(PetscOptionsGetInt(NULL,NULL,"-n",&n,&flg));
  if (!flg) n=m+2;
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

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        Compute singular values
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

  CHKERRQ(SVDCreate(PETSC_COMM_WORLD,&svd));
  CHKERRQ(SVDSetOperators(svd,A,NULL));

  /* test some interface functions */
  CHKERRQ(SVDGetOperators(svd,&B,NULL));
  CHKERRQ(MatView(B,PETSC_VIEWER_STDOUT_WORLD));
  CHKERRQ(SVDSetConvergenceTest(svd,SVD_CONV_ABS));
  CHKERRQ(SVDSetStoppingTest(svd,SVD_STOP_BASIC));
  /* test monitors */
  CHKERRQ(PetscViewerAndFormatCreate(PETSC_VIEWER_STDOUT_WORLD,PETSC_VIEWER_DEFAULT,&vf));
  CHKERRQ(SVDMonitorSet(svd,(PetscErrorCode (*)(SVD,PetscInt,PetscInt,PetscReal*,PetscReal*,PetscInt,void*))SVDMonitorFirst,vf,(PetscErrorCode (*)(void**))PetscViewerAndFormatDestroy));
  /* CHKERRQ(SVDMonitorCancel(svd)); */
  CHKERRQ(SVDSetFromOptions(svd));

  /* query properties and print them */
  CHKERRQ(SVDGetProblemType(svd,&ptype));
  CHKERRQ(PetscPrintf(PETSC_COMM_WORLD," Problem type = %d",(int)ptype));
  CHKERRQ(SVDIsGeneralized(svd,&flg));
  if (flg) CHKERRQ(PetscPrintf(PETSC_COMM_WORLD," generalized"));
  CHKERRQ(SVDGetImplicitTranspose(svd,&tmode));
  CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"\n Transpose mode is %s\n",tmode?"implicit":"explicit"));
  CHKERRQ(SVDGetConvergenceTest(svd,&conv));
  CHKERRQ(PetscPrintf(PETSC_COMM_WORLD," Convergence test is %s\n",ctest[conv]));
  CHKERRQ(SVDGetStoppingTest(svd,&stop));
  CHKERRQ(PetscPrintf(PETSC_COMM_WORLD," Stopping test is %s\n",stest[stop]));
  CHKERRQ(SVDGetWhichSingularTriplets(svd,&which));
  CHKERRQ(PetscPrintf(PETSC_COMM_WORLD," Which = %s\n",which?"smallest":"largest"));

  /* call the solver */
  CHKERRQ(SVDSolve(svd));
  CHKERRQ(SVDGetConvergedReason(svd,&reason));
  CHKERRQ(SVDGetIterationNumber(svd,&its));
  CHKERRQ(PetscPrintf(PETSC_COMM_WORLD," Finished - converged reason = %d\n",(int)reason));
  /* CHKERRQ(PetscPrintf(PETSC_COMM_WORLD," its = %" PetscInt_FMT "\n",its)); */

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
                    Display solution and clean up
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  CHKERRQ(SVDErrorView(svd,SVD_ERROR_RELATIVE,PETSC_VIEWER_STDOUT_WORLD));
  CHKERRQ(SVDDestroy(&svd));
  CHKERRQ(MatDestroy(&A));
  CHKERRQ(SlepcFinalize());
  return 0;
}

/*TEST

   testset:
      args: -svd_monitor_cancel
      filter: grep -v "Transpose mode"
      output_file: output/test4_1.out
      test:
         suffix: 1_lanczos
         args: -svd_type lanczos
      test:
         suffix: 1_randomized
         args: -svd_type randomized
      test:
         suffix: 1_trlanczos
         args: -svd_type trlanczos -svd_ncv 12 -svd_trlanczos_restart 0.6
      test:
         suffix: 1_cross
         args: -svd_type cross
      test:
         suffix: 1_cross_exp
         args: -svd_type cross -svd_cross_explicitmatrix
      test:
         suffix: 1_cross_exp_imp
         args: -svd_type cross -svd_cross_explicitmatrix -svd_implicittranspose
         requires: !complex
      test:
         suffix: 1_cyclic
         args: -svd_type cyclic
      test:
         suffix: 1_cyclic_imp
         args: -svd_type cyclic -svd_implicittranspose
      test:
         suffix: 1_cyclic_exp
         args: -svd_type cyclic -svd_cyclic_explicitmatrix
      test:
         suffix: 1_lapack
         args: -svd_type lapack
      test:
         suffix: 1_scalapack
         args: -svd_type scalapack
         requires: scalapack

   testset:
      args: -svd_monitor_cancel  -mat_type aijcusparse
      requires: cuda !single
      filter: grep -v "Transpose mode" | sed -e "s/seqaijcusparse/seqaij/"
      output_file: output/test4_1.out
      test:
         suffix: 2_cuda_lanczos
         args: -svd_type lanczos
      test:
         suffix: 2_cuda_trlanczos
         args: -svd_type trlanczos -svd_ncv 12
      test:
         suffix: 2_cuda_cross
         args: -svd_type cross

   test:
      suffix: 3
      nsize: 2
      args: -svd_type trlanczos -svd_ncv 14 -svd_monitor_cancel -ds_parallel synchronized

TEST*/
