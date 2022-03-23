/*
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
   SLEPc - Scalable Library for Eigenvalue Problem Computations
   Copyright (c) 2002-2021, Universitat Politecnica de Valencia, Spain

   This file is part of SLEPc.
   SLEPc is distributed under a 2-clause BSD license (see LICENSE).
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
*/

static char help[] = "Test matrix function evaluation via diagonalization.\n\n";

#include <slepcfn.h>

int main(int argc,char **argv)
{
  PetscErrorCode ierr;
  FN             fn;
  Mat            A,F,G;
  PetscInt       i,j,n=10;
  PetscReal      nrm;
  PetscScalar    *As,alpha,beta;
  PetscViewer    viewer;
  PetscBool      verbose;

  ierr = SlepcInitialize(&argc,&argv,(char*)0,help);if (ierr) return ierr;
  CHKERRQ(PetscOptionsGetInt(NULL,NULL,"-n",&n,NULL));
  CHKERRQ(PetscOptionsHasName(NULL,NULL,"-verbose",&verbose));
  CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"Matrix function of symmetric/Hermitian matrix, n=%" PetscInt_FMT ".\n",n));

  /* Create function object */
  CHKERRQ(FNCreate(PETSC_COMM_WORLD,&fn));
  CHKERRQ(FNSetType(fn,FNEXP));   /* default to exponential */
#if defined(PETSC_USE_COMPLEX)
  alpha = PetscCMPLX(0.3,0.8);
  beta  = PetscCMPLX(1.1,-0.1);
#else
  alpha = 0.3;
  beta  = 1.1;
#endif
  CHKERRQ(FNSetScale(fn,alpha,beta));
  CHKERRQ(FNSetFromOptions(fn));

  /* Set up viewer */
  CHKERRQ(PetscViewerASCIIGetStdout(PETSC_COMM_WORLD,&viewer));
  if (verbose) CHKERRQ(PetscViewerPushFormat(viewer,PETSC_VIEWER_ASCII_MATLAB));

  /* Create a symmetric/Hermitian Toeplitz matrix */
  CHKERRQ(MatCreateSeqDense(PETSC_COMM_SELF,n,n,NULL,&A));
  CHKERRQ(PetscObjectSetName((PetscObject)A,"A"));
  CHKERRQ(MatDenseGetArray(A,&As));
  for (i=0;i<n;i++) As[i+i*n]=2.0;
  for (j=1;j<3;j++) {
    for (i=0;i<n-j;i++) {
#if defined(PETSC_USE_COMPLEX)
      As[i+(i+j)*n]=PetscCMPLX(1.0,0.1); As[(i+j)+i*n]=PetscCMPLX(1.0,-0.1);
#else
      As[i+(i+j)*n]=0.5; As[(i+j)+i*n]=0.5;
#endif
    }
  }
  CHKERRQ(MatDenseRestoreArray(A,&As));
  if (verbose) {
    CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"Matrix A - - - - - - - -\n"));
    CHKERRQ(MatView(A,viewer));
  }

  /* compute matrix function */
  CHKERRQ(MatCreateSeqDense(PETSC_COMM_SELF,n,n,NULL,&F));
  CHKERRQ(PetscObjectSetName((PetscObject)F,"F"));
  CHKERRQ(FNEvaluateFunctionMat(fn,A,F));
  if (verbose) {
    CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"Computed f(A) - - - - - - -\n"));
    CHKERRQ(MatView(F,viewer));
  }

  /* Repeat with MAT_HERMITIAN flag set */
  CHKERRQ(MatSetOption(A,MAT_HERMITIAN,PETSC_TRUE));
  CHKERRQ(MatCreateSeqDense(PETSC_COMM_SELF,n,n,NULL,&G));
  CHKERRQ(PetscObjectSetName((PetscObject)G,"G"));
  CHKERRQ(FNEvaluateFunctionMat(fn,A,G));
  if (verbose) {
    CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"Computed f(A) symm - - - - - - -\n"));
    CHKERRQ(MatView(G,viewer));
  }

  /* compare the two results */
  CHKERRQ(MatAXPY(F,-1.0,G,SAME_NONZERO_PATTERN));
  CHKERRQ(MatNorm(F,NORM_FROBENIUS,&nrm));
  if (nrm>100*PETSC_MACHINE_EPSILON) CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"Warning: the norm of F-G is %g\n",(double)nrm));
  else CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"Computed results match.\n"));

  CHKERRQ(MatDestroy(&A));
  CHKERRQ(MatDestroy(&F));
  CHKERRQ(MatDestroy(&G));
  CHKERRQ(FNDestroy(&fn));
  ierr = SlepcFinalize();
  return ierr;
}

/*TEST

   test:
      suffix: 1
      nsize: 1
      args: -fn_type {{exp sqrt}shared output}
      output_file: output/test12_1.out

   test:
      suffix: 1_rational
      nsize: 1
      args: -fn_type rational -fn_rational_numerator 2,-1.5 -fn_rational_denominator 1,0.8
      output_file: output/test12_1.out

TEST*/
