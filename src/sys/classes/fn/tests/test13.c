/*
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
   SLEPc - Scalable Library for Eigenvalue Problem Computations
   Copyright (c) 2002-2021, Universitat Politecnica de Valencia, Spain

   This file is part of SLEPc.
   SLEPc is distributed under a 2-clause BSD license (see LICENSE).
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
*/

static char help[] = "Test matrix logarithm.\n\n";

#include <slepcfn.h>

/*
   Compute matrix logarithm B = logm(A)
 */
PetscErrorCode TestMatLog(FN fn,Mat A,PetscViewer viewer,PetscBool verbose,PetscBool inplace)
{
  PetscBool      set,flg;
  PetscScalar    tau,eta;
  PetscInt       n;
  Mat            F,R;
  Vec            v,f0;
  FN             fnexp;
  PetscReal      nrm;

  PetscFunctionBeginUser;
  CHKERRQ(MatGetSize(A,&n,NULL));
  CHKERRQ(MatCreateSeqDense(PETSC_COMM_SELF,n,n,NULL,&F));
  CHKERRQ(PetscObjectSetName((PetscObject)F,"F"));
  CHKERRQ(MatCreateSeqDense(PETSC_COMM_SELF,n,n,NULL,&R));
  CHKERRQ(PetscObjectSetName((PetscObject)R,"R"));
  CHKERRQ(FNGetScale(fn,&tau,&eta));
  /* compute matrix logarithm */
  if (inplace) {
    CHKERRQ(MatCopy(A,F,SAME_NONZERO_PATTERN));
    CHKERRQ(MatIsHermitianKnown(A,&set,&flg));
    if (set && flg) CHKERRQ(MatSetOption(F,MAT_HERMITIAN,PETSC_TRUE));
    CHKERRQ(FNEvaluateFunctionMat(fn,F,NULL));
  } else CHKERRQ(FNEvaluateFunctionMat(fn,A,F));
  if (verbose) {
    CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"Matrix A - - - - - - - -\n"));
    CHKERRQ(MatView(A,viewer));
    CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"Computed logm(A) - - - - - - -\n"));
    CHKERRQ(MatView(F,viewer));
  }
  /* check error ||expm(F)-A||_F */
  CHKERRQ(FNCreate(PETSC_COMM_WORLD,&fnexp));
  CHKERRQ(FNSetType(fnexp,FNEXP));
  CHKERRQ(MatCopy(F,R,SAME_NONZERO_PATTERN));
  if (eta!=1.0) CHKERRQ(MatScale(R,1.0/eta));
  CHKERRQ(FNEvaluateFunctionMat(fnexp,R,NULL));
  CHKERRQ(FNDestroy(&fnexp));
  CHKERRQ(MatAXPY(R,-tau,A,SAME_NONZERO_PATTERN));
  CHKERRQ(MatNorm(R,NORM_FROBENIUS,&nrm));
  if (nrm<100*PETSC_MACHINE_EPSILON) CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"||expm(F)-A||_F < 100*eps\n"));
  else CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"||expm(F)-A||_F = %g\n",(double)nrm));
  /* check FNEvaluateFunctionMatVec() */
  CHKERRQ(MatCreateVecs(A,&v,&f0));
  CHKERRQ(MatGetColumnVector(F,f0,0));
  CHKERRQ(FNEvaluateFunctionMatVec(fn,A,v));
  CHKERRQ(VecAXPY(v,-1.0,f0));
  CHKERRQ(VecNorm(v,NORM_2,&nrm));
  if (nrm>100*PETSC_MACHINE_EPSILON) CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"Warning: the norm of f(A)*e_1-v is %g\n",(double)nrm));
  CHKERRQ(MatDestroy(&F));
  CHKERRQ(MatDestroy(&R));
  CHKERRQ(VecDestroy(&v));
  CHKERRQ(VecDestroy(&f0));
  PetscFunctionReturn(0);
}

int main(int argc,char **argv)
{
  PetscErrorCode ierr;
  FN             fn;
  Mat            A;
  PetscInt       i,j,n=10;
  PetscScalar    *As;
  PetscViewer    viewer;
  PetscBool      verbose,inplace,random,triang;

  ierr = SlepcInitialize(&argc,&argv,(char*)0,help);if (ierr) return ierr;
  CHKERRQ(PetscOptionsGetInt(NULL,NULL,"-n",&n,NULL));
  CHKERRQ(PetscOptionsHasName(NULL,NULL,"-verbose",&verbose));
  CHKERRQ(PetscOptionsHasName(NULL,NULL,"-inplace",&inplace));
  CHKERRQ(PetscOptionsHasName(NULL,NULL,"-random",&random));
  CHKERRQ(PetscOptionsHasName(NULL,NULL,"-triang",&triang));
  CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"Matrix logarithm, n=%" PetscInt_FMT ".\n",n));

  /* Create logarithm function object */
  CHKERRQ(FNCreate(PETSC_COMM_WORLD,&fn));
  CHKERRQ(FNSetType(fn,FNLOG));
  CHKERRQ(FNSetFromOptions(fn));

  /* Set up viewer */
  CHKERRQ(PetscViewerASCIIGetStdout(PETSC_COMM_WORLD,&viewer));
  CHKERRQ(FNView(fn,viewer));
  if (verbose) CHKERRQ(PetscViewerPushFormat(viewer,PETSC_VIEWER_ASCII_MATLAB));

  /* Create matrices */
  CHKERRQ(MatCreateSeqDense(PETSC_COMM_SELF,n,n,NULL,&A));
  CHKERRQ(PetscObjectSetName((PetscObject)A,"A"));

  if (random) CHKERRQ(MatSetRandom(A,NULL));
  else {
    /* Fill A with a non-symmetric Toeplitz matrix */
    CHKERRQ(MatDenseGetArray(A,&As));
    for (i=0;i<n;i++) As[i+i*n]=2.0;
    for (j=1;j<3;j++) {
      for (i=0;i<n-j;i++) {
        As[i+(i+j)*n]=1.0;
        if (!triang) As[(i+j)+i*n]=-1.0;
      }
    }
    As[(n-1)*n] = -5.0;
    As[0] = 2.01;
    CHKERRQ(MatDenseRestoreArray(A,&As));
  }
  CHKERRQ(TestMatLog(fn,A,viewer,verbose,inplace));

  CHKERRQ(MatDestroy(&A));
  CHKERRQ(FNDestroy(&fn));
  ierr = SlepcFinalize();
  return ierr;
}

/*TEST

   testset:
      filter: grep -v "computing matrix functions"
      output_file: output/test13_1.out
      test:
         suffix: 1
         args: -fn_scale .04,2 -n 75
         requires: c99_complex !__float128
      test:
         suffix: 1_triang
         args: -fn_scale .04,2 -n 75 -triang
         requires: c99_complex !__float128
      test:
         suffix: 1_random
         args: -fn_scale .02,2 -n 75 -random
         requires: complex
         filter_output: sed -e 's/04/02/'

TEST*/
