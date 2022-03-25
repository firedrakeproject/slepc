/*
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
   SLEPc - Scalable Library for Eigenvalue Problem Computations
   Copyright (c) 2002-2021, Universitat Politecnica de Valencia, Spain

   This file is part of SLEPc.
   SLEPc is distributed under a 2-clause BSD license (see LICENSE).
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
*/

static char help[] = "Test matrix exponential.\n\n";

#include <slepcfn.h>

/*
   Compute matrix exponential B = expm(A)
 */
PetscErrorCode TestMatExp(FN fn,Mat A,PetscViewer viewer,PetscBool verbose,PetscBool inplace,PetscBool checkerror)
{
  PetscScalar    tau,eta;
  PetscBool      set,flg;
  PetscInt       n;
  Mat            F,R,Finv,Acopy;
  Vec            v,f0;
  FN             finv;
  PetscReal      nrm,nrmf;

  PetscFunctionBeginUser;
  CHKERRQ(MatGetSize(A,&n,NULL));
  CHKERRQ(MatDuplicate(A,MAT_DO_NOT_COPY_VALUES,&F));
  CHKERRQ(PetscObjectSetName((PetscObject)F,"F"));
  /* compute matrix exponential */
  if (inplace) {
    CHKERRQ(MatCopy(A,F,SAME_NONZERO_PATTERN));
    CHKERRQ(MatIsHermitianKnown(A,&set,&flg));
    if (set && flg) CHKERRQ(MatSetOption(F,MAT_HERMITIAN,PETSC_TRUE));
    CHKERRQ(FNEvaluateFunctionMat(fn,F,NULL));
  } else {
    CHKERRQ(MatDuplicate(A,MAT_COPY_VALUES,&Acopy));
    CHKERRQ(FNEvaluateFunctionMat(fn,A,F));
    /* check that A has not been modified */
    CHKERRQ(MatAXPY(Acopy,-1.0,A,SAME_NONZERO_PATTERN));
    CHKERRQ(MatNorm(Acopy,NORM_1,&nrm));
    if (nrm>100*PETSC_MACHINE_EPSILON) CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"Warning: the input matrix has changed by %g\n",(double)nrm));
    CHKERRQ(MatDestroy(&Acopy));
  }
  if (verbose) {
    CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"Matrix A - - - - - - - -\n"));
    CHKERRQ(MatView(A,viewer));
    CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"Computed expm(A) - - - - - - -\n"));
    CHKERRQ(MatView(F,viewer));
  }
  /* print matrix norm for checking */
  CHKERRQ(MatNorm(F,NORM_1,&nrmf));
  CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"The 1-norm of f(A) is %g\n",(double)nrmf));
  if (checkerror) {
    CHKERRQ(MatDuplicate(A,MAT_DO_NOT_COPY_VALUES,&Finv));
    CHKERRQ(PetscObjectSetName((PetscObject)Finv,"Finv"));
    CHKERRQ(FNGetScale(fn,&tau,&eta));
    /* compute inverse exp(-tau*A)/eta */
    CHKERRQ(FNCreate(PETSC_COMM_WORLD,&finv));
    CHKERRQ(FNSetType(finv,FNEXP));
    CHKERRQ(FNSetFromOptions(finv));
    CHKERRQ(FNSetScale(finv,-tau,1.0/eta));
    if (inplace) {
      CHKERRQ(MatCopy(A,Finv,SAME_NONZERO_PATTERN));
      CHKERRQ(MatIsHermitianKnown(A,&set,&flg));
      if (set && flg) CHKERRQ(MatSetOption(Finv,MAT_HERMITIAN,PETSC_TRUE));
      CHKERRQ(FNEvaluateFunctionMat(finv,Finv,NULL));
    } else CHKERRQ(FNEvaluateFunctionMat(finv,A,Finv));
    CHKERRQ(FNDestroy(&finv));
    /* check error ||F*Finv-I||_F */
    CHKERRQ(MatMatMult(F,Finv,MAT_INITIAL_MATRIX,PETSC_DEFAULT,&R));
    CHKERRQ(MatShift(R,-1.0));
    CHKERRQ(MatNorm(R,NORM_FROBENIUS,&nrm));
    if (nrm<100*PETSC_MACHINE_EPSILON) CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"||exp(A)*exp(-A)-I||_F < 100*eps\n"));
    else CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"||exp(A)*exp(-A)-I||_F = %g\n",(double)nrm));
    CHKERRQ(MatDestroy(&R));
    CHKERRQ(MatDestroy(&Finv));
  }
  /* check FNEvaluateFunctionMatVec() */
  CHKERRQ(MatCreateVecs(A,&v,&f0));
  CHKERRQ(MatGetColumnVector(F,f0,0));
  CHKERRQ(FNEvaluateFunctionMatVec(fn,A,v));
  CHKERRQ(VecAXPY(v,-1.0,f0));
  CHKERRQ(VecNorm(v,NORM_2,&nrm));
  if (nrm/nrmf>100*PETSC_MACHINE_EPSILON) CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"Warning: the norm of f(A)*e_1-v is %g\n",(double)nrm));
  CHKERRQ(MatDestroy(&F));
  CHKERRQ(VecDestroy(&v));
  CHKERRQ(VecDestroy(&f0));
  PetscFunctionReturn(0);
}

int main(int argc,char **argv)
{
  FN             fn;
  Mat            A;
  PetscInt       i,j,n=10;
  PetscScalar    *As;
  PetscViewer    viewer;
  PetscBool      verbose,inplace,checkerror;

  CHKERRQ(SlepcInitialize(&argc,&argv,(char*)0,help));
  CHKERRQ(PetscOptionsGetInt(NULL,NULL,"-n",&n,NULL));
  CHKERRQ(PetscOptionsHasName(NULL,NULL,"-verbose",&verbose));
  CHKERRQ(PetscOptionsHasName(NULL,NULL,"-inplace",&inplace));
  CHKERRQ(PetscOptionsHasName(NULL,NULL,"-checkerror",&checkerror));
  CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"Matrix exponential, n=%" PetscInt_FMT ".\n",n));

  /* Create exponential function object */
  CHKERRQ(FNCreate(PETSC_COMM_WORLD,&fn));
  CHKERRQ(FNSetType(fn,FNEXP));
  CHKERRQ(FNSetFromOptions(fn));

  /* Set up viewer */
  CHKERRQ(PetscViewerASCIIGetStdout(PETSC_COMM_WORLD,&viewer));
  CHKERRQ(FNView(fn,viewer));
  if (verbose) CHKERRQ(PetscViewerPushFormat(viewer,PETSC_VIEWER_ASCII_MATLAB));

  /* Create matrices */
  CHKERRQ(MatCreateSeqDense(PETSC_COMM_SELF,n,n,NULL,&A));
  CHKERRQ(PetscObjectSetName((PetscObject)A,"A"));

  /* Fill A with a symmetric Toeplitz matrix */
  CHKERRQ(MatDenseGetArray(A,&As));
  for (i=0;i<n;i++) As[i+i*n]=2.0;
  for (j=1;j<3;j++) {
    for (i=0;i<n-j;i++) { As[i+(i+j)*n]=1.0; As[(i+j)+i*n]=1.0; }
  }
  CHKERRQ(MatDenseRestoreArray(A,&As));
  CHKERRQ(MatSetOption(A,MAT_HERMITIAN,PETSC_TRUE));
  CHKERRQ(TestMatExp(fn,A,viewer,verbose,inplace,checkerror));

  /* Repeat with non-symmetric A */
  CHKERRQ(MatDenseGetArray(A,&As));
  for (j=1;j<3;j++) {
    for (i=0;i<n-j;i++) { As[(i+j)+i*n]=-1.0; }
  }
  CHKERRQ(MatDenseRestoreArray(A,&As));
  CHKERRQ(MatSetOption(A,MAT_HERMITIAN,PETSC_FALSE));
  CHKERRQ(TestMatExp(fn,A,viewer,verbose,inplace,checkerror));

  CHKERRQ(MatDestroy(&A));
  CHKERRQ(FNDestroy(&fn));
  CHKERRQ(SlepcFinalize());
  return 0;
}

/*TEST

   testset:
      filter: grep -v "computing matrix functions"
      output_file: output/test3_1.out
      test:
         suffix: 1
         args: -fn_method {{0 1}}
      test:
         suffix: 1_subdiagonalpade
         args: -fn_method {{2 3}}
         requires: c99_complex !single
      test:
         suffix: 1_cuda
         args: -fn_method 4
         requires: cuda
      test:
         suffix: 1_magma
         args: -fn_method {{5 6 7 8}}
         requires: cuda magma
      test:
         suffix: 2
         args: -inplace -fn_method{{0 1}}
      test:
         suffix: 2_subdiagonalpade
         args: -inplace -fn_method{{2 3}}
         requires: c99_complex !single
      test:
         suffix: 2_cuda
         args: -inplace -fn_method 4
         requires: cuda
      test:
         suffix: 2_magma
         args: -inplace -fn_method {{5 6 7 8}}
         requires: cuda magma

   testset:
      args: -fn_scale 0.1
      filter: grep -v "computing matrix functions"
      output_file: output/test3_3.out
      test:
         suffix: 3
         args: -fn_method {{0 1}}
      test:
        suffix: 3_subdiagonalpade
        args: -fn_method {{2 3}}
        requires: c99_complex !single

   testset:
      args: -n 120 -fn_scale 0.6,1.5
      filter: grep -v "computing matrix functions"
      output_file: output/test3_4.out
      test:
         suffix: 4
         args: -fn_method {{0 1}}
         requires: !single
      test:
        suffix: 4_subdiagonalpade
        args: -fn_method {{2 3}}
        requires: c99_complex !single

   test:
      suffix: 5
      args: -fn_scale 30 -fn_method {{2 3}}
      filter: grep -v "computing matrix functions"
      requires: c99_complex !single
      output_file: output/test3_5.out

   test:
      suffix: 6
      args: -fn_scale 1e-9 -fn_method {{2 3}}
      filter: grep -v "computing matrix functions"
      requires: c99_complex !single
      output_file: output/test3_6.out

TEST*/
