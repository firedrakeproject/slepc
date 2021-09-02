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
  PetscErrorCode ierr;
  PetscScalar    tau,eta;
  PetscBool      set,flg;
  PetscInt       n;
  Mat            F,R,Finv,Acopy;
  Vec            v,f0;
  FN             finv;
  PetscReal      nrm,nrmf;

  PetscFunctionBeginUser;
  ierr = MatGetSize(A,&n,NULL);CHKERRQ(ierr);
  ierr = MatDuplicate(A,MAT_DO_NOT_COPY_VALUES,&F);CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject)F,"F");CHKERRQ(ierr);
  /* compute matrix exponential */
  if (inplace) {
    ierr = MatCopy(A,F,SAME_NONZERO_PATTERN);CHKERRQ(ierr);
    ierr = MatIsHermitianKnown(A,&set,&flg);CHKERRQ(ierr);
    if (set && flg) { ierr = MatSetOption(F,MAT_HERMITIAN,PETSC_TRUE);CHKERRQ(ierr); }
    ierr = FNEvaluateFunctionMat(fn,F,NULL);CHKERRQ(ierr);
  } else {
    ierr = MatDuplicate(A,MAT_COPY_VALUES,&Acopy);CHKERRQ(ierr);
    ierr = FNEvaluateFunctionMat(fn,A,F);CHKERRQ(ierr);
    /* check that A has not been modified */
    ierr = MatAXPY(Acopy,-1.0,A,SAME_NONZERO_PATTERN);CHKERRQ(ierr);
    ierr = MatNorm(Acopy,NORM_1,&nrm);CHKERRQ(ierr);
    if (nrm>100*PETSC_MACHINE_EPSILON) {
      ierr = PetscPrintf(PETSC_COMM_WORLD,"Warning: the input matrix has changed by %g\n",(double)nrm);CHKERRQ(ierr);
    }
    ierr = MatDestroy(&Acopy);CHKERRQ(ierr);
  }
  if (verbose) {
    ierr = PetscPrintf(PETSC_COMM_WORLD,"Matrix A - - - - - - - -\n");CHKERRQ(ierr);
    ierr = MatView(A,viewer);CHKERRQ(ierr);
    ierr = PetscPrintf(PETSC_COMM_WORLD,"Computed expm(A) - - - - - - -\n");CHKERRQ(ierr);
    ierr = MatView(F,viewer);CHKERRQ(ierr);
  }
  /* print matrix norm for checking */
  ierr = MatNorm(F,NORM_1,&nrmf);CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD,"The 1-norm of f(A) is %g\n",(double)nrmf);CHKERRQ(ierr);
  if (checkerror) {
    ierr = MatDuplicate(A,MAT_DO_NOT_COPY_VALUES,&Finv);CHKERRQ(ierr);
    ierr = PetscObjectSetName((PetscObject)Finv,"Finv");CHKERRQ(ierr);
    ierr = FNGetScale(fn,&tau,&eta);CHKERRQ(ierr);
    /* compute inverse exp(-tau*A)/eta */
    ierr = FNCreate(PETSC_COMM_WORLD,&finv);CHKERRQ(ierr);
    ierr = FNSetType(finv,FNEXP);CHKERRQ(ierr);
    ierr = FNSetFromOptions(finv);CHKERRQ(ierr);
    ierr = FNSetScale(finv,-tau,1.0/eta);CHKERRQ(ierr);
    if (inplace) {
      ierr = MatCopy(A,Finv,SAME_NONZERO_PATTERN);CHKERRQ(ierr);
      ierr = MatIsHermitianKnown(A,&set,&flg);CHKERRQ(ierr);
      if (set && flg) { ierr = MatSetOption(Finv,MAT_HERMITIAN,PETSC_TRUE);CHKERRQ(ierr); }
      ierr = FNEvaluateFunctionMat(finv,Finv,NULL);CHKERRQ(ierr);
    } else {
      ierr = FNEvaluateFunctionMat(finv,A,Finv);CHKERRQ(ierr);
    }
    ierr = FNDestroy(&finv);CHKERRQ(ierr);
    /* check error ||F*Finv-I||_F */
    ierr = MatMatMult(F,Finv,MAT_INITIAL_MATRIX,PETSC_DEFAULT,&R);CHKERRQ(ierr);
    ierr = MatShift(R,-1.0);CHKERRQ(ierr);
    ierr = MatNorm(R,NORM_FROBENIUS,&nrm);CHKERRQ(ierr);
    if (nrm<100*PETSC_MACHINE_EPSILON) {
      ierr = PetscPrintf(PETSC_COMM_WORLD,"||exp(A)*exp(-A)-I||_F < 100*eps\n");CHKERRQ(ierr);
    } else {
      ierr = PetscPrintf(PETSC_COMM_WORLD,"||exp(A)*exp(-A)-I||_F = %g\n",(double)nrm);CHKERRQ(ierr);
    }
    ierr = MatDestroy(&R);CHKERRQ(ierr);
    ierr = MatDestroy(&Finv);CHKERRQ(ierr);
  }
  /* check FNEvaluateFunctionMatVec() */
  ierr = MatCreateVecs(A,&v,&f0);CHKERRQ(ierr);
  ierr = MatGetColumnVector(F,f0,0);CHKERRQ(ierr);
  ierr = FNEvaluateFunctionMatVec(fn,A,v);CHKERRQ(ierr);
  ierr = VecAXPY(v,-1.0,f0);CHKERRQ(ierr);
  ierr = VecNorm(v,NORM_2,&nrm);CHKERRQ(ierr);
  if (nrm/nrmf>100*PETSC_MACHINE_EPSILON) {
    ierr = PetscPrintf(PETSC_COMM_WORLD,"Warning: the norm of f(A)*e_1-v is %g\n",(double)nrm);CHKERRQ(ierr);
  }
  ierr = MatDestroy(&F);CHKERRQ(ierr);
  ierr = VecDestroy(&v);CHKERRQ(ierr);
  ierr = VecDestroy(&f0);CHKERRQ(ierr);
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
  PetscBool      verbose,inplace,checkerror;

  ierr = SlepcInitialize(&argc,&argv,(char*)0,help);if (ierr) return ierr;
  ierr = PetscOptionsGetInt(NULL,NULL,"-n",&n,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsHasName(NULL,NULL,"-verbose",&verbose);CHKERRQ(ierr);
  ierr = PetscOptionsHasName(NULL,NULL,"-inplace",&inplace);CHKERRQ(ierr);
  ierr = PetscOptionsHasName(NULL,NULL,"-checkerror",&checkerror);CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD,"Matrix exponential, n=%D.\n",n);CHKERRQ(ierr);

  /* Create exponential function object */
  ierr = FNCreate(PETSC_COMM_WORLD,&fn);CHKERRQ(ierr);
  ierr = FNSetType(fn,FNEXP);CHKERRQ(ierr);
  ierr = FNSetFromOptions(fn);CHKERRQ(ierr);

  /* Set up viewer */
  ierr = PetscViewerASCIIGetStdout(PETSC_COMM_WORLD,&viewer);CHKERRQ(ierr);
  ierr = FNView(fn,viewer);CHKERRQ(ierr);
  if (verbose) {
    ierr = PetscViewerPushFormat(viewer,PETSC_VIEWER_ASCII_MATLAB);CHKERRQ(ierr);
  }

  /* Create matrices */
  ierr = MatCreateSeqDense(PETSC_COMM_SELF,n,n,NULL,&A);CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject)A,"A");CHKERRQ(ierr);

  /* Fill A with a symmetric Toeplitz matrix */
  ierr = MatDenseGetArray(A,&As);CHKERRQ(ierr);
  for (i=0;i<n;i++) As[i+i*n]=2.0;
  for (j=1;j<3;j++) {
    for (i=0;i<n-j;i++) { As[i+(i+j)*n]=1.0; As[(i+j)+i*n]=1.0; }
  }
  ierr = MatDenseRestoreArray(A,&As);CHKERRQ(ierr);
  ierr = MatSetOption(A,MAT_HERMITIAN,PETSC_TRUE);CHKERRQ(ierr);
  ierr = TestMatExp(fn,A,viewer,verbose,inplace,checkerror);CHKERRQ(ierr);

  /* Repeat with non-symmetric A */
  ierr = MatDenseGetArray(A,&As);CHKERRQ(ierr);
  for (j=1;j<3;j++) {
    for (i=0;i<n-j;i++) { As[(i+j)+i*n]=-1.0; }
  }
  ierr = MatDenseRestoreArray(A,&As);CHKERRQ(ierr);
  ierr = MatSetOption(A,MAT_HERMITIAN,PETSC_FALSE);CHKERRQ(ierr);
  ierr = TestMatExp(fn,A,viewer,verbose,inplace,checkerror);CHKERRQ(ierr);

  ierr = MatDestroy(&A);CHKERRQ(ierr);
  ierr = FNDestroy(&fn);CHKERRQ(ierr);
  ierr = SlepcFinalize();
  return ierr;
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
