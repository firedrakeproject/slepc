/*
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
   SLEPc - Scalable Library for Eigenvalue Problem Computations
   Copyright (c) 2002-2021, Universitat Politecnica de Valencia, Spain

   This file is part of SLEPc.
   SLEPc is distributed under a 2-clause BSD license (see LICENSE).
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
*/

static char help[] = "Test matrix square root.\n\n";

#include <slepcfn.h>

/*
   Compute matrix square root B = sqrtm(A)
   Check result as norm(B*B-A)
 */
PetscErrorCode TestMatSqrt(FN fn,Mat A,PetscViewer viewer,PetscBool verbose,PetscBool inplace)
{
  PetscScalar    tau,eta;
  PetscReal      nrm;
  PetscBool      set,flg;
  PetscInt       n;
  Mat            S,R;
  Vec            v,f0;

  PetscFunctionBeginUser;
  CHKERRQ(MatGetSize(A,&n,NULL));
  CHKERRQ(MatCreateSeqDense(PETSC_COMM_SELF,n,n,NULL,&S));
  CHKERRQ(PetscObjectSetName((PetscObject)S,"S"));
  CHKERRQ(FNGetScale(fn,&tau,&eta));
  /* compute square root */
  if (inplace) {
    CHKERRQ(MatCopy(A,S,SAME_NONZERO_PATTERN));
    CHKERRQ(MatIsHermitianKnown(A,&set,&flg));
    if (set && flg) CHKERRQ(MatSetOption(S,MAT_HERMITIAN,PETSC_TRUE));
    CHKERRQ(FNEvaluateFunctionMat(fn,S,NULL));
  } else {
    CHKERRQ(FNEvaluateFunctionMat(fn,A,S));
  }
  if (verbose) {
    CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"Matrix A - - - - - - - -\n"));
    CHKERRQ(MatView(A,viewer));
    CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"Computed sqrtm(A) - - - - - - -\n"));
    CHKERRQ(MatView(S,viewer));
  }
  /* check error ||S*S-A||_F */
  CHKERRQ(MatMatMult(S,S,MAT_INITIAL_MATRIX,PETSC_DEFAULT,&R));
  if (eta!=1.0) {
    CHKERRQ(MatScale(R,1.0/(eta*eta)));
  }
  CHKERRQ(MatAXPY(R,-tau,A,SAME_NONZERO_PATTERN));
  CHKERRQ(MatNorm(R,NORM_FROBENIUS,&nrm));
  if (nrm<100*PETSC_MACHINE_EPSILON) {
    CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"||S*S-A||_F < 100*eps\n"));
  } else {
    CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"||S*S-A||_F = %g\n",(double)nrm));
  }
  /* check FNEvaluateFunctionMatVec() */
  CHKERRQ(MatCreateVecs(A,&v,&f0));
  CHKERRQ(MatGetColumnVector(S,f0,0));
  CHKERRQ(FNEvaluateFunctionMatVec(fn,A,v));
  CHKERRQ(VecAXPY(v,-1.0,f0));
  CHKERRQ(VecNorm(v,NORM_2,&nrm));
  if (nrm>100*PETSC_MACHINE_EPSILON) {
    CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"Warning: the norm of f(A)*e_1-v is %g\n",(double)nrm));
  }
  CHKERRQ(MatDestroy(&S));
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
  PetscBool      verbose,inplace;
  PetscRandom    myrand;
  PetscReal      v;

  ierr = SlepcInitialize(&argc,&argv,(char*)0,help);if (ierr) return ierr;
  CHKERRQ(PetscOptionsGetInt(NULL,NULL,"-n",&n,NULL));
  CHKERRQ(PetscOptionsHasName(NULL,NULL,"-verbose",&verbose));
  CHKERRQ(PetscOptionsHasName(NULL,NULL,"-inplace",&inplace));
  CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"Matrix square root, n=%" PetscInt_FMT ".\n",n));

  /* Create function object */
  CHKERRQ(FNCreate(PETSC_COMM_WORLD,&fn));
  CHKERRQ(FNSetType(fn,FNSQRT));
  CHKERRQ(FNSetFromOptions(fn));

  /* Set up viewer */
  CHKERRQ(PetscViewerASCIIGetStdout(PETSC_COMM_WORLD,&viewer));
  CHKERRQ(FNView(fn,viewer));
  if (verbose) {
    CHKERRQ(PetscViewerPushFormat(viewer,PETSC_VIEWER_ASCII_MATLAB));
  }

  /* Create matrix */
  CHKERRQ(MatCreateSeqDense(PETSC_COMM_SELF,n,n,NULL,&A));
  CHKERRQ(PetscObjectSetName((PetscObject)A,"A"));

  /* Compute square root of a symmetric matrix A */
  CHKERRQ(MatDenseGetArray(A,&As));
  for (i=0;i<n;i++) As[i+i*n]=2.5;
  for (j=1;j<3;j++) {
    for (i=0;i<n-j;i++) { As[i+(i+j)*n]=1.0; As[(i+j)+i*n]=1.0; }
  }
  CHKERRQ(MatDenseRestoreArray(A,&As));
  CHKERRQ(MatSetOption(A,MAT_HERMITIAN,PETSC_TRUE));
  CHKERRQ(TestMatSqrt(fn,A,viewer,verbose,inplace));

  /* Repeat with upper triangular A */
  CHKERRQ(MatDenseGetArray(A,&As));
  for (j=1;j<3;j++) {
    for (i=0;i<n-j;i++) As[(i+j)+i*n]=0.0;
  }
  CHKERRQ(MatDenseRestoreArray(A,&As));
  CHKERRQ(MatSetOption(A,MAT_HERMITIAN,PETSC_FALSE));
  CHKERRQ(TestMatSqrt(fn,A,viewer,verbose,inplace));

  /* Repeat with non-symmetic A */
  CHKERRQ(PetscRandomCreate(PETSC_COMM_WORLD,&myrand));
  CHKERRQ(PetscRandomSetFromOptions(myrand));
  CHKERRQ(PetscRandomSetInterval(myrand,0.0,1.0));
  CHKERRQ(MatDenseGetArray(A,&As));
  for (j=1;j<3;j++) {
    for (i=0;i<n-j;i++) {
      CHKERRQ(PetscRandomGetValueReal(myrand,&v));
      As[(i+j)+i*n]=v;
    }
  }
  CHKERRQ(MatDenseRestoreArray(A,&As));
  CHKERRQ(PetscRandomDestroy(&myrand));
  CHKERRQ(MatSetOption(A,MAT_HERMITIAN,PETSC_FALSE));
  CHKERRQ(TestMatSqrt(fn,A,viewer,verbose,inplace));

  CHKERRQ(MatDestroy(&A));
  CHKERRQ(FNDestroy(&fn));
  ierr = SlepcFinalize();
  return ierr;
}

/*TEST

   testset:
      args: -fn_scale .05,2 -n 100
      filter: grep -v "computing matrix functions"
      output_file: output/test7_1.out
      timeoutfactor: 2
      test:
         suffix: 1
         args: -fn_method {{0 1 2}}
      test:
         suffix: 1_sadeghi
         args: -fn_method 3
         requires: !single
      test:
         suffix: 1_cuda
         args: -fn_method 4
         requires: cuda !single
      test:
         suffix: 1_magma
         args: -fn_method {{5 6}}
         requires: cuda magma !single
      test:
         suffix: 2
         args: -inplace -fn_method {{0 1 2}}
      test:
         suffix: 2_sadeghi
         args: -inplace -fn_method 3
         requires: !single
      test:
         suffix: 2_cuda
         args: -inplace -fn_method 4
         requires: cuda !single
      test:
         suffix: 2_magma
         args: -inplace -fn_method {{5 6}}
         requires: cuda magma !single

   testset:
      nsize: 3
      args: -fn_scale .05,2 -n 100 -fn_parallel synchronized
      filter: grep -v "computing matrix functions" | grep -v "SYNCHRONIZED" | sed -e "s/3 MPI/1 MPI/g"
      output_file: output/test7_1.out
      test:
         suffix: 3
      test:
         suffix: 3_inplace
         args: -inplace

TEST*/
