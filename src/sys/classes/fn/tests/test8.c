/*
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
   SLEPc - Scalable Library for Eigenvalue Problem Computations
   Copyright (c) 2002-2021, Universitat Politecnica de Valencia, Spain

   This file is part of SLEPc.
   SLEPc is distributed under a 2-clause BSD license (see LICENSE).
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
*/

static char help[] = "Test matrix inverse square root.\n\n";

#include <slepcfn.h>

/*
   Compute matrix inverse square root B = inv(sqrtm(A))
   Check result as norm(B*B*A-I)
 */
PetscErrorCode TestMatInvSqrt(FN fn,Mat A,PetscViewer viewer,PetscBool verbose,PetscBool inplace)
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
  /* compute inverse square root */
  if (inplace) {
    CHKERRQ(MatCopy(A,S,SAME_NONZERO_PATTERN));
    CHKERRQ(MatIsHermitianKnown(A,&set,&flg));
    if (set && flg) CHKERRQ(MatSetOption(S,MAT_HERMITIAN,PETSC_TRUE));
    CHKERRQ(FNEvaluateFunctionMat(fn,S,NULL));
  } else CHKERRQ(FNEvaluateFunctionMat(fn,A,S));
  if (verbose) {
    CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"Matrix A - - - - - - - -\n"));
    CHKERRQ(MatView(A,viewer));
    CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"Computed inv(sqrtm(A)) - - - - - - -\n"));
    CHKERRQ(MatView(S,viewer));
  }
  /* check error ||S*S*A-I||_F */
  CHKERRQ(MatMatMult(S,S,MAT_INITIAL_MATRIX,PETSC_DEFAULT,&R));
  if (eta!=1.0) CHKERRQ(MatScale(R,1.0/(eta*eta)));
  CHKERRQ(MatCreateVecs(A,&v,&f0));
  CHKERRQ(MatGetColumnVector(S,f0,0));
  CHKERRQ(MatCopy(R,S,SAME_NONZERO_PATTERN));
  CHKERRQ(MatDestroy(&R));
  if (tau!=1.0) CHKERRQ(MatScale(S,tau));
  CHKERRQ(MatMatMult(S,A,MAT_INITIAL_MATRIX,PETSC_DEFAULT,&R));
  CHKERRQ(MatShift(R,-1.0));
  CHKERRQ(MatNorm(R,NORM_FROBENIUS,&nrm));
  CHKERRQ(MatDestroy(&R));
  if (nrm<100*PETSC_MACHINE_EPSILON) CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"||S*S*A-I||_F < 100*eps\n"));
  else CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"||S*S*A-I||_F = %g\n",(double)nrm));
  /* check FNEvaluateFunctionMatVec() */
  CHKERRQ(FNEvaluateFunctionMatVec(fn,A,v));
  CHKERRQ(VecAXPY(v,-1.0,f0));
  CHKERRQ(VecNorm(v,NORM_2,&nrm));
  if (nrm>100*PETSC_MACHINE_EPSILON) CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"Warning: the norm of f(A)*e_1-v is %g\n",(double)nrm));
  CHKERRQ(MatDestroy(&S));
  CHKERRQ(VecDestroy(&v));
  CHKERRQ(VecDestroy(&f0));
  PetscFunctionReturn(0);
}

int main(int argc,char **argv)
{
  FN             fn;
  Mat            A;
  PetscInt       i,j,n=10;
  PetscScalar    x,y,yp,*As;
  PetscViewer    viewer;
  PetscBool      verbose,inplace;
  PetscRandom    myrand;
  PetscReal      v;
  char           strx[50],str[50];

  CHKERRQ(SlepcInitialize(&argc,&argv,(char*)0,help));
  CHKERRQ(PetscOptionsGetInt(NULL,NULL,"-n",&n,NULL));
  CHKERRQ(PetscOptionsHasName(NULL,NULL,"-verbose",&verbose));
  CHKERRQ(PetscOptionsHasName(NULL,NULL,"-inplace",&inplace));
  CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"Matrix inverse square root, n=%" PetscInt_FMT ".\n",n));

  /* Create function object */
  CHKERRQ(FNCreate(PETSC_COMM_WORLD,&fn));
  CHKERRQ(FNSetType(fn,FNINVSQRT));
  CHKERRQ(FNSetFromOptions(fn));

  /* Set up viewer */
  CHKERRQ(PetscViewerASCIIGetStdout(PETSC_COMM_WORLD,&viewer));
  CHKERRQ(FNView(fn,viewer));
  if (verbose) CHKERRQ(PetscViewerPushFormat(viewer,PETSC_VIEWER_ASCII_MATLAB));

  /* Scalar evaluation */
  x = 2.2;
  CHKERRQ(SlepcSNPrintfScalar(strx,sizeof(strx),x,PETSC_FALSE));
  CHKERRQ(FNEvaluateFunction(fn,x,&y));
  CHKERRQ(FNEvaluateDerivative(fn,x,&yp));
  CHKERRQ(SlepcSNPrintfScalar(str,sizeof(str),y,PETSC_FALSE));
  CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"  f(%s)=%s\n",strx,str));
  CHKERRQ(SlepcSNPrintfScalar(str,sizeof(str),yp,PETSC_FALSE));
  CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"  f'(%s)=%s\n",strx,str));

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
  CHKERRQ(TestMatInvSqrt(fn,A,viewer,verbose,inplace));

  /* Repeat with upper triangular A */
  CHKERRQ(MatDenseGetArray(A,&As));
  for (j=1;j<3;j++) {
    for (i=0;i<n-j;i++) As[(i+j)+i*n]=0.0;
  }
  CHKERRQ(MatDenseRestoreArray(A,&As));
  CHKERRQ(MatSetOption(A,MAT_HERMITIAN,PETSC_FALSE));
  CHKERRQ(TestMatInvSqrt(fn,A,viewer,verbose,inplace));

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
  CHKERRQ(TestMatInvSqrt(fn,A,viewer,verbose,inplace));

  CHKERRQ(MatDestroy(&A));
  CHKERRQ(FNDestroy(&fn));
  CHKERRQ(SlepcFinalize());
  return 0;
}

/*TEST

   testset:
      args: -fn_scale 0.9,0.5 -n 10
      filter: grep -v "computing matrix functions"
      output_file: output/test8_1.out
      test:
         suffix: 1
         args: -fn_method {{0 1 2 3}}
      test:
         suffix: 1_cuda
         args: -fn_method 4
         requires: cuda
      test:
         suffix: 1_magma
         args: -fn_method {{5 6}}
         requires: cuda magma
      test:
         suffix: 2
         args: -inplace -fn_method {{0 1 2 3}}
      test:
         suffix: 2_cuda
         args: -inplace -fn_method 4
         requires: cuda
      test:
         suffix: 2_magma
         args: -inplace -fn_method {{5 6}}
         requires: cuda magma

TEST*/
