/*
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
   SLEPc - Scalable Library for Eigenvalue Problem Computations
   Copyright (c) 2002-2021, Universitat Politecnica de Valencia, Spain

   This file is part of SLEPc.
   SLEPc is distributed under a 2-clause BSD license (see LICENSE).
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
*/

static char help[] = "Test matrix rational function.\n\n";

#include <slepcfn.h>

/*
   Compute matrix rational function B = q(A)\p(A)
 */
PetscErrorCode TestMatRational(FN fn,Mat A,PetscViewer viewer,PetscBool verbose,PetscBool inplace)
{
  PetscBool      set,flg;
  PetscInt       n;
  Mat            F;
  Vec            v,f0;
  PetscReal      nrm;

  PetscFunctionBeginUser;
  CHKERRQ(MatGetSize(A,&n,NULL));
  CHKERRQ(MatCreateSeqDense(PETSC_COMM_SELF,n,n,NULL,&F));
  CHKERRQ(PetscObjectSetName((PetscObject)F,"F"));
  /* compute matrix function */
  if (inplace) {
    CHKERRQ(MatCopy(A,F,SAME_NONZERO_PATTERN));
    CHKERRQ(MatIsHermitianKnown(A,&set,&flg));
    if (set && flg) CHKERRQ(MatSetOption(F,MAT_HERMITIAN,PETSC_TRUE));
    CHKERRQ(FNEvaluateFunctionMat(fn,F,NULL));
  } else CHKERRQ(FNEvaluateFunctionMat(fn,A,F));
  if (verbose) {
    CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"Matrix A - - - - - - - -\n"));
    CHKERRQ(MatView(A,viewer));
    CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"Computed f(A) - - - - - - -\n"));
    CHKERRQ(MatView(F,viewer));
  }
  /* print matrix norm for checking */
  CHKERRQ(MatNorm(F,NORM_1,&nrm));
  CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"The 1-norm of f(A) is %g\n",(double)nrm));
  /* check FNEvaluateFunctionMatVec() */
  CHKERRQ(MatCreateVecs(A,&v,&f0));
  CHKERRQ(MatGetColumnVector(F,f0,0));
  CHKERRQ(FNEvaluateFunctionMatVec(fn,A,v));
  CHKERRQ(VecAXPY(v,-1.0,f0));
  CHKERRQ(VecNorm(v,NORM_2,&nrm));
  if (nrm>100*PETSC_MACHINE_EPSILON) CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"Warning: the norm of f(A)*e_1-v is %g\n",(double)nrm));
  CHKERRQ(MatDestroy(&F));
  CHKERRQ(VecDestroy(&v));
  CHKERRQ(VecDestroy(&f0));
  PetscFunctionReturn(0);
}

int main(int argc,char **argv)
{
  FN             fn;
  Mat            A;
  PetscInt       i,j,n=10,np,nq;
  PetscScalar    *As,p[10],q[10];
  PetscViewer    viewer;
  PetscBool      verbose,inplace;

  CHKERRQ(SlepcInitialize(&argc,&argv,(char*)0,help));
  CHKERRQ(PetscOptionsGetInt(NULL,NULL,"-n",&n,NULL));
  CHKERRQ(PetscOptionsHasName(NULL,NULL,"-verbose",&verbose));
  CHKERRQ(PetscOptionsHasName(NULL,NULL,"-inplace",&inplace));
  CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"Matrix rational function, n=%" PetscInt_FMT ".\n",n));

  /* Create rational function r(x)=p(x)/q(x) */
  CHKERRQ(FNCreate(PETSC_COMM_WORLD,&fn));
  CHKERRQ(FNSetType(fn,FNRATIONAL));
  np = 2; nq = 3;
  p[0] = -3.1; p[1] = 1.1;
  q[0] = 1.0; q[1] = -2.0; q[2] = 3.5;
  CHKERRQ(FNRationalSetNumerator(fn,np,p));
  CHKERRQ(FNRationalSetDenominator(fn,nq,q));
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
  CHKERRQ(TestMatRational(fn,A,viewer,verbose,inplace));

  /* Repeat with same matrix as non-symmetric */
  CHKERRQ(MatSetOption(A,MAT_HERMITIAN,PETSC_FALSE));
  CHKERRQ(TestMatRational(fn,A,viewer,verbose,inplace));

  CHKERRQ(MatDestroy(&A));
  CHKERRQ(FNDestroy(&fn));
  CHKERRQ(SlepcFinalize());
  return 0;
}

/*TEST

   testset:
      output_file: output/test5_1.out
      requires: !single
      test:
         suffix: 1
      test:
         suffix: 2
         args: -inplace

TEST*/
