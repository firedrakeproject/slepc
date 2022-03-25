/*
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
   SLEPc - Scalable Library for Eigenvalue Problem Computations
   Copyright (c) 2002-2021, Universitat Politecnica de Valencia, Spain

   This file is part of SLEPc.
   SLEPc is distributed under a 2-clause BSD license (see LICENSE).
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
*/
/*
   Define the function

        f(x) = (1-x^2) exp(-x/(1+x^2))

   with the following tree:

            f(x)                  f(x)              (combined by product)
           /    \                 g(x) = 1-x^2      (polynomial)
        g(x)    h(x)              h(x)              (combined by composition)
               /    \             r(x) = -x/(1+x^2) (rational)
             r(x)   e(x)          e(x) = exp(x)     (exponential)
*/

static char help[] = "Test combined function.\n\n";

#include <slepcfn.h>

/*
   Compute matrix function B = (I-A^2) exp(-(I+A^2)\A)
 */
PetscErrorCode TestMatCombine(FN fn,Mat A,PetscViewer viewer,PetscBool verbose,PetscBool inplace)
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
  CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"The 1-norm of f(A) is %6.3f\n",(double)nrm));
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
  FN             f,g,h,e,r,fcopy;
  Mat            A;
  PetscInt       i,j,n=10,np,nq;
  PetscScalar    x,y,yp,*As,p[10],q[10];
  char           strx[50],str[50];
  PetscViewer    viewer;
  PetscBool      verbose,inplace;

  CHKERRQ(SlepcInitialize(&argc,&argv,(char*)0,help));
  CHKERRQ(PetscOptionsGetInt(NULL,NULL,"-n",&n,NULL));
  CHKERRQ(PetscOptionsHasName(NULL,NULL,"-verbose",&verbose));
  CHKERRQ(PetscOptionsHasName(NULL,NULL,"-inplace",&inplace));
  CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"Combined function, n=%" PetscInt_FMT ".\n",n));

  /* Create function */

  /* e(x) = exp(x) */
  CHKERRQ(FNCreate(PETSC_COMM_WORLD,&e));
  CHKERRQ(FNSetType(e,FNEXP));
  CHKERRQ(FNSetFromOptions(e));
  /* r(x) = x/(1+x^2) */
  CHKERRQ(FNCreate(PETSC_COMM_WORLD,&r));
  CHKERRQ(FNSetType(r,FNRATIONAL));
  CHKERRQ(FNSetFromOptions(r));
  np = 2; nq = 3;
  p[0] = -1.0; p[1] = 0.0;
  q[0] = 1.0; q[1] = 0.0; q[2] = 1.0;
  CHKERRQ(FNRationalSetNumerator(r,np,p));
  CHKERRQ(FNRationalSetDenominator(r,nq,q));
  /* h(x) */
  CHKERRQ(FNCreate(PETSC_COMM_WORLD,&h));
  CHKERRQ(FNSetType(h,FNCOMBINE));
  CHKERRQ(FNSetFromOptions(h));
  CHKERRQ(FNCombineSetChildren(h,FN_COMBINE_COMPOSE,r,e));
  /* g(x) = 1-x^2 */
  CHKERRQ(FNCreate(PETSC_COMM_WORLD,&g));
  CHKERRQ(FNSetType(g,FNRATIONAL));
  CHKERRQ(FNSetFromOptions(g));
  np = 3;
  p[0] = -1.0; p[1] = 0.0; p[2] = 1.0;
  CHKERRQ(FNRationalSetNumerator(g,np,p));
  /* f(x) */
  CHKERRQ(FNCreate(PETSC_COMM_WORLD,&f));
  CHKERRQ(FNSetType(f,FNCOMBINE));
  CHKERRQ(FNSetFromOptions(f));
  CHKERRQ(FNCombineSetChildren(f,FN_COMBINE_MULTIPLY,g,h));

  /* Set up viewer */
  CHKERRQ(PetscViewerASCIIGetStdout(PETSC_COMM_WORLD,&viewer));
  CHKERRQ(FNView(f,viewer));
  if (verbose) CHKERRQ(PetscViewerPushFormat(viewer,PETSC_VIEWER_ASCII_MATLAB));

  /* Scalar evaluation */
  x = 2.2;
  CHKERRQ(SlepcSNPrintfScalar(strx,sizeof(strx),x,PETSC_FALSE));
  CHKERRQ(FNEvaluateFunction(f,x,&y));
  CHKERRQ(FNEvaluateDerivative(f,x,&yp));
  CHKERRQ(SlepcSNPrintfScalar(str,sizeof(str),y,PETSC_FALSE));
  CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"  f(%s)=%s\n",strx,str));
  CHKERRQ(SlepcSNPrintfScalar(str,sizeof(str),yp,PETSC_FALSE));
  CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"  f'(%s)=%s\n",strx,str));

  /* Test duplication */
  CHKERRQ(FNDuplicate(f,PetscObjectComm((PetscObject)f),&fcopy));

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
  CHKERRQ(TestMatCombine(fcopy,A,viewer,verbose,inplace));

  /* Repeat with same matrix as non-symmetric */
  CHKERRQ(MatSetOption(A,MAT_HERMITIAN,PETSC_FALSE));
  CHKERRQ(TestMatCombine(fcopy,A,viewer,verbose,inplace));

  CHKERRQ(MatDestroy(&A));
  CHKERRQ(FNDestroy(&f));
  CHKERRQ(FNDestroy(&fcopy));
  CHKERRQ(FNDestroy(&g));
  CHKERRQ(FNDestroy(&h));
  CHKERRQ(FNDestroy(&e));
  CHKERRQ(FNDestroy(&r));
  CHKERRQ(SlepcFinalize());
  return 0;
}

/*TEST

   test:
      suffix: 1
      nsize: 1

   test:
      suffix: 2
      nsize: 1
      args: -inplace
      output_file: output/test6_1.out

TEST*/
