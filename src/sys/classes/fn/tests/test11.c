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

        f(x) = (exp(x)-1)/x    (the phi_1 function)

   with the following tree:

            f(x)                  f(x)              (combined by division)
           /    \                 p(x) = x          (polynomial)
        a(x)    p(x)              a(x)              (combined by addition)
       /    \                     e(x) = exp(x)     (exponential)
     e(x)   c(x)                  c(x) = -1         (constant)
*/

static char help[] = "Another test of a combined function.\n\n";

#include <slepcfn.h>

/*
   Compute matrix function B = A\(exp(A)-I)
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
  FN             f,p,a,e,c,f1,f2;
  FNCombineType  ctype;
  Mat            A;
  PetscInt       i,j,n=10,np;
  PetscScalar    x,y,yp,*As,coeffs[10];
  char           strx[50],str[50];
  PetscViewer    viewer;
  PetscBool      verbose,inplace;

  CHKERRQ(SlepcInitialize(&argc,&argv,(char*)0,help));
  CHKERRQ(PetscOptionsGetInt(NULL,NULL,"-n",&n,NULL));
  CHKERRQ(PetscOptionsHasName(NULL,NULL,"-verbose",&verbose));
  CHKERRQ(PetscOptionsHasName(NULL,NULL,"-inplace",&inplace));
  CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"Phi1 via a combined function, n=%" PetscInt_FMT ".\n",n));

  /* Create function */

  /* e(x) = exp(x) */
  CHKERRQ(FNCreate(PETSC_COMM_WORLD,&e));
  CHKERRQ(PetscObjectSetName((PetscObject)e,"e"));
  CHKERRQ(FNSetType(e,FNEXP));
  CHKERRQ(FNSetFromOptions(e));
  /* c(x) = -1 */
  CHKERRQ(FNCreate(PETSC_COMM_WORLD,&c));
  CHKERRQ(PetscObjectSetName((PetscObject)c,"c"));
  CHKERRQ(FNSetType(c,FNRATIONAL));
  CHKERRQ(FNSetFromOptions(c));
  np = 1;
  coeffs[0] = -1.0;
  CHKERRQ(FNRationalSetNumerator(c,np,coeffs));
  /* a(x) */
  CHKERRQ(FNCreate(PETSC_COMM_WORLD,&a));
  CHKERRQ(PetscObjectSetName((PetscObject)a,"a"));
  CHKERRQ(FNSetType(a,FNCOMBINE));
  CHKERRQ(FNSetFromOptions(a));
  CHKERRQ(FNCombineSetChildren(a,FN_COMBINE_ADD,e,c));
  /* p(x) = x */
  CHKERRQ(FNCreate(PETSC_COMM_WORLD,&p));
  CHKERRQ(PetscObjectSetName((PetscObject)p,"p"));
  CHKERRQ(FNSetType(p,FNRATIONAL));
  CHKERRQ(FNSetFromOptions(p));
  np = 2;
  coeffs[0] = 1.0; coeffs[1] = 0.0;
  CHKERRQ(FNRationalSetNumerator(p,np,coeffs));
  /* f(x) */
  CHKERRQ(FNCreate(PETSC_COMM_WORLD,&f));
  CHKERRQ(PetscObjectSetName((PetscObject)f,"f"));
  CHKERRQ(FNSetType(f,FNCOMBINE));
  CHKERRQ(FNSetFromOptions(f));
  CHKERRQ(FNCombineSetChildren(f,FN_COMBINE_DIVIDE,a,p));

  /* Set up viewer */
  CHKERRQ(PetscViewerASCIIGetStdout(PETSC_COMM_WORLD,&viewer));
  CHKERRQ(FNCombineGetChildren(f,&ctype,&f1,&f2));
  CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"Two functions combined with division:\n"));
  CHKERRQ(FNView(f1,viewer));
  CHKERRQ(FNView(f2,viewer));
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

  /* Create matrices */
  CHKERRQ(MatCreateSeqDense(PETSC_COMM_SELF,n,n,NULL,&A));
  CHKERRQ(PetscObjectSetName((PetscObject)A,"A"));

  /* Fill A with 1-D Laplacian matrix */
  CHKERRQ(MatDenseGetArray(A,&As));
  for (i=0;i<n;i++) As[i+i*n]=2.0;
  j=1;
  for (i=0;i<n-j;i++) { As[i+(i+j)*n]=-1.0; As[(i+j)+i*n]=-1.0; }
  CHKERRQ(MatDenseRestoreArray(A,&As));
  CHKERRQ(MatSetOption(A,MAT_HERMITIAN,PETSC_TRUE));
  CHKERRQ(TestMatCombine(f,A,viewer,verbose,inplace));

  /* Repeat with same matrix as non-symmetric */
  CHKERRQ(MatSetOption(A,MAT_HERMITIAN,PETSC_FALSE));
  CHKERRQ(TestMatCombine(f,A,viewer,verbose,inplace));

  CHKERRQ(MatDestroy(&A));
  CHKERRQ(FNDestroy(&f));
  CHKERRQ(FNDestroy(&p));
  CHKERRQ(FNDestroy(&a));
  CHKERRQ(FNDestroy(&e));
  CHKERRQ(FNDestroy(&c));
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
      output_file: output/test11_1.out

TEST*/
