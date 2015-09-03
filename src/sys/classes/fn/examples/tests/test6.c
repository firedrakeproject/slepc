/*
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
   SLEPc - Scalable Library for Eigenvalue Problem Computations
   Copyright (c) 2002-2015, Universitat Politecnica de Valencia, Spain

   This file is part of SLEPc.

   SLEPc is free software: you can redistribute it and/or modify it under  the
   terms of version 3 of the GNU Lesser General Public License as published by
   the Free Software Foundation.

   SLEPc  is  distributed in the hope that it will be useful, but WITHOUT  ANY
   WARRANTY;  without even the implied warranty of MERCHANTABILITY or  FITNESS
   FOR  A  PARTICULAR PURPOSE. See the GNU Lesser General Public  License  for
   more details.

   You  should have received a copy of the GNU Lesser General  Public  License
   along with SLEPc. If not, see <http://www.gnu.org/licenses/>.
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
*/
/*
   Define the function

        f(x) = (1-x^2) exp( -x/(1+x^2) )

   with the following tree:

            f(x)                  f(x)              (combined by product)
           /    \                 g(x) = 1-x^2      (polynomial)
        g(x)    h(x)              h(x)              (combined by composition)
               /    \             r(x) = -x/(1+x^2) (rational)
             r(x)   e(x)          e(x) = exp(x)     (exponential)
*/

static char help[] = "Test combined function.\n\n";

#include <slepcfn.h>

#undef __FUNCT__
#define __FUNCT__ "main"
int main(int argc,char **argv)
{
  PetscErrorCode ierr;
  FN             f,g,h,e,r;
  Mat            A,B;
  PetscInt       i,j,n=10,np,nq;
  PetscReal      nrm;
  PetscScalar    x,y,yp,*As,p[10],q[10];
  char           strx[50],str[50];
  PetscViewer    viewer;
  PetscBool      verbose;

  SlepcInitialize(&argc,&argv,(char*)0,help);
  ierr = PetscOptionsGetInt(NULL,"-n",&n,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsHasName(NULL,"-verbose",&verbose);CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD,"Combined function, n=%D.\n",n);CHKERRQ(ierr);

  /* Create function */

  /* e(x) = exp(x) */
  ierr = FNCreate(PETSC_COMM_WORLD,&e);CHKERRQ(ierr);
  ierr = FNSetType(e,FNEXP);CHKERRQ(ierr);
  /* r(x) = x/(1+x^2) */
  ierr = FNCreate(PETSC_COMM_WORLD,&r);CHKERRQ(ierr);
  ierr = FNSetType(r,FNRATIONAL);CHKERRQ(ierr);
  np = 2; nq = 3;
  p[0] = -1.0; p[1] = 0.0;
  q[0] = 1.0; q[1] = 0.0; q[2] = 1.0;
  ierr = FNRationalSetNumerator(r,np,p);CHKERRQ(ierr);
  ierr = FNRationalSetDenominator(r,nq,q);CHKERRQ(ierr);
  /* h(x) */
  ierr = FNCreate(PETSC_COMM_WORLD,&h);CHKERRQ(ierr);
  ierr = FNSetType(h,FNCOMBINE);CHKERRQ(ierr);
  ierr = FNCombineSetChildren(h,FN_COMBINE_COMPOSE,r,e);CHKERRQ(ierr);
  /* g(x) = 1-x^2 */
  ierr = FNCreate(PETSC_COMM_WORLD,&g);CHKERRQ(ierr);
  ierr = FNSetType(g,FNRATIONAL);CHKERRQ(ierr);
  np = 3;
  p[0] = -1.0; p[1] = 0.0; p[2] = 1.0;
  ierr = FNRationalSetNumerator(g,np,p);CHKERRQ(ierr);
  /* f(x) */
  ierr = FNCreate(PETSC_COMM_WORLD,&f);CHKERRQ(ierr);
  ierr = FNSetType(f,FNCOMBINE);CHKERRQ(ierr);
  ierr = FNCombineSetChildren(f,FN_COMBINE_MULTIPLY,g,h);CHKERRQ(ierr);

  /* Set up viewer */
  ierr = PetscViewerASCIIGetStdout(PETSC_COMM_WORLD,&viewer);CHKERRQ(ierr);
  ierr = FNView(f,viewer);CHKERRQ(ierr);
  if (verbose) {
    ierr = PetscViewerPushFormat(viewer,PETSC_VIEWER_ASCII_MATLAB);CHKERRQ(ierr);
  }

  /* Scalar evaluation */
  x = 2.2;
  ierr = SlepcSNPrintfScalar(strx,50,x,PETSC_FALSE);CHKERRQ(ierr);
  ierr = FNEvaluateFunction(f,x,&y);CHKERRQ(ierr);
  ierr = FNEvaluateDerivative(f,x,&yp);CHKERRQ(ierr);
  ierr = SlepcSNPrintfScalar(str,50,y,PETSC_FALSE);CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD,"  f(%s)=%s\n",strx,str);CHKERRQ(ierr);
  ierr = SlepcSNPrintfScalar(str,50,yp,PETSC_FALSE);CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD,"  f'(%s)=%s\n",strx,str);CHKERRQ(ierr);

  /* Create matrices */
  ierr = MatCreateSeqDense(PETSC_COMM_SELF,n,n,NULL,&A);CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject)A,"A");CHKERRQ(ierr);
  ierr = MatCreateSeqDense(PETSC_COMM_SELF,n,n,NULL,&B);CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject)B,"B");CHKERRQ(ierr);

  /* Fill A with a symmetric Toeplitz matrix */
  ierr = MatDenseGetArray(A,&As);CHKERRQ(ierr);
  for (i=0;i<n;i++) As[i+i*n]=2.0;
  for (j=1;j<3;j++) {
    for (i=0;i<n-j;i++) { As[i+(i+j)*n]=1.0; As[(i+j)+i*n]=1.0; }
  }
  ierr = MatDenseRestoreArray(A,&As);CHKERRQ(ierr);
  ierr = MatSetOption(A,MAT_HERMITIAN,PETSC_TRUE);CHKERRQ(ierr);
  if (verbose) {
    ierr = PetscPrintf(PETSC_COMM_WORLD,"Matrix A - - - - - - - -\n");CHKERRQ(ierr);
    ierr = MatView(A,viewer);CHKERRQ(ierr);
  }

  /* Evaluate matrix function */
  ierr = FNEvaluateFunctionMat(f,A,B);CHKERRQ(ierr);
  if (verbose) {
    ierr = PetscPrintf(PETSC_COMM_WORLD,"Computed f(A) - - - - - - -\n");CHKERRQ(ierr);
    ierr = MatView(B,viewer);CHKERRQ(ierr);
  }
  ierr = MatNorm(B,NORM_1,&nrm);CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD,"The 1-norm of f(A) is %g\n",(double)nrm);CHKERRQ(ierr);

  /* Repeat with same matrix as non-symmetric */
  ierr = MatSetOption(A,MAT_HERMITIAN,PETSC_FALSE);CHKERRQ(ierr);

  /* Evaluate matrix function */
  ierr = FNEvaluateFunctionMat(f,A,B);CHKERRQ(ierr);
  if (verbose) {
    ierr = PetscPrintf(PETSC_COMM_WORLD,"Computed f(A) - - - - - - -\n");CHKERRQ(ierr);
    ierr = MatView(B,viewer);CHKERRQ(ierr);
  }
  ierr = MatNorm(B,NORM_1,&nrm);CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD,"The 1-norm of f(A) is %g\n",(double)nrm);CHKERRQ(ierr);

  ierr = MatDestroy(&A);CHKERRQ(ierr);
  ierr = MatDestroy(&B);CHKERRQ(ierr);
  ierr = FNDestroy(&f);CHKERRQ(ierr);
  ierr = FNDestroy(&g);CHKERRQ(ierr);
  ierr = FNDestroy(&h);CHKERRQ(ierr);
  ierr = FNDestroy(&e);CHKERRQ(ierr);
  ierr = FNDestroy(&r);CHKERRQ(ierr);
  ierr = SlepcFinalize();
  return 0;
}
