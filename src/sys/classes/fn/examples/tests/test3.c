/*
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
   SLEPc - Scalable Library for Eigenvalue Problem Computations
   Copyright (c) 2002-2016, Universitat Politecnica de Valencia, Spain

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

static char help[] = "Test matrix exponential.\n\n";

#include <slepcfn.h>

/*
   Compute matrix exponential B = expm(A)
 */
PetscErrorCode TestMatExp(FN fn,Mat A,PetscViewer viewer,PetscBool verbose,PetscBool inplace)
{
  PetscErrorCode ierr;
  PetscBool      set,flg;
  PetscInt       n;
  Mat            F;
  Vec            v,f0;
  PetscReal      nrm;

  PetscFunctionBeginUser;
  ierr = MatGetSize(A,&n,NULL);CHKERRQ(ierr);
  ierr = MatCreateSeqDense(PETSC_COMM_SELF,n,n,NULL,&F);CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject)F,"F");CHKERRQ(ierr);
  /* compute square root */
  if (inplace) {
    ierr = MatCopy(A,F,SAME_NONZERO_PATTERN);CHKERRQ(ierr);
    ierr = MatIsHermitianKnown(A,&set,&flg);CHKERRQ(ierr);
    if (set && flg) { ierr = MatSetOption(F,MAT_HERMITIAN,PETSC_TRUE);CHKERRQ(ierr); }
    ierr = FNEvaluateFunctionMat(fn,F,NULL);CHKERRQ(ierr);
  } else {
    ierr = FNEvaluateFunctionMat(fn,A,F);CHKERRQ(ierr);
  }
  if (verbose) {
    ierr = PetscPrintf(PETSC_COMM_WORLD,"Matrix A - - - - - - - -\n");CHKERRQ(ierr);
    ierr = MatView(A,viewer);CHKERRQ(ierr);
    ierr = PetscPrintf(PETSC_COMM_WORLD,"Computed expm(A) - - - - - - -\n");CHKERRQ(ierr);
    ierr = MatView(F,viewer);CHKERRQ(ierr);
  }
  /* print matrix norm for checking */
  ierr = MatNorm(F,NORM_1,&nrm);CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD,"The 1-norm of f(A) is %g\n",(double)nrm);CHKERRQ(ierr);
  /* check FNEvaluateFunctionMatVec() */
  ierr = MatCreateVecs(A,&v,&f0);CHKERRQ(ierr);
  ierr = MatGetColumnVector(F,f0,0);CHKERRQ(ierr);
  ierr = FNEvaluateFunctionMatVec(fn,A,v);CHKERRQ(ierr);
  ierr = VecAXPY(v,-1.0,f0);CHKERRQ(ierr);
  ierr = VecNorm(v,NORM_2,&nrm);CHKERRQ(ierr);
  if (nrm>100*PETSC_MACHINE_EPSILON) {
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
  PetscScalar    *As,tau=1.0,eta=1.0;
  PetscViewer    viewer;
  PetscBool      verbose,inplace;

  ierr = SlepcInitialize(&argc,&argv,(char*)0,help);if (ierr) return ierr;
  ierr = PetscOptionsGetInt(NULL,NULL,"-n",&n,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetScalar(NULL,NULL,"-tau",&tau,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetScalar(NULL,NULL,"-eta",&eta,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsHasName(NULL,NULL,"-verbose",&verbose);CHKERRQ(ierr);
  ierr = PetscOptionsHasName(NULL,NULL,"-inplace",&inplace);CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD,"Matrix exponential, n=%D.\n",n);CHKERRQ(ierr);

  /* Create exponential function eta*exp(tau*x) */
  ierr = FNCreate(PETSC_COMM_WORLD,&fn);CHKERRQ(ierr);
  ierr = FNSetType(fn,FNEXP);CHKERRQ(ierr);
  ierr = FNSetScale(fn,tau,eta);CHKERRQ(ierr);

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
  ierr = TestMatExp(fn,A,viewer,verbose,inplace);CHKERRQ(ierr);

  /* Repeat with non-symmetric A */
  ierr = MatDenseGetArray(A,&As);CHKERRQ(ierr);
  for (j=1;j<3;j++) {
    for (i=0;i<n-j;i++) { As[(i+j)+i*n]=-1.0; }
  }
  ierr = MatDenseRestoreArray(A,&As);CHKERRQ(ierr);
  ierr = MatSetOption(A,MAT_HERMITIAN,PETSC_FALSE);CHKERRQ(ierr);
  ierr = TestMatExp(fn,A,viewer,verbose,inplace);CHKERRQ(ierr);

  ierr = MatDestroy(&A);CHKERRQ(ierr);
  ierr = FNDestroy(&fn);CHKERRQ(ierr);
  ierr = SlepcFinalize();
  return ierr;
}
