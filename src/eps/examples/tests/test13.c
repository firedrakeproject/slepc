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

static char help[] = "Test EPSSetArbitrarySelection.\n\n";

#include <slepceps.h>

#undef __FUNCT__
#define __FUNCT__ "MyArbitrarySelection"
PetscErrorCode MyArbitrarySelection(PetscScalar eigr,PetscScalar eigi,Vec xr,Vec xi,PetscScalar *rr,PetscScalar *ri,void *ctx)
{
  PetscErrorCode  ierr;
  Vec             xref = *(Vec*)ctx;

  PetscFunctionBeginUser;
  ierr = VecDot(xr,xref,rr);CHKERRQ(ierr);
  *rr = PetscAbsScalar(*rr);
  if (ri) *ri = 0.0;
  PetscFunctionReturn(0);
}

#undef __funct__
#define __funct__ "main"
int main(int argc,char **argv)
{
  Mat            A;           /* problem matrices */
  EPS            eps;         /* eigenproblem solver context */
  PetscScalar    seigr,seigi;
  PetscReal      tol=1000*PETSC_MACHINE_EPSILON;
  Vec            sxr,sxi;
  PetscInt       n=30,i,Istart,Iend,nconv;
  PetscErrorCode ierr;

  SlepcInitialize(&argc,&argv,(char*)0,help);

  ierr = PetscOptionsGetInt(NULL,NULL,"-n",&n,NULL);CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD,"\nTridiagonal with zero diagonal, n=%D\n\n",n);CHKERRQ(ierr);

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
           Create matrix tridiag([-1 0 -1])
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  ierr = MatCreate(PETSC_COMM_WORLD,&A);CHKERRQ(ierr);
  ierr = MatSetSizes(A,PETSC_DECIDE,PETSC_DECIDE,n,n);CHKERRQ(ierr);
  ierr = MatSetFromOptions(A);CHKERRQ(ierr);
  ierr = MatSetUp(A);CHKERRQ(ierr);

  ierr = MatGetOwnershipRange(A,&Istart,&Iend);CHKERRQ(ierr);
  for (i=Istart;i<Iend;i++) {
    if (i>0) { ierr = MatSetValue(A,i,i-1,-1.0,INSERT_VALUES);CHKERRQ(ierr); }
    if (i<n-1) { ierr = MatSetValue(A,i,i+1,-1.0,INSERT_VALUES);CHKERRQ(ierr); }
  }
  ierr = MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
                        Create the eigensolver
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  ierr = EPSCreate(PETSC_COMM_WORLD,&eps);CHKERRQ(ierr);
  ierr = EPSSetProblemType(eps,EPS_HEP);CHKERRQ(ierr);
  ierr = EPSSetTolerances(eps,tol,PETSC_DEFAULT);CHKERRQ(ierr);
  ierr = EPSSetOperators(eps,A,NULL);CHKERRQ(ierr);
  ierr = EPSSetWhichEigenpairs(eps,EPS_SMALLEST_REAL);CHKERRQ(ierr);
  ierr = EPSSetFromOptions(eps);CHKERRQ(ierr);

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
                Solve eigenproblem and store some solution
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  ierr = EPSSolve(eps);CHKERRQ(ierr);
  ierr = MatCreateVecs(A,&sxr,NULL);CHKERRQ(ierr);
  ierr = MatCreateVecs(A,&sxi,NULL);CHKERRQ(ierr);
  ierr = EPSGetConverged(eps,&nconv);CHKERRQ(ierr);
  if (nconv>0) {
    ierr = EPSGetEigenpair(eps,0,&seigr,&seigi,sxr,sxi);CHKERRQ(ierr);
    ierr = EPSErrorView(eps,EPS_ERROR_RELATIVE,NULL);CHKERRQ(ierr);

    /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
                 Solve eigenproblem using an arbitrary selection
       - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
    ierr = EPSSetArbitrarySelection(eps,MyArbitrarySelection,&sxr);CHKERRQ(ierr);
    ierr = EPSSetWhichEigenpairs(eps,EPS_LARGEST_MAGNITUDE);CHKERRQ(ierr);
    ierr = EPSSolve(eps);CHKERRQ(ierr);
    ierr = EPSErrorView(eps,EPS_ERROR_RELATIVE,NULL);CHKERRQ(ierr);
  } else {
    ierr = PetscPrintf(PETSC_COMM_WORLD,"Problem: no eigenpairs converged.\n");CHKERRQ(ierr);
  }

  ierr = EPSDestroy(&eps);CHKERRQ(ierr);
  ierr = VecDestroy(&sxr);CHKERRQ(ierr);
  ierr = VecDestroy(&sxi);CHKERRQ(ierr);
  ierr = MatDestroy(&A);CHKERRQ(ierr);
  ierr = SlepcFinalize();
  return ierr;
}
