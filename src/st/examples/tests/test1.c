/*
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
   SLEPc - Scalable Library for Eigenvalue Problem Computations
   Copyright (c) 2002-2012, Universitat Politecnica de Valencia, Spain

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

static char help[] = "Test ST with shell matrices as problem matrices.\n\n";

#include <slepceps.h>

static PetscErrorCode MatShift_Shell(Mat S,PetscScalar shift);
static PetscErrorCode MatGetDiagonal_Shell(Mat S,Vec diag);
static PetscErrorCode MatMultTranspose_Shell(Mat S,Vec x,Vec y);
static PetscErrorCode MatMult_Shell(Mat S,Vec x,Vec y);

#undef __FUNCT__
#define __FUNCT__ "main"
int main(int argc,char **argv)
{
  Mat            A, S;        /* problem matrix */
  EPS            eps;         /* eigenproblem solver context */
  const EPSType  type;
  PetscScalar    value[3];
  PetscInt       n=30,i,Istart,Iend,col[3],nev;
  PetscBool      FirstBlock=PETSC_FALSE,LastBlock=PETSC_FALSE;
  PetscErrorCode ierr;

  SlepcInitialize(&argc,&argv,(char*)0,help);

  ierr = PetscOptionsGetInt(PETSC_NULL,"-n",&n,PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD,"\n1-D Laplacian Eigenproblem, n=%D\n\n",n);CHKERRQ(ierr);

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
     Compute the operator matrix that defines the eigensystem, Ax=kx
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

  ierr = MatCreate(PETSC_COMM_WORLD,&A);CHKERRQ(ierr);
  ierr = MatSetSizes(A,PETSC_DECIDE,PETSC_DECIDE,n,n);CHKERRQ(ierr);
  ierr = MatSetFromOptions(A);CHKERRQ(ierr);
  ierr = MatSetUp(A);CHKERRQ(ierr);
  
  ierr = MatGetOwnershipRange(A,&Istart,&Iend);CHKERRQ(ierr);
  if (Istart==0) FirstBlock=PETSC_TRUE;
  if (Iend==n) LastBlock=PETSC_TRUE;
  value[0]=-1.0; value[1]=2.0; value[2]=-1.0;
  for (i=(FirstBlock? Istart+1: Istart); i<(LastBlock? Iend-1: Iend); i++) {
    col[0]=i-1; col[1]=i; col[2]=i+1;
    ierr = MatSetValues(A,1,&i,3,col,value,INSERT_VALUES);CHKERRQ(ierr);
  }
  if (LastBlock) {
    i=n-1; col[0]=n-2; col[1]=n-1;
    ierr = MatSetValues(A,1,&i,2,col,value,INSERT_VALUES);CHKERRQ(ierr);
  }
  if (FirstBlock) {
    i=0; col[0]=0; col[1]=1; value[0]=2.0; value[1]=-1.0;
    ierr = MatSetValues(A,1,&i,2,col,value,INSERT_VALUES);CHKERRQ(ierr);
  }

  ierr = MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);

  /* Create the shell version of A */
  ierr = MatCreateShell(PETSC_COMM_WORLD,PETSC_DECIDE,PETSC_DECIDE,n,n,&A,&S);CHKERRQ(ierr);
  ierr = MatSetFromOptions(S);CHKERRQ(ierr);
  ierr = MatShellSetOperation(S,MATOP_MULT,(void(*)())MatMult_Shell);CHKERRQ(ierr);
  ierr = MatShellSetOperation(S,MATOP_MULT_TRANSPOSE,(void(*)())MatMultTranspose_Shell);CHKERRQ(ierr);
  ierr = MatShellSetOperation(S,MATOP_GET_DIAGONAL,(void(*)())MatGetDiagonal_Shell);CHKERRQ(ierr);
  ierr = MatShellSetOperation(S,MATOP_SHIFT,(void(*)())MatShift_Shell);CHKERRQ(ierr);

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
                Create the eigensolver and set various options
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

  ierr = EPSCreate(PETSC_COMM_WORLD,&eps);CHKERRQ(ierr);
  ierr = EPSSetOperators(eps,S,PETSC_NULL);CHKERRQ(ierr);
  ierr = EPSSetProblemType(eps,EPS_HEP);CHKERRQ(ierr);
  ierr = EPSSetFromOptions(eps);CHKERRQ(ierr);

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
                      Solve the eigensystem
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

  ierr = EPSSolve(eps);CHKERRQ(ierr);
  ierr = EPSGetType(eps,&type);CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD," Solution method: %s\n\n",type);CHKERRQ(ierr);
  ierr = EPSGetDimensions(eps,&nev,PETSC_NULL,PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD," Number of requested eigenvalues: %D\n",nev);CHKERRQ(ierr);

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
                    Display solution and clean up
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

  ierr = EPSPrintSolution(eps,PETSC_NULL);CHKERRQ(ierr);
  ierr = EPSDestroy(&eps);CHKERRQ(ierr);
  ierr = MatDestroy(&A);CHKERRQ(ierr);
  ierr = MatDestroy(&S);CHKERRQ(ierr);
  ierr = SlepcFinalize();CHKERRQ(ierr);
  return 0;
}

#undef __FUNCT__
#define __FUNCT__ "MatMult_Shell"
static PetscErrorCode MatMult_Shell(Mat S,Vec x,Vec y)
{
  PetscErrorCode    ierr;
  Mat               *A;
  
  PetscFunctionBegin;
  ierr = MatShellGetContext(S,&A);CHKERRQ(ierr);
  ierr = MatMult(*A,x,y);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatMultTranspose_Shell"
static PetscErrorCode MatMultTranspose_Shell(Mat S,Vec x,Vec y)
{
  PetscErrorCode    ierr;
  Mat               *A;
  
  PetscFunctionBegin;
  ierr = MatShellGetContext(S,&A);CHKERRQ(ierr);
  ierr = MatMultTranspose(*A,x,y);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatGetDiagonal_Shell"
static PetscErrorCode MatGetDiagonal_Shell(Mat S,Vec diag)
{
  PetscErrorCode    ierr;
  Mat               *A;
  
  PetscFunctionBegin;
  ierr = MatShellGetContext(S,&A);CHKERRQ(ierr);
  ierr = MatGetDiagonal(*A,diag);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatShift_Shell"
static PetscErrorCode MatShift_Shell(Mat S,PetscScalar shift)
{
  PetscErrorCode    ierr;
  Mat               *A;
  
  PetscFunctionBegin;
  ierr = MatShellGetContext(S,&A);CHKERRQ(ierr);
  ierr = MatShift(*A,shift);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
 
