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

static char help[] = "Test MatCreateTile.\n\n";

#include <slepcsys.h>

int main(int argc,char **argv)
{
  Mat            T,E,A;
  PetscInt       i,Istart,Iend,n=10;
  PetscErrorCode ierr;

  ierr = SlepcInitialize(&argc,&argv,(char*)0,help);if (ierr) return ierr;
  ierr = PetscOptionsGetInt(NULL,NULL,"-n",&n,NULL);CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD,"MatCreateTile test, n=%D\n",n);CHKERRQ(ierr);

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Create T=tridiag([-1 2 -1],n,n) and E=eye(n)
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

  ierr = MatCreate(PETSC_COMM_WORLD,&T);CHKERRQ(ierr);
  ierr = MatSetSizes(T,PETSC_DECIDE,PETSC_DECIDE,n,n);CHKERRQ(ierr);
  ierr = MatSetFromOptions(T);CHKERRQ(ierr);
  ierr = MatSetUp(T);CHKERRQ(ierr);

  ierr = MatGetOwnershipRange(T,&Istart,&Iend);CHKERRQ(ierr);
  for (i=Istart;i<Iend;i++) {
    if (i>0) { ierr = MatSetValue(T,i,i-1,-1.0,INSERT_VALUES);CHKERRQ(ierr); }
    if (i<n-1) { ierr = MatSetValue(T,i,i+1,-1.0,INSERT_VALUES);CHKERRQ(ierr); }
    ierr = MatSetValue(T,i,i,2.0,INSERT_VALUES);CHKERRQ(ierr);
  }
  ierr = MatAssemblyBegin(T,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(T,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);

  ierr = MatCreate(PETSC_COMM_WORLD,&E);CHKERRQ(ierr);
  ierr = MatSetSizes(E,PETSC_DECIDE,PETSC_DECIDE,n,n);CHKERRQ(ierr);
  ierr = MatSetFromOptions(E);CHKERRQ(ierr);
  ierr = MatSetUp(E);CHKERRQ(ierr);

  ierr = MatGetOwnershipRange(E,&Istart,&Iend);CHKERRQ(ierr);
  for (i=Istart;i<Iend;i++) {
    ierr = MatSetValue(E,i,i,1.0,INSERT_VALUES);CHKERRQ(ierr);
  }
  ierr = MatAssemblyBegin(E,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(E,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Create tiled matrix A = [ 2*T -E; 0 3*T ]
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  ierr = MatCreateTile(2.0,T,-1.0,E,0.0,E,3.0,T,&A);CHKERRQ(ierr);
  ierr = MatView(A,NULL);CHKERRQ(ierr);

  ierr = MatDestroy(&T);CHKERRQ(ierr);
  ierr = MatDestroy(&E);CHKERRQ(ierr);
  ierr = MatDestroy(&A);CHKERRQ(ierr);
  ierr = SlepcFinalize();
  return ierr;
}

