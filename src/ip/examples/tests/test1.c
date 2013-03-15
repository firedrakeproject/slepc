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

static char help[] = "Test IPQRDecomposition.\n\n";

#include <slepcip.h>
#include <slepcvec.h>

#undef __FUNCT__
#define __FUNCT__ "main"
int main(int argc,char **argv)
{
  PetscErrorCode ierr;
  IP             ip;
  Vec            *V,t;
  PetscInt       i,n=15,k=6;
  PetscRandom    rctx;
  PetscReal      lev;
  PetscBool      cont;

  SlepcInitialize(&argc,&argv,(char*)0,help);
  ierr = PetscOptionsGetInt(NULL,"-n",&n,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetInt(NULL,"-k",&k,NULL);CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD,"QR decomposition of %D random vectors of length %D.\n",k,n);CHKERRQ(ierr); 
  ierr = PetscRandomCreate(PETSC_COMM_WORLD,&rctx);CHKERRQ(ierr);
  ierr = PetscRandomSetFromOptions(rctx);CHKERRQ(ierr);
  ierr = VecCreate(PETSC_COMM_WORLD,&t);CHKERRQ(ierr);
  ierr = VecSetSizes(t,n,PETSC_DECIDE);CHKERRQ(ierr);
  ierr = VecSetFromOptions(t);CHKERRQ(ierr);
  ierr = IPCreate(PETSC_COMM_WORLD,&ip);CHKERRQ(ierr);
  ierr = IPSetFromOptions(ip);CHKERRQ(ierr);
  ierr = IPView(ip,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsHasName(NULL,"-contiguous",&cont);CHKERRQ(ierr);

  /* with/without contiguous storage */
  if (cont) {
    ierr = SlepcVecSetTemplate(t);CHKERRQ(ierr);
    ierr = PetscPrintf(PETSC_COMM_WORLD,"With contiguous storage.\n");CHKERRQ(ierr); 
  } else {
    ierr = PetscPrintf(PETSC_COMM_WORLD,"With regular storage.\n");CHKERRQ(ierr); 
  }
  ierr = VecDuplicateVecs(t,k,&V);CHKERRQ(ierr);

  /* check orthogonality of QR of random vectors */
  for (i=0;i<k;i++) {
    ierr = VecSetRandom(V[i],rctx);CHKERRQ(ierr);
  }
  ierr = IPQRDecomposition(ip,V,0,k,NULL,k);CHKERRQ(ierr);
  ierr = SlepcCheckOrthogonality(V,k,NULL,k,NULL,NULL,&lev);CHKERRQ(ierr);
  if (lev<100*PETSC_MACHINE_EPSILON) {
    ierr = PetscPrintf(PETSC_COMM_WORLD,"Level of orthogonality below 100*eps\n");CHKERRQ(ierr); 
  } else {
    ierr = PetscPrintf(PETSC_COMM_WORLD,"Level of orthogonality: %G\n",lev);CHKERRQ(ierr); 
  }

  ierr = VecDestroyVecs(k,&V);CHKERRQ(ierr);
  ierr = IPDestroy(&ip);CHKERRQ(ierr);
  ierr = VecDestroy(&t);CHKERRQ(ierr);
  ierr = PetscRandomDestroy(&rctx);CHKERRQ(ierr);
  ierr = SlepcFinalize();
  return 0;
}
