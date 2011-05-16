/*
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
   SLEPc - Scalable Library for Eigenvalue Problem Computations
   Copyright (c) 2002-2010, Universidad Politecnica de Valencia, Spain

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

#include "slepcip.h"
#include "slepcvec.h"

#undef __FUNCT__
#define __FUNCT__ "main"
int main( int argc, char **argv )
{
  PetscErrorCode ierr;
  IP             ip;
  Vec            *V,t;
  PetscInt       i,n=15,k=6;
  PetscRandom    rctx;
  PetscScalar    lev;
  PetscBool      cont;

  SlepcInitialize(&argc,&argv,(char*)0,help);
  ierr = PetscPrintf(PETSC_COMM_WORLD,"QR decomposition of random vectors.\n",lev);CHKERRQ(ierr); 
  ierr = PetscRandomCreate(PETSC_COMM_WORLD,&rctx);CHKERRQ(ierr);
  ierr = PetscRandomSetFromOptions(rctx);CHKERRQ(ierr);
  ierr = VecCreate(PETSC_COMM_WORLD,&t);CHKERRQ(ierr);
  ierr = VecSetSizes(t,n,PETSC_DECIDE);CHKERRQ(ierr);
  ierr = VecSetFromOptions(t);CHKERRQ(ierr);
  ierr = IPCreate(PETSC_COMM_WORLD,&ip);CHKERRQ(ierr);
  ierr = IPSetFromOptions(ip);CHKERRQ(ierr);
  ierr = IPView(ip,PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscOptionsHasName(PETSC_NULL,"-contiguous",&cont);CHKERRQ(ierr);

  /* with/without contiguous storage */
  if (cont) {
    ierr = SlepcVecDuplicateVecs(t,k,&V);CHKERRQ(ierr);
    ierr = PetscPrintf(PETSC_COMM_WORLD,"With contiguous storage.\n",lev);CHKERRQ(ierr); 
  } else {
    ierr = VecDuplicateVecs(t,k,&V);CHKERRQ(ierr);
    ierr = PetscPrintf(PETSC_COMM_WORLD,"With regular storage.\n",lev);CHKERRQ(ierr); 
  }

  /* check orthogonality of QR of random vectors */
  for (i=0;i<k;i++) { ierr = VecSetRandom(V[i],rctx);CHKERRQ(ierr); }
  ierr = IPQRDecomposition(ip,V,0,k,PETSC_NULL,k);CHKERRQ(ierr);
  ierr = SlepcCheckOrthogonality(V,k,PETSC_NULL,k,PETSC_NULL,&lev);CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD,"Level of orthogonality: %A\n",lev);CHKERRQ(ierr); 

  if (cont) { ierr = SlepcVecDestroyVecs(k,&V);CHKERRQ(ierr); }
  else { ierr = VecDestroyVecs(k,&V);CHKERRQ(ierr); }
  ierr = IPDestroy(&ip);CHKERRQ(ierr);
  ierr = VecDestroy(&t);CHKERRQ(ierr);
  ierr = PetscRandomDestroy(&rctx);CHKERRQ(ierr);
  ierr = SlepcFinalize();CHKERRQ(ierr);
  return 0;
}
