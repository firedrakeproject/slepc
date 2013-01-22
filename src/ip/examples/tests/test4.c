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

static char help[] = "Test IPPseudoOrthogonalize.\n\n";

#include "slepcip.h"

#undef __FUNCT__
#define __FUNCT__ "main"
int main(int argc, char **argv)
{
  PetscErrorCode ierr;
  Mat            B;  
  IP             ip;
  Vec            *V,t;
  PetscInt       i,j,n=15,k=6,Istart,Iend;
  PetscRandom    rctx;
  PetscReal      lev,norm,*omega;
  PetscScalar    *vals;
  PetscBool      lindep;

  SlepcInitialize(&argc,&argv,(char*)0,help);
  ierr = PetscOptionsGetInt(PETSC_NULL,"-n",&n,PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetInt(PETSC_NULL,"-k",&k,PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD,"Pseudo-orthogonalization of %D random vectors of length %D.\n",k,n);CHKERRQ(ierr); 

  /* Create sip matrix (standard involutionary permutation) */
  ierr = MatCreate(PETSC_COMM_WORLD,&B);CHKERRQ(ierr);
  ierr = MatSetSizes(B,PETSC_DECIDE,PETSC_DECIDE,n,n);CHKERRQ(ierr);
  ierr = MatSetFromOptions(B);CHKERRQ(ierr);
  ierr = MatSetUp(B);CHKERRQ(ierr);
  ierr = MatGetOwnershipRange(B,&Istart,&Iend);CHKERRQ(ierr);
  for (i=Istart;i<Iend;i++) {
    ierr = MatSetValue(B,i,n-i-1,1.0,INSERT_VALUES);CHKERRQ(ierr);
  }
  ierr = MatAssemblyBegin(B,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(B,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  
  /* Create random vectors */
  ierr = PetscRandomCreate(PETSC_COMM_WORLD,&rctx);CHKERRQ(ierr);
  ierr = PetscRandomSetInterval(rctx,-1.0,1.0);CHKERRQ(ierr);
  ierr = PetscRandomSetFromOptions(rctx);CHKERRQ(ierr);
  ierr = MatGetVecs(B,PETSC_NULL,&t);CHKERRQ(ierr);
  ierr = VecDuplicateVecs(t,k,&V);CHKERRQ(ierr);
  for (i=0;i<k;i++) {
    ierr = VecSetRandom(V[i],rctx);CHKERRQ(ierr);
  }

  /* Create IP object */
  ierr = IPCreate(PETSC_COMM_WORLD,&ip);CHKERRQ(ierr);
  ierr = IPSetType(ip,IPINDEFINITE);CHKERRQ(ierr);
  ierr = IPSetMatrix(ip,B);CHKERRQ(ierr);
  ierr = IPSetFromOptions(ip);CHKERRQ(ierr);
  ierr = IPView(ip,PETSC_NULL);CHKERRQ(ierr);

  /* Orthogonalize random vectors */
  ierr = PetscMalloc(sizeof(PetscReal)*k,&omega);CHKERRQ(ierr);
  for (i=0;i<k;i++) {
    ierr = IPPseudoOrthogonalize(ip,i,V,omega,V[i],PETSC_NULL,&norm,&lindep);CHKERRQ(ierr);
    if (norm==0.0 || lindep) SETERRQ(((PetscObject)ip)->comm,1,"Linearly dependent vector found");
    ierr = VecScale(V[i],1.0/norm);CHKERRQ(ierr);
    omega[i] = norm/PetscAbs(norm);
  }

  /* Check orthogonality */
  ierr = PetscMalloc(k*sizeof(PetscScalar),&vals);CHKERRQ(ierr);
  lev = 0.0;
  for (i=0;i<k;i++) {
    ierr = MatMultTranspose(B,V[i],t);CHKERRQ(ierr);
    ierr = VecMDot(t,k,V,vals);CHKERRQ(ierr);
    for (j=0;j<k;j++) {
      lev = PetscMax(lev, PetscAbsScalar((j==i)? (vals[j]-omega[j]): vals[j]));
    }
  }
  ierr = PetscFree(vals);CHKERRQ(ierr);

  if (lev<100*PETSC_MACHINE_EPSILON) {
    ierr = PetscPrintf(PETSC_COMM_WORLD,"Level of orthogonality below 100*eps\n");CHKERRQ(ierr); 
  } else {
    ierr = PetscPrintf(PETSC_COMM_WORLD,"Level of orthogonality: %G\n",lev);CHKERRQ(ierr);
  }

  ierr = MatDestroy(&B);CHKERRQ(ierr);
  ierr = VecDestroyVecs(k,&V);CHKERRQ(ierr);
  ierr = IPDestroy(&ip);CHKERRQ(ierr);
  ierr = VecDestroy(&t);CHKERRQ(ierr);
  ierr = PetscRandomDestroy(&rctx);CHKERRQ(ierr);
  ierr = PetscFree(omega);CHKERRQ(ierr);
  ierr = SlepcFinalize();CHKERRQ(ierr);
  return 0;
}
