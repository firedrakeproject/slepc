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

static char help[] = "Test split reductions in BV.\n\n";

#include <slepcbv.h>

#undef __FUNCT__
#define __FUNCT__ "main"
int main(int argc,char **argv)
{
  PetscErrorCode ierr;
  Vec            t,v,w,y,z,zsplit;
  BV             X;
  PetscInt       i,j,n=10,k=5;
  PetscScalar    *zarray;
  PetscReal      nrm;

  SlepcInitialize(&argc,&argv,(char*)0,help);
  ierr = PetscOptionsGetInt(NULL,NULL,"-n",&n,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetInt(NULL,NULL,"-k",&k,NULL);CHKERRQ(ierr);
  if (k<3) SETERRQ(PETSC_COMM_SELF,1,"Should specify at least k=3 columns");
  ierr = PetscPrintf(PETSC_COMM_WORLD,"BV split ops (%D columns of dimension %D).\n",k,n);CHKERRQ(ierr);

  /* Create template vector */
  ierr = VecCreate(PETSC_COMM_WORLD,&t);CHKERRQ(ierr);
  ierr = VecSetSizes(t,PETSC_DECIDE,n);CHKERRQ(ierr);
  ierr = VecSetFromOptions(t);CHKERRQ(ierr);
  ierr = VecDuplicate(t,&v);CHKERRQ(ierr);
  ierr = VecSet(v,1.0);CHKERRQ(ierr);

  /* Create BV object X */
  ierr = BVCreate(PETSC_COMM_WORLD,&X);CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject)X,"X");CHKERRQ(ierr);
  ierr = BVSetSizesFromVec(X,t,k);CHKERRQ(ierr);
  ierr = BVSetFromOptions(X);CHKERRQ(ierr);

  /* Fill X entries */
  for (j=0;j<k;j++) {
    ierr = BVGetColumn(X,j,&w);CHKERRQ(ierr);
    ierr = VecSet(w,0.0);CHKERRQ(ierr);
    for (i=0;i<4;i++) {
      if (i+j<n) {
        ierr = VecSetValue(w,i+j,(PetscScalar)(3*i+j-2),INSERT_VALUES);CHKERRQ(ierr);
      }
    }
    ierr = VecAssemblyBegin(w);CHKERRQ(ierr);
    ierr = VecAssemblyEnd(w);CHKERRQ(ierr);
    ierr = BVRestoreColumn(X,j,&w);CHKERRQ(ierr);
  }

  /* Use regular operations */
  ierr = VecCreateSeq(PETSC_COMM_SELF,k+6,&z);CHKERRQ(ierr);
  ierr = VecGetArray(z,&zarray);CHKERRQ(ierr);
  ierr = BVGetColumn(X,0,&w);CHKERRQ(ierr);
  ierr = VecDot(w,v,zarray);CHKERRQ(ierr);
  ierr = BVRestoreColumn(X,0,&w);CHKERRQ(ierr);
  ierr = BVDotVec(X,v,zarray+1);CHKERRQ(ierr);
  ierr = BVDotColumn(X,2,zarray+1+k);CHKERRQ(ierr);

  ierr = BVGetColumn(X,1,&y);CHKERRQ(ierr);
  ierr = VecNorm(y,NORM_2,&nrm);CHKERRQ(ierr);
  ierr = BVRestoreColumn(X,1,&y);CHKERRQ(ierr);
  zarray[k+3] = nrm;
  ierr = BVNormVec(X,v,NORM_2,&nrm);CHKERRQ(ierr);
  zarray[k+4] = nrm;
  ierr = BVNormColumn(X,0,NORM_2,&nrm);CHKERRQ(ierr);
  zarray[k+5] = nrm;
  ierr = VecRestoreArray(z,&zarray);CHKERRQ(ierr);

  /* Use split operations */
  ierr = VecCreateSeq(PETSC_COMM_SELF,k+6,&zsplit);CHKERRQ(ierr);
  ierr = VecGetArray(zsplit,&zarray);CHKERRQ(ierr);
  ierr = BVGetColumn(X,0,&w);CHKERRQ(ierr);
  ierr = VecDotBegin(w,v,zarray);CHKERRQ(ierr);
  ierr = BVDotVecBegin(X,v,zarray+1);CHKERRQ(ierr);
  ierr = BVDotColumnBegin(X,2,zarray+1+k);CHKERRQ(ierr);
  ierr = VecDotEnd(w,v,zarray);CHKERRQ(ierr);
  ierr = BVRestoreColumn(X,0,&w);CHKERRQ(ierr);
  ierr = BVDotVecEnd(X,v,zarray+1);CHKERRQ(ierr);
  ierr = BVDotColumnEnd(X,2,zarray+1+k);CHKERRQ(ierr);

  ierr = BVGetColumn(X,1,&y);CHKERRQ(ierr);
  ierr = VecNormBegin(y,NORM_2,&nrm);CHKERRQ(ierr);
  ierr = BVNormVecBegin(X,v,NORM_2,&nrm);CHKERRQ(ierr);
  ierr = BVNormColumnBegin(X,0,NORM_2,&nrm);CHKERRQ(ierr);
  ierr = VecNormEnd(y,NORM_2,&nrm);CHKERRQ(ierr);
  ierr = BVRestoreColumn(X,1,&y);CHKERRQ(ierr);
  zarray[k+3] = nrm;
  ierr = BVNormVecEnd(X,v,NORM_2,&nrm);CHKERRQ(ierr);
  zarray[k+4] = nrm;
  ierr = BVNormColumnEnd(X,0,NORM_2,&nrm);CHKERRQ(ierr);
  zarray[k+5] = nrm;
  ierr = VecRestoreArray(zsplit,&zarray);CHKERRQ(ierr);

  /* Show difference */
  ierr = VecAXPY(z,-1.0,zsplit);CHKERRQ(ierr);
  ierr = VecNorm(z,NORM_1,&nrm);CHKERRQ(ierr);
  ierr = PetscSynchronizedPrintf(PETSC_COMM_WORLD,"%g\n",(double)nrm);CHKERRQ(ierr);
  ierr = PetscSynchronizedFlush(PETSC_COMM_WORLD,NULL);CHKERRQ(ierr);

  ierr = BVDestroy(&X);CHKERRQ(ierr);
  ierr = VecDestroy(&t);CHKERRQ(ierr);
  ierr = VecDestroy(&v);CHKERRQ(ierr);
  ierr = VecDestroy(&z);CHKERRQ(ierr);
  ierr = VecDestroy(&zsplit);CHKERRQ(ierr);
  ierr = SlepcFinalize();
  return ierr;
}
