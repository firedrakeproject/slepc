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

static char help[] = "Test setting FN parameters from the command line.\n\n";

#include <slepcfn.h>

#undef __FUNCT__
#define __FUNCT__ "main"
int main(int argc,char **argv)
{
  PetscErrorCode ierr;
  FN             fn1,fn2;

  SlepcInitialize(&argc,&argv,(char*)0,help);
  ierr = FNCreate(PETSC_COMM_WORLD,&fn1);CHKERRQ(ierr);
  ierr = FNSetOptionsPrefix(fn1,"f1_");CHKERRQ(ierr);
  ierr = FNSetFromOptions(fn1);CHKERRQ(ierr);
  ierr = FNView(fn1,NULL);CHKERRQ(ierr);
  ierr = FNDestroy(&fn1);CHKERRQ(ierr);
  ierr = FNCreate(PETSC_COMM_WORLD,&fn2);CHKERRQ(ierr);
  ierr = FNSetOptionsPrefix(fn2,"f2_");CHKERRQ(ierr);
  ierr = FNSetFromOptions(fn2);CHKERRQ(ierr);
  ierr = FNView(fn2,NULL);CHKERRQ(ierr);
  ierr = FNDestroy(&fn2);CHKERRQ(ierr);
  ierr = SlepcFinalize();
  return ierr;
}
