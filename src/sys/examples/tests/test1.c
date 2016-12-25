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

/* "Demonstrates SlepcInitializeNoArguments()." */

#include <slepcsys.h>

int main(int argc,char **argv)
{
  PetscErrorCode ierr;
  PetscBool      isInitialized;

  ierr = SlepcInitialized(&isInitialized);if (ierr) return ierr;
  if (!isInitialized) {
    ierr = SlepcInitializeNoArguments();if (ierr) return ierr;
    ierr = PetscPrintf(PETSC_COMM_WORLD,"Initialize SLEPc.\n");CHKERRQ(ierr);
  } else {
    ierr = PetscPrintf(PETSC_COMM_WORLD,"SLEPc was already initialized.\n");CHKERRQ(ierr);
  }
  ierr = SlepcFinalize();
  return ierr;
}
