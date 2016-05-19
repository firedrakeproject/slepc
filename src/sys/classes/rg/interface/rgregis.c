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

#include <slepc/private/rgimpl.h>      /*I "slepcrg.h" I*/

PETSC_EXTERN PetscErrorCode RGCreate_Interval(RG);
PETSC_EXTERN PetscErrorCode RGCreate_Ellipse(RG);
PETSC_EXTERN PetscErrorCode RGCreate_Ring(RG);
PETSC_EXTERN PetscErrorCode RGCreate_Polygon(RG);

#undef __FUNCT__
#define __FUNCT__ "RGRegisterAll"
/*@C
   RGRegisterAll - Registers all of the regions in the RG package.

   Not Collective

   Level: advanced
@*/
PetscErrorCode RGRegisterAll(void)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (RGRegisterAllCalled) PetscFunctionReturn(0);
  RGRegisterAllCalled = PETSC_TRUE;
  ierr = RGRegister(RGINTERVAL,RGCreate_Interval);CHKERRQ(ierr);
  ierr = RGRegister(RGELLIPSE,RGCreate_Ellipse);CHKERRQ(ierr);
  ierr = RGRegister(RGRING,RGCreate_Ring);CHKERRQ(ierr);
  ierr = RGRegister(RGPOLYGON,RGCreate_Polygon);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

