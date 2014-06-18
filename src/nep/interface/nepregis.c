/*
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
   SLEPc - Scalable Library for Eigenvalue Problem Computations
   Copyright (c) 2002-2013, Universitat Politecnica de Valencia, Spain

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

#include <slepc-private/nepimpl.h>      /*I "slepcnep.h" I*/

PETSC_EXTERN PetscErrorCode NEPCreate_RII(NEP);
PETSC_EXTERN PetscErrorCode NEPCreate_SLP(NEP);
PETSC_EXTERN PetscErrorCode NEPCreate_NArnoldi(NEP);
PETSC_EXTERN PetscErrorCode NEPCreate_NLEIGS(NEP);

#undef __FUNCT__
#define __FUNCT__ "NEPRegisterAll"
/*@C
   NEPRegisterAll - Registers all the solvers in the NEP package.

   Not Collective

   Level: advanced

.seealso:  NEPRegister()
@*/
PetscErrorCode NEPRegisterAll(void)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  NEPRegisterAllCalled = PETSC_TRUE;
  ierr = NEPRegister(NEPRII,NEPCreate_RII);CHKERRQ(ierr);
  ierr = NEPRegister(NEPSLP,NEPCreate_SLP);CHKERRQ(ierr);
  ierr = NEPRegister(NEPNARNOLDI,NEPCreate_NArnoldi);CHKERRQ(ierr);
  /*ierr = NEPRegister(NEPNLEIGS,NEPCreate_NLEIGS);CHKERRQ(ierr);*/
  PetscFunctionReturn(0);
}
