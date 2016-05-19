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

#include <slepc/private/pepimpl.h>      /*I "slepcpep.h" I*/

PETSC_EXTERN PetscErrorCode PEPCreate_Linear(PEP);
PETSC_EXTERN PetscErrorCode PEPCreate_QArnoldi(PEP);
PETSC_EXTERN PetscErrorCode PEPCreate_TOAR(PEP);
PETSC_EXTERN PetscErrorCode PEPCreate_STOAR(PEP);
PETSC_EXTERN PetscErrorCode PEPCreate_JD(PEP);

#undef __FUNCT__
#define __FUNCT__ "PEPRegisterAll"
/*@C
   PEPRegisterAll - Registers all the solvers in the PEP package.

   Not Collective

   Level: advanced

.seealso:  PEPRegister()
@*/
PetscErrorCode PEPRegisterAll(void)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (PEPRegisterAllCalled) PetscFunctionReturn(0);
  PEPRegisterAllCalled = PETSC_TRUE;
  ierr = PEPRegister(PEPLINEAR,PEPCreate_Linear);CHKERRQ(ierr);
  ierr = PEPRegister(PEPQARNOLDI,PEPCreate_QArnoldi);CHKERRQ(ierr);
  ierr = PEPRegister(PEPTOAR,PEPCreate_TOAR);CHKERRQ(ierr);
  ierr = PEPRegister(PEPSTOAR,PEPCreate_STOAR);CHKERRQ(ierr);
  ierr = PEPRegister(PEPJD,PEPCreate_JD);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

