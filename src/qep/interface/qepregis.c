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

#include <slepc-private/qepimpl.h>      /*I "slepcqep.h" I*/

PETSC_EXTERN PetscErrorCode QEPCreate_Linear(QEP);
PETSC_EXTERN PetscErrorCode QEPCreate_QArnoldi(QEP);
PETSC_EXTERN PetscErrorCode QEPCreate_QLanczos(QEP);
PETSC_EXTERN PetscErrorCode QEPCreate_STOAR(QEP);
PETSC_EXTERN PetscErrorCode QEPCreate_TOAR(QEP);

#undef __FUNCT__
#define __FUNCT__ "QEPRegisterAll"
/*@C
   QEPRegisterAll - Registers all the solvers in the QEP package.

   Not Collective

   Level: advanced

.seealso:  QEPRegister()
@*/
PetscErrorCode QEPRegisterAll(void)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  QEPRegisterAllCalled = PETSC_TRUE;
  ierr = QEPRegister(QEPLINEAR,QEPCreate_Linear);CHKERRQ(ierr);
  ierr = QEPRegister(QEPQARNOLDI,QEPCreate_QArnoldi);CHKERRQ(ierr);
  ierr = QEPRegister(QEPQLANCZOS,QEPCreate_QLanczos);CHKERRQ(ierr);
  ierr = QEPRegister(QEPSTOAR,QEPCreate_STOAR);CHKERRQ(ierr);
  ierr = QEPRegister(QEPTOAR,QEPCreate_TOAR);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
