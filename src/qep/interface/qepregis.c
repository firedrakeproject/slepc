/*
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
   SLEPc - Scalable Library for Eigenvalue Problem Computations
   Copyright (c) 2002-2011, Universitat Politecnica de Valencia, Spain

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

#include <private/qepimpl.h>      /*I "slepcqep.h" I*/

EXTERN_C_BEGIN
extern PetscErrorCode QEPCreate_Linear(QEP);
extern PetscErrorCode QEPCreate_QArnoldi(QEP);
EXTERN_C_END
  
#undef __FUNCT__  
#define __FUNCT__ "QEPRegisterAll"
/*@C
   QEPRegisterAll - Registers all the solvers in the QEP package.

   Not Collective

   Level: advanced

.seealso:  QEPRegisterDynamic()
@*/
PetscErrorCode QEPRegisterAll(const char *path)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  QEPRegisterAllCalled = PETSC_TRUE;
  ierr = QEPRegisterDynamic(QEPLINEAR,path,"QEPCreate_Linear",QEPCreate_Linear);CHKERRQ(ierr);
  ierr = QEPRegisterDynamic(QEPQARNOLDI,path,"QEPCreate_QArnoldi",QEPCreate_QArnoldi);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
