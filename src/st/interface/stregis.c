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

#include <private/stimpl.h>          /*I   "slepcst.h"   I*/

EXTERN_C_BEGIN
extern PetscErrorCode STCreate_Shell(ST);
extern PetscErrorCode STCreate_Shift(ST);
extern PetscErrorCode STCreate_Sinvert(ST);
extern PetscErrorCode STCreate_Cayley(ST);
extern PetscErrorCode STCreate_Fold(ST);
extern PetscErrorCode STCreate_Precond(ST);
EXTERN_C_END

#undef __FUNCT__  
#define __FUNCT__ "STRegisterAll"
/*@C
   STRegisterAll - Registers all of the spectral transformations in the ST package.

   Not Collective

   Input Parameter:
.  path - the library where the routines are to be found (optional)

   Level: advanced

.seealso: STRegisterDynamic()
@*/
PetscErrorCode STRegisterAll(const char *path)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  STRegisterAllCalled = PETSC_TRUE;
  ierr = STRegisterDynamic(STSHELL  ,path,"STCreate_Shell",STCreate_Shell);CHKERRQ(ierr);
  ierr = STRegisterDynamic(STSHIFT  ,path,"STCreate_Shift",STCreate_Shift);CHKERRQ(ierr);
  ierr = STRegisterDynamic(STSINVERT,path,"STCreate_Sinvert",STCreate_Sinvert);CHKERRQ(ierr);
  ierr = STRegisterDynamic(STCAYLEY ,path,"STCreate_Cayley",STCreate_Cayley);CHKERRQ(ierr);
  ierr = STRegisterDynamic(STFOLD   ,path,"STCreate_Fold",STCreate_Fold);CHKERRQ(ierr);
  ierr = STRegisterDynamic(STPRECOND,path,"STCreate_Precond",STCreate_Precond);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

