/*
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
      SLEPc - Scalable Library for Eigenvalue Problem Computations
      Copyright (c) 2002-2007, Universidad Politecnica de Valencia, Spain

      This file is part of SLEPc. See the README file for conditions of use
      and additional information.
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
*/

#include "src/st/stimpl.h"          /*I   "slepcst.h"   I*/

EXTERN_C_BEGIN
EXTERN PetscErrorCode STCreate_Shell(ST);
EXTERN PetscErrorCode STCreate_Shift(ST);
EXTERN PetscErrorCode STCreate_Sinvert(ST);
EXTERN PetscErrorCode STCreate_Cayley(ST);
EXTERN PetscErrorCode STCreate_Fold(ST);
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
PetscErrorCode STRegisterAll(char *path)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = STRegisterDynamic(STSHELL  ,path,"STCreate_Shell",STCreate_Shell);CHKERRQ(ierr);
  ierr = STRegisterDynamic(STSHIFT  ,path,"STCreate_Shift",STCreate_Shift);CHKERRQ(ierr);
  ierr = STRegisterDynamic(STSINV   ,path,"STCreate_Sinvert",STCreate_Sinvert);CHKERRQ(ierr);
  ierr = STRegisterDynamic(STCAYLEY ,path,"STCreate_Cayley",STCreate_Cayley);CHKERRQ(ierr);
  ierr = STRegisterDynamic(STFOLD   ,path,"STCreate_Fold",STCreate_Fold);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

