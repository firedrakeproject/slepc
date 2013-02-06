/*
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
   SLEPc - Scalable Library for Eigenvalue Problem Computations
   Copyright (c) 2002-2012, Universitat Politecnica de Valencia, Spain

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

EXTERN_C_BEGIN
extern PetscErrorCode NEPCreate_RII(NEP);
EXTERN_C_END
  
#undef __FUNCT__  
#define __FUNCT__ "NEPRegisterAll"
/*@C
   NEPRegisterAll - Registers all the solvers in the NEP package.

   Not Collective

   Level: advanced

.seealso:  NEPRegisterDynamic()
@*/
PetscErrorCode NEPRegisterAll(const char *path)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  NEPRegisterAllCalled = PETSC_TRUE;
  ierr = NEPRegisterDynamic(NEPRII,path,"NEPCreate_RII",NEPCreate_RII);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
