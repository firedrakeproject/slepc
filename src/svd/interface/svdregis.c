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

#include <slepc-private/svdimpl.h>       /*I "slepcsvd.h" I*/

EXTERN_C_BEGIN
extern PetscErrorCode SVDCreate_Cross(SVD);
extern PetscErrorCode SVDCreate_Cyclic(SVD);
extern PetscErrorCode SVDCreate_LAPACK(SVD);
extern PetscErrorCode SVDCreate_Lanczos(SVD);
extern PetscErrorCode SVDCreate_TRLanczos(SVD);
EXTERN_C_END
  
#undef __FUNCT__
#define __FUNCT__ "SVDRegisterAll"
/*@C
   SVDRegisterAll - Registers all the singular value solvers in the SVD package.

   Not Collective

   Level: advanced

.seealso:  SVDRegisterDynamic()
@*/
PetscErrorCode SVDRegisterAll(const char *path)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  SVDRegisterAllCalled = PETSC_TRUE;
  ierr = SVDRegisterDynamic(SVDCROSS,path,"SVDCreate_Cross",SVDCreate_Cross);CHKERRQ(ierr);
  ierr = SVDRegisterDynamic(SVDCYCLIC,path,"SVDCreate_Cyclic",SVDCreate_Cyclic);CHKERRQ(ierr);
  ierr = SVDRegisterDynamic(SVDLAPACK,path,"SVDCreate_LAPACK",SVDCreate_LAPACK);CHKERRQ(ierr);
  ierr = SVDRegisterDynamic(SVDLANCZOS,path,"SVDCreate_Lanczos",SVDCreate_Lanczos);CHKERRQ(ierr);
  ierr = SVDRegisterDynamic(SVDTRLANCZOS,path,"SVDCreate_TRLanczoS",SVDCreate_TRLanczos);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
