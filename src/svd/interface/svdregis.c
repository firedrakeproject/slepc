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

#include <slepc/private/svdimpl.h>       /*I "slepcsvd.h" I*/

PETSC_EXTERN PetscErrorCode SVDCreate_Cross(SVD);
PETSC_EXTERN PetscErrorCode SVDCreate_Cyclic(SVD);
PETSC_EXTERN PetscErrorCode SVDCreate_LAPACK(SVD);
PETSC_EXTERN PetscErrorCode SVDCreate_Lanczos(SVD);
PETSC_EXTERN PetscErrorCode SVDCreate_TRLanczos(SVD);

#undef __FUNCT__
#define __FUNCT__ "SVDRegisterAll"
/*@C
   SVDRegisterAll - Registers all the singular value solvers in the SVD package.

   Not Collective

   Level: advanced

.seealso:  SVDRegister()
@*/
PetscErrorCode SVDRegisterAll(void)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (SVDRegisterAllCalled) PetscFunctionReturn(0);
  SVDRegisterAllCalled = PETSC_TRUE;
  ierr = SVDRegister(SVDCROSS,SVDCreate_Cross);CHKERRQ(ierr);
  ierr = SVDRegister(SVDCYCLIC,SVDCreate_Cyclic);CHKERRQ(ierr);
  ierr = SVDRegister(SVDLAPACK,SVDCreate_LAPACK);CHKERRQ(ierr);
  ierr = SVDRegister(SVDLANCZOS,SVDCreate_Lanczos);CHKERRQ(ierr);
  ierr = SVDRegister(SVDTRLANCZOS,SVDCreate_TRLanczos);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

