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

#include <slepcst.h>
#include <slepcds.h>
#include <slepcfn.h>
#include <slepcbv.h>
#include <slepcrg.h>

#if defined(PETSC_HAVE_DYNAMIC_LIBRARIES)

#if defined(PETSC_USE_SINGLE_LIBRARY)
PETSC_EXTERN PetscErrorCode PetscDLLibraryRegister_slepceps(void);
PETSC_EXTERN PetscErrorCode PetscDLLibraryRegister_slepcnep(void);
PETSC_EXTERN PetscErrorCode PetscDLLibraryRegister_slepcpep(void);
PETSC_EXTERN PetscErrorCode PetscDLLibraryRegister_slepcsvd(void);
PETSC_EXTERN PetscErrorCode PetscDLLibraryRegister_slepcmfn(void);
#endif

#undef __FUNCT__
#if defined(PETSC_USE_SINGLE_LIBRARY)
#define __FUNCT__ "PetscDLLibraryRegister_slepc"
#else
#define __FUNCT__ "PetscDLLibraryRegister_slepcsys"
#endif
/*
  PetscDLLibraryRegister - This function is called when the dynamic library
  it is in is opened.

  This one registers all the basic objects ST, FN, DS, BV, RG.
 */
#if defined(PETSC_USE_SINGLE_LIBRARY)
PETSC_EXTERN PetscErrorCode PetscDLLibraryRegister_slepc(void)
#else
PETSC_EXTERN PetscErrorCode PetscDLLibraryRegister_slepcsys(void)
#endif
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = STInitializePackage();CHKERRQ(ierr);
  ierr = DSInitializePackage();CHKERRQ(ierr);
  ierr = FNInitializePackage();CHKERRQ(ierr);
  ierr = BVInitializePackage();CHKERRQ(ierr);
  ierr = RGInitializePackage();CHKERRQ(ierr);

#if defined(PETSC_USE_SINGLE_LIBRARY)
  ierr = PetscDLLibraryRegister_slepceps();CHKERRQ(ierr);
  ierr = PetscDLLibraryRegister_slepcnep();CHKERRQ(ierr);
  ierr = PetscDLLibraryRegister_slepcpep();CHKERRQ(ierr);
  ierr = PetscDLLibraryRegister_slepcsvd();CHKERRQ(ierr);
  ierr = PetscDLLibraryRegister_slepcmfn();CHKERRQ(ierr);
#endif
  PetscFunctionReturn(0);
}
#endif /* PETSC_HAVE_DYNAMIC_LIBRARIES */

