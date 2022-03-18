/*
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
   SLEPc - Scalable Library for Eigenvalue Problem Computations
   Copyright (c) 2002-2021, Universitat Politecnica de Valencia, Spain

   This file is part of SLEPc.
   SLEPc is distributed under a 2-clause BSD license (see LICENSE).
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
*/

#include <slepcst.h>
#include <slepcds.h>
#include <slepcfn.h>
#include <slepcbv.h>
#include <slepcrg.h>

#if defined(PETSC_HAVE_DYNAMIC_LIBRARIES)

#if defined(PETSC_USE_SINGLE_LIBRARY)
SLEPC_EXTERN PetscErrorCode PetscDLLibraryRegister_slepceps(void);
SLEPC_EXTERN PetscErrorCode PetscDLLibraryRegister_slepcnep(void);
SLEPC_EXTERN PetscErrorCode PetscDLLibraryRegister_slepcpep(void);
SLEPC_EXTERN PetscErrorCode PetscDLLibraryRegister_slepcsvd(void);
SLEPC_EXTERN PetscErrorCode PetscDLLibraryRegister_slepcmfn(void);
SLEPC_EXTERN PetscErrorCode PetscDLLibraryRegister_slepclme(void);
#endif

/*
  PetscDLLibraryRegister - This function is called when the dynamic library
  it is in is opened.

  This one registers all the basic objects ST, FN, DS, BV, RG.
 */
#if defined(PETSC_USE_SINGLE_LIBRARY)
SLEPC_EXTERN PetscErrorCode PetscDLLibraryRegister_slepc(void)
#else
SLEPC_EXTERN PetscErrorCode PetscDLLibraryRegister_slepcsys(void)
#endif
{
  PetscFunctionBegin;
  CHKERRQ(STInitializePackage());
  CHKERRQ(DSInitializePackage());
  CHKERRQ(FNInitializePackage());
  CHKERRQ(BVInitializePackage());
  CHKERRQ(RGInitializePackage());

#if defined(PETSC_USE_SINGLE_LIBRARY)
  CHKERRQ(PetscDLLibraryRegister_slepceps());
  CHKERRQ(PetscDLLibraryRegister_slepcnep());
  CHKERRQ(PetscDLLibraryRegister_slepcpep());
  CHKERRQ(PetscDLLibraryRegister_slepcsvd());
  CHKERRQ(PetscDLLibraryRegister_slepcmfn());
  CHKERRQ(PetscDLLibraryRegister_slepclme());
#endif
  PetscFunctionReturn(0);
}
#endif /* PETSC_HAVE_DYNAMIC_LIBRARIES */
