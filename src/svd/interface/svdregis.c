/*
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
   SLEPc - Scalable Library for Eigenvalue Problem Computations
   Copyright (c) 2002-2020, Universitat Politecnica de Valencia, Spain

   This file is part of SLEPc.
   SLEPc is distributed under a 2-clause BSD license (see LICENSE).
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
*/

#include <slepc/private/svdimpl.h>       /*I "slepcsvd.h" I*/

SLEPC_EXTERN PetscErrorCode SVDCreate_Cross(SVD);
SLEPC_EXTERN PetscErrorCode SVDCreate_Cyclic(SVD);
SLEPC_EXTERN PetscErrorCode SVDCreate_LAPACK(SVD);
SLEPC_EXTERN PetscErrorCode SVDCreate_Lanczos(SVD);
SLEPC_EXTERN PetscErrorCode SVDCreate_TRLanczos(SVD);
#if defined(SLEPC_HAVE_SCALAPACK)
SLEPC_EXTERN PetscErrorCode SVDCreate_ScaLAPACK(SVD);
#endif
#if defined(SLEPC_HAVE_ELEMENTAL)
SLEPC_EXTERN PetscErrorCode SVDCreate_Elemental(SVD);
#endif
#if defined(SLEPC_HAVE_PRIMME)
SLEPC_EXTERN PetscErrorCode SVDCreate_PRIMME(SVD);
#endif

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
#if defined(SLEPC_HAVE_SCALAPACK)
  ierr = SVDRegister(SVDSCALAPACK,SVDCreate_ScaLAPACK);CHKERRQ(ierr);
#endif
#if defined(SLEPC_HAVE_ELEMENTAL)
  ierr = SVDRegister(SVDELEMENTAL,SVDCreate_Elemental);CHKERRQ(ierr);
#endif
#if defined(SLEPC_HAVE_PRIMME)
  ierr = SVDRegister(SVDPRIMME,SVDCreate_PRIMME);CHKERRQ(ierr);
#endif
  PetscFunctionReturn(0);
}

