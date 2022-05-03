/*
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
   SLEPc - Scalable Library for Eigenvalue Problem Computations
   Copyright (c) 2002-, Universitat Politecnica de Valencia, Spain

   This file is part of SLEPc.
   SLEPc is distributed under a 2-clause BSD license (see LICENSE).
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
*/

#include <slepc/private/slepcimpl.h>
#include <slepcmagma.h>

static PetscBool SlepcBeganMagma = PETSC_FALSE;

PetscErrorCode SlepcMagmaInit(void)
{
  PetscFunctionBegin;
  if (!SlepcBeganMagma) {
    magma_init();
    SlepcBeganMagma = PETSC_TRUE;
    PetscCall(PetscRegisterFinalize(magma_finalize));
  }
  PetscFunctionReturn(0);
}
