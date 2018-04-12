/*
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
   SLEPc - Scalable Library for Eigenvalue Problem Computations
   Copyright (c) 2002-2018, Universitat Politecnica de Valencia, Spain

   This file is part of SLEPc.
   SLEPc is distributed under a 2-clause BSD license (see LICENSE).
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
*/

#include <petsc/private/fortranimpl.h>
#include <slepc/private/nepimpl.h>

#if defined(PETSC_HAVE_FORTRAN_CAPS)
#define nepnleigssetsingularitiesfunction_ nEPNLEIGSSETSINGULARITIESFUNCTION
#elif !defined(PETSC_HAVE_FORTRAN_UNDERSCORE)
#define nepnleigssetsingularitiesfunction_ nepnleigssetsingularitiesfunction
#endif

static struct {
  PetscFortranCallbackId singularities;
} _cb;

static PetscErrorCode oursingularitiesfunc(NEP nep,PetscInt *maxnp,PetscScalar *xi,void *ctx)
{
  PetscObjectUseFortranCallback(nep,_cb.singularities,(NEP*,PetscInt*,PetscScalar*,void*,PetscErrorCode*),(&nep,maxnp,xi,_ctx,&ierr));
}

PETSC_EXTERN void PETSC_STDCALL nepnleigssetsingularitiesfunction_(NEP *nep,void (PETSC_STDCALL *func)(NEP*,PetscInt*,PetscScalar*,void*,PetscErrorCode*),void *ctx,PetscErrorCode *ierr)
{
  CHKFORTRANNULLOBJECT(ctx);
  *ierr = PetscObjectSetFortranCallback((PetscObject)*nep,PETSC_FORTRAN_CALLBACK_CLASS,&_cb.singularities,(PetscVoidFunction)func,ctx); if (*ierr) return;
  *ierr = NEPNLEIGSSetSingularitiesFunction(*nep,oursingularitiesfunc,*nep);
}

