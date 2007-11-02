/*
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
      SLEPc - Scalable Library for Eigenvalue Problem Computations
      Copyright (c) 2002-2007, Universidad Politecnica de Valencia, Spain

      This file is part of SLEPc. See the README file for conditions of use
      and additional information.
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
*/

#include "private/zpetsc.h"
#include "slepcst.h"

#ifdef PETSC_HAVE_FORTRAN_CAPS
#define stshellsetapply_          STSHELLSETAPPLY
#define stshellsetapplytranspose_ STSHELLSETAPPLYTRANSPOSE
#define stshellsetbacktransform_  STSHELLSETBACKTRANSFORM
#elif !defined(PETSC_HAVE_FORTRAN_UNDERSCORE)
#define stshellsetapply_          stshellsetapply
#define stshellsetapplytranspose_ stshellsetapplytranspose
#define stshellsetbacktransform_  stshellsetbacktransform
#endif

EXTERN_C_BEGIN
static void (PETSC_STDCALL *f1)(void*,Vec*,Vec*,PetscErrorCode*);
static void (PETSC_STDCALL *f2)(void*,Vec*,Vec*,PetscErrorCode*);
static void (PETSC_STDCALL *f3)(void*,PetscScalar*,PetscScalar*,PetscErrorCode*);
EXTERN_C_END

/* These are not extern C because they are passed into non-extern C user level functions */
static PetscErrorCode ourshellapply(void *ctx,Vec x,Vec y)
{
  PetscErrorCode ierr = 0;
  (*f1)(ctx,&x,&y,&ierr);CHKERRQ(ierr);
  return 0;
}

static PetscErrorCode ourshellapplytranspose(void *ctx,Vec x,Vec y)
{
  PetscErrorCode ierr = 0;
  (*f2)(ctx,&x,&y,&ierr);CHKERRQ(ierr);
  return 0;
}

static PetscErrorCode ourshellbacktransform(void *ctx,PetscScalar *eigr,PetscScalar *eigi)
{
  PetscErrorCode ierr = 0;
  (*f3)(ctx,eigr,eigi,&ierr);CHKERRQ(ierr);
  return 0;
}

EXTERN_C_BEGIN

void PETSC_STDCALL stshellsetapply_(ST *st,void (PETSC_STDCALL *apply)(void*,Vec *,Vec *,PetscErrorCode*),
                                    PetscErrorCode *ierr)
{
  f1 = apply;
  *ierr = STShellSetApply(*st,ourshellapply);
}

void PETSC_STDCALL stshellsetapplytranspose_(ST *st,void (PETSC_STDCALL *applytranspose)(void*,Vec *,Vec *,PetscErrorCode*),
                                             PetscErrorCode *ierr)
{
  f2 = applytranspose;
  *ierr = STShellSetApplyTranspose(*st,ourshellapplytranspose);
}

void PETSC_STDCALL stshellsetbacktransform_(ST *st,void (PETSC_STDCALL *backtransform)(void*,PetscScalar*,PetscScalar*,PetscErrorCode*),
                                    PetscErrorCode *ierr)
{
  f3 = backtransform;
  *ierr = STShellSetBackTransform(*st,ourshellbacktransform);
}

EXTERN_C_END

