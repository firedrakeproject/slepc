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

#include <petsc/private/fortranimpl.h>
#include <slepcst.h>

#if defined(PETSC_HAVE_FORTRAN_CAPS)
#define stshellgetcontext_        STSHELLGETCONTEXT
#define stshellsetapply_          STSHELLSETAPPLY
#define stshellsetapplytranspose_ STSHELLSETAPPLYTRANSPOSE
#define stshellsetbacktransform_  STSHELLSETBACKTRANSFORM
#elif !defined(PETSC_HAVE_FORTRAN_UNDERSCORE)
#define stshellgetcontext_        stshellgetcontext
#define stshellsetapply_          stshellsetapply
#define stshellsetapplytranspose_ stshellsetapplytranspose
#define stshellsetbacktransform_  stshellsetbacktransform
#endif

static struct {
  PetscFortranCallbackId apply;
  PetscFortranCallbackId applytranspose;
  PetscFortranCallbackId backtransform;
} _cb;

/* These are not extern C because they are passed into non-extern C user level functions */
#undef __FUNCT__
#define __FUNCT__ "ourshellapply"
static PetscErrorCode ourshellapply(ST st,Vec x,Vec y)
{
  PetscObjectUseFortranCallback(st,_cb.apply,(ST*,Vec*,Vec*,PetscErrorCode*),(&st,&x,&y,&ierr));
}

#undef __FUNCT__
#define __FUNCT__ "ourshellapplytranspose"
static PetscErrorCode ourshellapplytranspose(ST st,Vec x,Vec y)
{
  PetscObjectUseFortranCallback(st,_cb.applytranspose,(ST*,Vec*,Vec*,PetscErrorCode*),(&st,&x,&y,&ierr));
}

#undef __FUNCT__
#define __FUNCT__ "ourshellbacktransform"
static PetscErrorCode ourshellbacktransform(ST st,PetscInt n,PetscScalar *eigr,PetscScalar *eigi)
{
  PetscObjectUseFortranCallback(st,_cb.backtransform,(ST*,PetscInt*,PetscScalar*,PetscScalar*,PetscErrorCode*),(&st,&n,eigr,eigi,&ierr));
}

PETSC_EXTERN void PETSC_STDCALL stshellgetcontext_(ST *st,void **ctx,PetscErrorCode *ierr)
{
  *ierr = STShellGetContext(*st,ctx);
}

PETSC_EXTERN void PETSC_STDCALL stshellsetapply_(ST *st,void (PETSC_STDCALL *apply)(void*,Vec*,Vec*,PetscErrorCode*),PetscErrorCode *ierr)
{
  *ierr = PetscObjectSetFortranCallback((PetscObject)*st,PETSC_FORTRAN_CALLBACK_CLASS,&_cb.apply,(PetscVoidFunction)apply,NULL); if (*ierr) return;
  *ierr = STShellSetApply(*st,ourshellapply);
}

PETSC_EXTERN void PETSC_STDCALL stshellsetapplytranspose_(ST *st,void (PETSC_STDCALL *applytranspose)(void*,Vec*,Vec*,PetscErrorCode*),PetscErrorCode *ierr)
{
  *ierr = PetscObjectSetFortranCallback((PetscObject)*st,PETSC_FORTRAN_CALLBACK_CLASS,&_cb.applytranspose,(PetscVoidFunction)applytranspose,NULL); if (*ierr) return;
  *ierr = STShellSetApplyTranspose(*st,ourshellapplytranspose);
}

PETSC_EXTERN void PETSC_STDCALL stshellsetbacktransform_(ST *st,void (PETSC_STDCALL *backtransform)(void*,PetscScalar*,PetscScalar*,PetscErrorCode*),PetscErrorCode *ierr)
{
  *ierr = PetscObjectSetFortranCallback((PetscObject)*st,PETSC_FORTRAN_CALLBACK_CLASS,&_cb.backtransform,(PetscVoidFunction)backtransform,NULL); if (*ierr) return;
  *ierr = STShellSetBackTransform(*st,ourshellbacktransform);
}

