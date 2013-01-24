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

#include <petsc-private/fortranimpl.h>
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

/* These are not extern C because they are passed into non-extern C user level functions */
static PetscErrorCode ourshellapply(ST st,Vec x,Vec y)
{
  PetscErrorCode ierr = 0;
  (*(void (PETSC_STDCALL*)(ST*,Vec*,Vec*,PetscErrorCode*))(((PetscObject)st)->fortran_func_pointers[0]))(&st,&x,&y,&ierr);CHKERRQ(ierr);
  return 0;
}

static PetscErrorCode ourshellapplytranspose(ST st,Vec x,Vec y)
{
  PetscErrorCode ierr = 0;
  (*(void (PETSC_STDCALL*)(ST*,Vec*,Vec*,PetscErrorCode*))(((PetscObject)st)->fortran_func_pointers[1]))(&st,&x,&y,&ierr);CHKERRQ(ierr);
  return 0;
}

static PetscErrorCode ourshellbacktransform(ST st,PetscInt n,PetscScalar *eigr,PetscScalar *eigi)
{
  PetscErrorCode ierr = 0;
  (*(void (PETSC_STDCALL*)(ST*,PetscInt*,PetscScalar*,PetscScalar*,PetscErrorCode*))(((PetscObject)st)->fortran_func_pointers[2]))(&st,&n,eigr,eigi,&ierr);CHKERRQ(ierr);
  return 0;
}

EXTERN_C_BEGIN

void PETSC_STDCALL stshellgetcontext_(ST *st,void **ctx,PetscErrorCode *ierr)
{
  *ierr = STShellGetContext(*st,ctx);
}

void PETSC_STDCALL stshellsetapply_(ST *st,void (PETSC_STDCALL *apply)(void*,Vec *,Vec *,PetscErrorCode*),
                                    PetscErrorCode *ierr)
{
  PetscObjectAllocateFortranPointers(*st,3);
  ((PetscObject)*st)->fortran_func_pointers[0] = (PetscVoidFunction)apply;
  *ierr = STShellSetApply(*st,ourshellapply);
}

void PETSC_STDCALL stshellsetapplytranspose_(ST *st,void (PETSC_STDCALL *applytranspose)(void*,Vec *,Vec *,PetscErrorCode*),
                                             PetscErrorCode *ierr)
{
  PetscObjectAllocateFortranPointers(*st,3);
  ((PetscObject)*st)->fortran_func_pointers[1] = (PetscVoidFunction)applytranspose;
  *ierr = STShellSetApplyTranspose(*st,ourshellapplytranspose);
}

void PETSC_STDCALL stshellsetbacktransform_(ST *st,void (PETSC_STDCALL *backtransform)(void*,PetscScalar*,PetscScalar*,PetscErrorCode*),PetscErrorCode *ierr)
{
  PetscObjectAllocateFortranPointers(*st,3);
  ((PetscObject)*st)->fortran_func_pointers[2] = (PetscVoidFunction)backtransform;
  *ierr = STShellSetBackTransform(*st,ourshellbacktransform);
}

EXTERN_C_END

