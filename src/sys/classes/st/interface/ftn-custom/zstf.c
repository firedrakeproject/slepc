/*
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
   SLEPc - Scalable Library for Eigenvalue Problem Computations
   Copyright (c) 2002-2019, Universitat Politecnica de Valencia, Spain

   This file is part of SLEPc.
   SLEPc is distributed under a 2-clause BSD license (see LICENSE).
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
*/

#include <petsc/private/fortranimpl.h>
#include <slepcst.h>

#if defined(PETSC_HAVE_FORTRAN_CAPS)
#define stsettype_                STSETTYPE
#define stgettype_                STGETTYPE
#define stsetoptionsprefix_       STSETOPTIONSPREFIX
#define stappendoptionsprefix_    STAPPENDOPTIONSPREFIX
#define stgetoptionsprefix_       STGETOPTIONSPREFIX
#define stview_                   STVIEW
#elif !defined(PETSC_HAVE_FORTRAN_UNDERSCORE)
#define stsettype_                stsettype
#define stgettype_                stgettype
#define stsetoptionsprefix_       stsetoptionsprefix
#define stappendoptionsprefix_    stappendoptionsprefix
#define stgetoptionsprefix_       stgetoptionsprefix
#define stview_                   stview
#endif

SLEPC_EXTERN void PETSC_STDCALL stsettype_(ST *st,char *type PETSC_MIXED_LEN(len),PetscErrorCode *ierr PETSC_END_LEN(len))
{
  char *t;

  FIXCHAR(type,len,t);
  *ierr = STSetType(*st,t);if (*ierr) return;
  FREECHAR(type,t);
}

SLEPC_EXTERN void PETSC_STDCALL stgettype_(ST *st,char *name PETSC_MIXED_LEN(len),PetscErrorCode *ierr PETSC_END_LEN(len))
{
  STType tname;

  *ierr = STGetType(*st,&tname); if (*ierr) return;
  *ierr = PetscStrncpy(name,tname,len);if (*ierr) return;
  FIXRETURNCHAR(PETSC_TRUE,name,len);
}

SLEPC_EXTERN void PETSC_STDCALL stsetoptionsprefix_(ST *st,char *prefix PETSC_MIXED_LEN(len),PetscErrorCode *ierr PETSC_END_LEN(len))
{
  char *t;

  FIXCHAR(prefix,len,t);
  *ierr = STSetOptionsPrefix(*st,t);if (*ierr) return;
  FREECHAR(prefix,t);
}

SLEPC_EXTERN void PETSC_STDCALL stappendoptionsprefix_(ST *st,char *prefix PETSC_MIXED_LEN(len),PetscErrorCode *ierr PETSC_END_LEN(len))
{
  char *t;

  FIXCHAR(prefix,len,t);
  *ierr = STAppendOptionsPrefix(*st,t);if (*ierr) return;
  FREECHAR(prefix,t);
}

SLEPC_EXTERN void PETSC_STDCALL stgetoptionsprefix_(ST *st,char *prefix PETSC_MIXED_LEN(len),PetscErrorCode *ierr PETSC_END_LEN(len))
{
  const char *tname;

  *ierr = STGetOptionsPrefix(*st,&tname); if (*ierr) return;
  *ierr = PetscStrncpy(prefix,tname,len);if (*ierr) return;
  FIXRETURNCHAR(PETSC_TRUE,prefix,len);
}

SLEPC_EXTERN void PETSC_STDCALL stview_(ST *st,PetscViewer *viewer,PetscErrorCode *ierr)
{
  PetscViewer v;
  PetscPatchDefaultViewers_Fortran(viewer,v);
  *ierr = STView(*st,v);
}

