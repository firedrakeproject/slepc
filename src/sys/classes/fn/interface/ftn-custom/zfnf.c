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
#include <slepc/private/slepcimpl.h>
#include <slepc/private/fnimpl.h>

#if defined(PETSC_HAVE_FORTRAN_CAPS)
#define fnview_                    FNVIEW
#define fnsetoptionsprefix_        FNSETOPTIONSPREFIX
#define fnappendoptionsprefix_     FNAPPENDOPTIONSPREFIX
#define fngetoptionsprefix_        FNGETOPTIONSPREFIX
#define fnsettype_                 FNSETTYPE
#define fngettype_                 FNGETTYPE
#elif !defined(PETSC_HAVE_FORTRAN_UNDERSCORE)
#define fnview_                    fnview
#define fnsetoptionsprefix_        fnsetoptionsprefix
#define fnappendoptionsprefix_     fnappendoptionsprefix
#define fngetoptionsprefix_        fngetoptionsprefix
#define fnsettype_                 fnsettype
#define fngettype_                 fngettype
#endif

PETSC_EXTERN void PETSC_STDCALL fnview_(FN *fn,PetscViewer *viewer,PetscErrorCode *ierr)
{
  PetscViewer v;
  PetscPatchDefaultViewers_Fortran(viewer,v);
  *ierr = FNView(*fn,v);
}

PETSC_EXTERN void PETSC_STDCALL fnsetoptionsprefix_(FN *fn,CHAR prefix PETSC_MIXED_LEN(len),PetscErrorCode *ierr PETSC_END_LEN(len))
{
  char *t;

  FIXCHAR(prefix,len,t);
  *ierr = FNSetOptionsPrefix(*fn,t);
  FREECHAR(prefix,t);
}

PETSC_EXTERN void PETSC_STDCALL fnappendoptionsprefix_(FN *fn,CHAR prefix PETSC_MIXED_LEN(len),PetscErrorCode *ierr PETSC_END_LEN(len))
{
  char *t;

  FIXCHAR(prefix,len,t);
  *ierr = FNAppendOptionsPrefix(*fn,t);
  FREECHAR(prefix,t);
}

PETSC_EXTERN void PETSC_STDCALL fngetoptionsprefix_(FN *fn,CHAR prefix PETSC_MIXED_LEN(len),PetscErrorCode *ierr PETSC_END_LEN(len))
{
  const char *tname;

  *ierr = FNGetOptionsPrefix(*fn,&tname); if (*ierr) return;
  *ierr = PetscStrncpy(prefix,tname,len);
  FIXRETURNCHAR(PETSC_TRUE,prefix,len);
}

PETSC_EXTERN void PETSC_STDCALL fnsettype_(FN *fn,CHAR type PETSC_MIXED_LEN(len),PetscErrorCode *ierr PETSC_END_LEN(len))
{
  char *t;

  FIXCHAR(type,len,t);
  *ierr = FNSetType(*fn,t);
  FREECHAR(type,t);
}

PETSC_EXTERN void PETSC_STDCALL fngettype_(FN *fn,CHAR name PETSC_MIXED_LEN(len),PetscErrorCode *ierr PETSC_END_LEN(len))
{
  FNType tname;

  *ierr = FNGetType(*fn,&tname);if (*ierr) return;
  *ierr = PetscStrncpy(name,tname,len);
  FIXRETURNCHAR(PETSC_TRUE,name,len);
}


