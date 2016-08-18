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
#include <slepcrg.h>

#if defined(PETSC_HAVE_FORTRAN_CAPS)
#define rgsettype_                RGSETTYPE
#define rggettype_                RGGETTYPE
#define rgsetoptionsprefix_       RGSETOPTIONSPREFIX
#define rgappendoptionsprefix_    RGAPPENDOPTIONSPREFIX
#define rggetoptionsprefix_       RGGETOPTIONSPREFIX
#define rgview_                   RGVIEW
#elif !defined(PETSC_HAVE_FORTRAN_UNDERSCORE)
#define rgsettype_                rgsettype
#define rggettype_                rggettype
#define rgsetoptionsprefix_       rgsetoptionsprefix
#define rgappendoptionsprefix_    rgappendoptionsprefix
#define rggetoptionsprefix_       rggetoptionsprefix
#define rgview_                   rgview
#endif

PETSC_EXTERN void PETSC_STDCALL rgsettype_(RG *rg,CHAR type PETSC_MIXED_LEN(len),PetscErrorCode *ierr PETSC_END_LEN(len))
{
  char *t;

  FIXCHAR(type,len,t);
  *ierr = RGSetType(*rg,t);
  FREECHAR(type,t);
}

PETSC_EXTERN void PETSC_STDCALL rggettype_(RG *rg,CHAR name PETSC_MIXED_LEN(len),PetscErrorCode *ierr PETSC_END_LEN(len))
{
  RGType tname;

  *ierr = RGGetType(*rg,&tname); if (*ierr) return;
  *ierr = PetscStrncpy(name,tname,len);
  FIXRETURNCHAR(PETSC_TRUE,name,len);
}

PETSC_EXTERN void PETSC_STDCALL rgsetoptionsprefix_(RG *rg,CHAR prefix PETSC_MIXED_LEN(len),PetscErrorCode *ierr PETSC_END_LEN(len))
{
  char *t;

  FIXCHAR(prefix,len,t);
  *ierr = RGSetOptionsPrefix(*rg,t);
  FREECHAR(prefix,t);
}

PETSC_EXTERN void PETSC_STDCALL rgappendoptionsprefix_(RG *rg,CHAR prefix PETSC_MIXED_LEN(len),PetscErrorCode *ierr PETSC_END_LEN(len))
{
  char *t;

  FIXCHAR(prefix,len,t);
  *ierr = RGAppendOptionsPrefix(*rg,t);
  FREECHAR(prefix,t);
}

PETSC_EXTERN void PETSC_STDCALL rggetoptionsprefix_(RG *rg,CHAR prefix PETSC_MIXED_LEN(len),PetscErrorCode *ierr PETSC_END_LEN(len))
{
  const char *tname;

  *ierr = RGGetOptionsPrefix(*rg,&tname); if (*ierr) return;
  *ierr = PetscStrncpy(prefix,tname,len);
  FIXRETURNCHAR(PETSC_TRUE,prefix,len);
}

PETSC_EXTERN void PETSC_STDCALL rgview_(RG *rg,PetscViewer *viewer,PetscErrorCode *ierr)
{
  PetscViewer v;
  PetscPatchDefaultViewers_Fortran(viewer,v);
  *ierr = RGView(*rg,v);
}

