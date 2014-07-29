/*
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
   SLEPc - Scalable Library for Eigenvalue Problem Computations
   Copyright (c) 2002-2014, Universitat Politecnica de Valencia, Spain

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
#define stsettype_                STSETTYPE
#define stgettype_                STGETTYPE
#define stcreate_                 STCREATE
#define stdestroy_                STDESTROY
#define stsetoptionsprefix_       STSETOPTIONSPREFIX
#define stappendoptionsprefix_    STAPPENDOPTIONSPREFIX
#define stgetoptionsprefix_       STGETOPTIONSPREFIX
#define stview_                   STVIEW
#define stgetmatmode_             STGETMATMODE
#elif !defined(PETSC_HAVE_FORTRAN_UNDERSCORE)
#define stsettype_                stsettype
#define stgettype_                stgettype
#define stcreate_                 stcreate
#define stdestroy_                stdestroy
#define stsetoptionsprefix_       stsetoptionsprefix
#define stappendoptionsprefix_    stappendoptionsprefix
#define stgetoptionsprefix_       stgetoptionsprefix
#define stview_                   stview
#define stgetmatmode_             stgetmatmode
#endif

PETSC_EXTERN void PETSC_STDCALL stsettype_(ST *st,CHAR type PETSC_MIXED_LEN(len),PetscErrorCode *ierr PETSC_END_LEN(len))
{
  char *t;

  FIXCHAR(type,len,t);
  *ierr = STSetType(*st,t);
  FREECHAR(type,t);
}

PETSC_EXTERN void PETSC_STDCALL stgettype_(ST *st,CHAR name PETSC_MIXED_LEN(len),PetscErrorCode *ierr PETSC_END_LEN(len))
{
  STType tname;

  *ierr = STGetType(*st,&tname); if (*ierr) return;
  *ierr = PetscStrncpy(name,tname,len);
  FIXRETURNCHAR(PETSC_TRUE,name,len);
}

PETSC_EXTERN void PETSC_STDCALL stcreate_(MPI_Fint *comm,ST *newst,PetscErrorCode *ierr)
{
  *ierr = STCreate(MPI_Comm_f2c(*(comm)),newst);
}

PETSC_EXTERN void PETSC_STDCALL stdestroy_(ST *st,PetscErrorCode *ierr)
{
  *ierr = STDestroy(st);
}

PETSC_EXTERN void PETSC_STDCALL stsetoptionsprefix_(ST *st,CHAR prefix PETSC_MIXED_LEN(len),PetscErrorCode *ierr PETSC_END_LEN(len))
{
  char *t;

  FIXCHAR(prefix,len,t);
  *ierr = STSetOptionsPrefix(*st,t);
  FREECHAR(prefix,t);
}

PETSC_EXTERN void PETSC_STDCALL stappendoptionsprefix_(ST *st,CHAR prefix PETSC_MIXED_LEN(len),PetscErrorCode *ierr PETSC_END_LEN(len))
{
  char *t;

  FIXCHAR(prefix,len,t);
  *ierr = STAppendOptionsPrefix(*st,t);
  FREECHAR(prefix,t);
}

PETSC_EXTERN void PETSC_STDCALL stgetoptionsprefix_(ST *st,CHAR prefix PETSC_MIXED_LEN(len),PetscErrorCode *ierr PETSC_END_LEN(len))
{
  const char *tname;

  *ierr = STGetOptionsPrefix(*st,&tname); if (*ierr) return;
  *ierr = PetscStrncpy(prefix,tname,len);
}

PETSC_EXTERN void PETSC_STDCALL stview_(ST *st,PetscViewer *viewer,PetscErrorCode *ierr)
{
  PetscViewer v;
  PetscPatchDefaultViewers_Fortran(viewer,v);
  *ierr = STView(*st,v);
}

PETSC_EXTERN void PETSC_STDCALL stgetmatmode_(ST *st,STMatMode *mode,PetscErrorCode *ierr)
{
  *ierr = STGetMatMode(*st,mode);
}

