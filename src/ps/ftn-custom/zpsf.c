/*
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
   SLEPc - Scalable Library for Eigenvalue Problem Computations
   Copyright (c) 2002-2011, Universitat Politecnica de Valencia, Spain

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
#include <slepcps.h>

#if defined(PETSC_HAVE_FORTRAN_CAPS)
#define pscreate_                 PSCREATE
#define psdestroy_                PSDESTROY
#define pssetoptionsprefix_       PSSETOPTIONSPREFIX
#define psappendoptionsprefix_    PSAPPENDOPTIONSPREFIX
#define psgetoptionsprefix_       PSSETOPTIONSPREFIX
#define psview_                   PSVIEW
#elif !defined(PETSC_HAVE_FORTRAN_UNDERSCORE)
#define pscreate_                 pscreate
#define psdestroy_                psdestroy
#define pssetoptionsprefix_       pssetoptionsprefix
#define psappendoptionsprefix_    psappendoptionsprefix
#define psgetoptionsprefix_       psgetoptionsprefix
#define psview_                   psview
#endif

EXTERN_C_BEGIN

void PETSC_STDCALL pscreate_(MPI_Fint *comm,PS *newps,PetscErrorCode *ierr)
{
  *ierr = PSCreate(MPI_Comm_f2c(*(comm)),newps);
}

void PETSC_STDCALL psdestroy_(PS *ps, PetscErrorCode *ierr)
{
  *ierr = PSDestroy(ps);
}

void PETSC_STDCALL pssetoptionsprefix_(PS *ps,CHAR prefix PETSC_MIXED_LEN(len),
                                       PetscErrorCode *ierr PETSC_END_LEN(len))
{
  char *t;

  FIXCHAR(prefix,len,t);
  *ierr = PSSetOptionsPrefix(*ps,t);
  FREECHAR(prefix,t);
}

void PETSC_STDCALL psappendoptionsprefix_(PS *ps,CHAR prefix PETSC_MIXED_LEN(len),
                                          PetscErrorCode *ierr PETSC_END_LEN(len))
{
  char *t;

  FIXCHAR(prefix,len,t);
  *ierr = PSAppendOptionsPrefix(*ps,t);
  FREECHAR(prefix,t);
}

void PETSC_STDCALL psgetoptionsprefix_(PS *ps,CHAR prefix PETSC_MIXED_LEN(len),PetscErrorCode *ierr PETSC_END_LEN(len))
{
  const char *tname;

  *ierr = PSGetOptionsPrefix(*ps,&tname); if (*ierr) return;
  *ierr = PetscStrncpy(prefix,tname,len);
}

void PETSC_STDCALL psview_(PS *ps,PetscViewer *viewer, PetscErrorCode *ierr)
{
  PetscViewer v;
  PetscPatchDefaultViewers_Fortran(viewer,v);
  *ierr = PSView(*ps,v);
}

EXTERN_C_END
