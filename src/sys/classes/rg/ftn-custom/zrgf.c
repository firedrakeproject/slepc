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
#include <slepcrg.h>

#if defined(PETSC_HAVE_FORTRAN_CAPS)
#define rgcreate_                 RGCREATE
#define rgdestroy_                RGDESTROY
#define rgsetoptionsprefix_       RGSETOPTIONSPREFIX
#define rgappendoptionsprefix_    RGAPPENDOPTIONSPREFIX
#define rggetoptionsprefix_       RGGETOPTIONSPREFIX
#define rgview_                   RGVIEW
#elif !defined(PETSC_HAVE_FORTRAN_UNDERSCORE)
#define rgcreate_                 rgcreate
#define rgdestroy_                rgdestroy
#define rgsetoptionsprefix_       rgsetoptionsprefix
#define rgappendoptionsprefix_    rgappendoptionsprefix
#define rggetoptionsprefix_       rggetoptionsprefix
#define rgview_                   rgview
#endif

PETSC_EXTERN void PETSC_STDCALL rgcreate_(MPI_Fint *comm,RG *newrg,PetscErrorCode *ierr)
{
  *ierr = RGCreate(MPI_Comm_f2c(*(comm)),newrg);
}

PETSC_EXTERN void PETSC_STDCALL rgdestroy_(RG *rg,PetscErrorCode *ierr)
{
  *ierr = RGDestroy(rg);
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
}

PETSC_EXTERN void PETSC_STDCALL rgview_(RG *rg,PetscViewer *viewer,PetscErrorCode *ierr)
{
  PetscViewer v;
  PetscPatchDefaultViewers_Fortran(viewer,v);
  *ierr = RGView(*rg,v);
}

