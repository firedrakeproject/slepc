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
#include <slepcds.h>

#if defined(PETSC_HAVE_FORTRAN_CAPS)
#define dscreate_                 DSCREATE
#define dsdestroy_                DSDESTROY
#define dssetoptionsprefix_       DSSETOPTIONSPREFIX
#define dsappendoptionsprefix_    DSAPPENDOPTIONSPREFIX
#define dsgetoptionsprefix_       DSGETOPTIONSPREFIX
#define dsview_                   DSVIEW
#elif !defined(PETSC_HAVE_FORTRAN_UNDERSCORE)
#define dscreate_                 dscreate
#define dsdestroy_                dsdestroy
#define dssetoptionsprefix_       dssetoptionsprefix
#define dsappendoptionsprefix_    dsappendoptionsprefix
#define dsgetoptionsprefix_       dsgetoptionsprefix
#define dsview_                   dsview
#endif

PETSC_EXTERN void PETSC_STDCALL dscreate_(MPI_Fint *comm,DS *newds,PetscErrorCode *ierr)
{
  *ierr = DSCreate(MPI_Comm_f2c(*(comm)),newds);
}

PETSC_EXTERN void PETSC_STDCALL dsdestroy_(DS *ds,PetscErrorCode *ierr)
{
  *ierr = DSDestroy(ds);
}

PETSC_EXTERN void PETSC_STDCALL dssetoptionsprefix_(DS *ds,CHAR prefix PETSC_MIXED_LEN(len),PetscErrorCode *ierr PETSC_END_LEN(len))
{
  char *t;

  FIXCHAR(prefix,len,t);
  *ierr = DSSetOptionsPrefix(*ds,t);
  FREECHAR(prefix,t);
}

PETSC_EXTERN void PETSC_STDCALL dsappendoptionsprefix_(DS *ds,CHAR prefix PETSC_MIXED_LEN(len),PetscErrorCode *ierr PETSC_END_LEN(len))
{
  char *t;

  FIXCHAR(prefix,len,t);
  *ierr = DSAppendOptionsPrefix(*ds,t);
  FREECHAR(prefix,t);
}

PETSC_EXTERN void PETSC_STDCALL dsgetoptionsprefix_(DS *ds,CHAR prefix PETSC_MIXED_LEN(len),PetscErrorCode *ierr PETSC_END_LEN(len))
{
  const char *tname;

  *ierr = DSGetOptionsPrefix(*ds,&tname); if (*ierr) return;
  *ierr = PetscStrncpy(prefix,tname,len);
}

PETSC_EXTERN void PETSC_STDCALL dsview_(DS *ds,PetscViewer *viewer,PetscErrorCode *ierr)
{
  PetscViewer v;
  PetscPatchDefaultViewers_Fortran(viewer,v);
  *ierr = DSView(*ds,v);
}

