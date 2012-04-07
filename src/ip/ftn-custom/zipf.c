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
#include <slepcip.h>

#if defined(PETSC_HAVE_FORTRAN_CAPS)
#define ipcreate_                 IPCREATE
#define ipdestroy_                IPDESTROY
#define ipsetoptionsprefix_       IPSETOPTIONSPREFIX
#define ipappendoptionsprefix_    IPAPPENDOPTIONSPREFIX
#define ipgetoptionsprefix_       IPSETOPTIONSPREFIX
#define ipgetorthogonalization_   IPGETORTHOGONALIZATION
#define ipview_                   IPVIEW
#define ipgetmatrix_              IPGETMATRIX
#elif !defined(PETSC_HAVE_FORTRAN_UNDERSCORE)
#define ipcreate_                 ipcreate
#define ipdestroy_                ipdestroy
#define ipsetoptionsprefix_       ipsetoptionsprefix
#define ipappendoptionsprefix_    ipappendoptionsprefix
#define ipgetoptionsprefix_       ipgetoptionsprefix
#define ipgetorthogonalization_   ipgetorthogonalization
#define ipview_                   ipview
#define ipgetmatrix_              ipgetmatrix
#endif

EXTERN_C_BEGIN

void PETSC_STDCALL ipcreate_(MPI_Fint *comm,IP *newip,PetscErrorCode *ierr)
{
  *ierr = IPCreate(MPI_Comm_f2c(*(comm)),newip);
}

void PETSC_STDCALL ipdestroy_(IP *ip, PetscErrorCode *ierr)
{
  *ierr = IPDestroy(ip);
}

void PETSC_STDCALL ipsetoptionsprefix_(IP *ip,CHAR prefix PETSC_MIXED_LEN(len),
                                       PetscErrorCode *ierr PETSC_END_LEN(len))
{
  char *t;

  FIXCHAR(prefix,len,t);
  *ierr = IPSetOptionsPrefix(*ip,t);
  FREECHAR(prefix,t);
}

void PETSC_STDCALL ipappendoptionsprefix_(IP *ip,CHAR prefix PETSC_MIXED_LEN(len),
                                          PetscErrorCode *ierr PETSC_END_LEN(len))
{
  char *t;

  FIXCHAR(prefix,len,t);
  *ierr = IPAppendOptionsPrefix(*ip,t);
  FREECHAR(prefix,t);
}

void PETSC_STDCALL ipgetoptionsprefix_(IP *ip,CHAR prefix PETSC_MIXED_LEN(len),PetscErrorCode *ierr PETSC_END_LEN(len))
{
  const char *tname;

  *ierr = IPGetOptionsPrefix(*ip,&tname); if (*ierr) return;
  *ierr = PetscStrncpy(prefix,tname,len);
}

void PETSC_STDCALL ipgetorthogonalization_(IP *ip,IPOrthogType *type,IPOrthogRefineType *refinement,PetscReal *eta,PetscErrorCode *ierr)
{
  *ierr = IPGetOrthogonalization(*ip,type,refinement,eta);
}

void PETSC_STDCALL ipview_(IP *ip,PetscViewer *viewer, PetscErrorCode *ierr)
{
  PetscViewer v;
  PetscPatchDefaultViewers_Fortran(viewer,v);
  *ierr = IPView(*ip,v);
}

void PETSC_STDCALL ipgetmatrix_(IP *ip,Mat *mat,PetscErrorCode *ierr)
{
  *ierr = IPGetMatrix(*ip,mat);
}

EXTERN_C_END
