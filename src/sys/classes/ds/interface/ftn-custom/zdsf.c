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
#include <slepcds.h>

#if defined(PETSC_HAVE_FORTRAN_CAPS)
#define dssettype_                DSSETTYPE
#define dsgettype_                DSGETTYPE
#define dssetoptionsprefix_       DSSETOPTIONSPREFIX
#define dsappendoptionsprefix_    DSAPPENDOPTIONSPREFIX
#define dsgetoptionsprefix_       DSGETOPTIONSPREFIX
#define dsview_                   DSVIEW
#define dsviewmat_                DSVIEWMAT
#define dsvectors_                DSVECTORS
#define dssort_                   DSSORT
#elif !defined(PETSC_HAVE_FORTRAN_UNDERSCORE)
#define dssettype_                dssettype
#define dsgettype_                dsgettype
#define dssetoptionsprefix_       dssetoptionsprefix
#define dsappendoptionsprefix_    dsappendoptionsprefix
#define dsgetoptionsprefix_       dsgetoptionsprefix
#define dsview_                   dsview
#define dsviewmat_                dsviewmat
#define dsvectors_                dsvectors
#define dssort_                   dssort
#endif

PETSC_EXTERN void PETSC_STDCALL dssettype_(DS *ds,CHAR type PETSC_MIXED_LEN(len),PetscErrorCode *ierr PETSC_END_LEN(len))
{
  char *t;

  FIXCHAR(type,len,t);
  *ierr = DSSetType(*ds,t);
  FREECHAR(type,t);
}

PETSC_EXTERN void PETSC_STDCALL dsgettype_(DS *ds,CHAR name PETSC_MIXED_LEN(len),PetscErrorCode *ierr PETSC_END_LEN(len))
{
  DSType tname;

  *ierr = DSGetType(*ds,&tname);if (*ierr) return;
  *ierr = PetscStrncpy(name,tname,len);
  FIXRETURNCHAR(PETSC_TRUE,name,len);
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
  FIXRETURNCHAR(PETSC_TRUE,prefix,len);
}

PETSC_EXTERN void PETSC_STDCALL dsview_(DS *ds,PetscViewer *viewer,PetscErrorCode *ierr)
{
  PetscViewer v;
  PetscPatchDefaultViewers_Fortran(viewer,v);
  *ierr = DSView(*ds,v);
}

PETSC_EXTERN void PETSC_STDCALL dsviewmat_(DS *ds,PetscViewer *viewer,DSMatType *m,PetscErrorCode *ierr)
{
  PetscViewer v;
  PetscPatchDefaultViewers_Fortran(viewer,v);
  *ierr = DSViewMat(*ds,v,*m);
}

PETSC_EXTERN void PETSC_STDCALL dsvectors_(DS *ds,DSMatType *mat,PetscInt *j,PetscReal *rnorm,PetscErrorCode *ierr)
{
  CHKFORTRANNULLINTEGER(j);
  CHKFORTRANNULLREAL(rnorm);
  *ierr = DSVectors(*ds,*mat,j,rnorm);
}

PETSC_EXTERN void PETSC_STDCALL dssort_(DS *ds,PetscScalar *eigr,PetscScalar *eigi,PetscScalar *rr,PetscScalar *ri,PetscInt *k,PetscErrorCode *ierr)
{
  CHKFORTRANNULLSCALAR(eigr);
  CHKFORTRANNULLSCALAR(eigi);
  CHKFORTRANNULLSCALAR(rr);
  CHKFORTRANNULLSCALAR(ri);
  CHKFORTRANNULLINTEGER(k);
  *ierr = DSSort(*ds,eigr,eigi,rr,ri,k);
}

