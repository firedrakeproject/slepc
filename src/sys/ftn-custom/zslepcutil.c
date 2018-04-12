/*
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
   SLEPc - Scalable Library for Eigenvalue Problem Computations
   Copyright (c) 2002-2018, Universitat Politecnica de Valencia, Spain

   This file is part of SLEPc.
   SLEPc is distributed under a 2-clause BSD license (see LICENSE).
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
*/

#include <slepc/private/slepcimpl.h>
#include <petsc/private/fortranimpl.h>

#if defined(PETSC_HAVE_FORTRAN_CAPS)
#define slepcconvmonitorcreate_       SLEPCCONVMONITORCREATE
#define slepcconvmonitordestroy_      SLEPCCONVMONITORDESTROY
#define slepcgetversion_              SLEPCGETVERSION
#define slepcgetversionnumber_        SLEPCGETVERSIONNUMBER
#elif !defined(PETSC_HAVE_FORTRAN_UNDERSCORE)
#define slepcconvmonitorcreate_       slepcconvmonitorcreate
#define slepcconvmonitordestroy_      slepcconvmonitordestroy
#define slepcgetversion_              slepcgetversion
#define slepcgetversionnumber_        slepcgetversionnumber
#endif

PETSC_EXTERN void PETSC_STDCALL slepcconvmonitorcreate_(PetscViewer *vin,PetscViewerFormat *format,SlepcConvMonitor *ctx,PetscErrorCode *ierr)
{
  PetscViewer v;
  PetscPatchDefaultViewers_Fortran(vin,v);
  *ierr = SlepcConvMonitorCreate(v,*format,ctx);
}

PETSC_EXTERN void slepcconvmonitordestroy_(SlepcConvMonitor *ctx,PetscErrorCode *ierr)
{
  *ierr = SlepcConvMonitorDestroy(ctx);
}

PETSC_EXTERN void PETSC_STDCALL slepcgetversion_(char *version PETSC_MIXED_LEN(len1),int *ierr PETSC_END_LEN(len1))
{
  *ierr = SlepcGetVersion(version,len1);
  FIXRETURNCHAR(PETSC_TRUE,version,len1);
}

PETSC_EXTERN void PETSC_STDCALL slepcgetversionnumber_(PetscInt *major,PetscInt *minor,PetscInt *subminor,PetscInt *release,PetscInt *ierr )
{
  CHKFORTRANNULLINTEGER(major);
  CHKFORTRANNULLINTEGER(minor);
  CHKFORTRANNULLINTEGER(subminor);
  CHKFORTRANNULLINTEGER(release);
  *ierr = SlepcGetVersionNumber(major,minor,subminor,release);
}

