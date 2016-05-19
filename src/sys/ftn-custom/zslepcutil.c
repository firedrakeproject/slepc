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

#include <slepc/private/slepcimpl.h>
#include <petsc/private/fortranimpl.h>

#if defined(PETSC_HAVE_FORTRAN_CAPS)
#define slepcconvmonitorcreate_       SLEPCCONVMONITORCREATE
#define slepcconvmonitordestroy_      SLEPCCONVMONITORDESTROY
#elif !defined(PETSC_HAVE_FORTRAN_UNDERSCORE)
#define slepcconvmonitorcreate_       slepcconvmonitorcreate
#define slepcconvmonitordestroy_      slepcconvmonitordestroy
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

