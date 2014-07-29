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
#include <slepceps.h>

#if defined(PETSC_HAVE_FORTRAN_CAPS)
#define epsprimmegetmethod_  EPSPRIMMEGETMETHOD
#elif !defined(PETSC_HAVE_FORTRAN_UNDERSCORE)
#define epsprimmegetmethod_  epsprimmegetmethod
#endif

PETSC_EXTERN void PETSC_STDCALL epsprimmegetmethod_(EPS *eps,EPSPRIMMEMethod *method,PetscErrorCode *ierr)
{
  *ierr = EPSPRIMMEGetMethod(*eps,method);
}

