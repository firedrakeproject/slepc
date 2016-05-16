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

#if defined(PETSC_HAVE_FORTRAN_CAPS)
#define slepcinitializefortran_     SLEPCINITIALIZEFORTRAN
#elif !defined(PETSC_HAVE_FORTRAN_UNDERSCORE)
#define slepcinitializefortran_     slepcinitializefortran
#endif

/*@C
   SlepcInitializeFortran - Routine that should be called from C after
   the call to SlepcInitialize() if one is using a C main program
   that calls Fortran routines that in turn call SLEPc routines.

   Collective on PETSC_COMM_WORLD

   Level: beginner

   Notes:
   SlepcInitializeFortran() initializes some of the default SLEPc variables
   for use in Fortran if a user's main program is written in C.
   SlepcInitializeFortran() is NOT needed if a user's main
   program is written in Fortran; in this case, just calling
   SlepcInitialize() in the main (Fortran) program is sufficient.

.seealso:  SlepcInitialize()

@*/

PetscErrorCode SlepcInitializeFortran(void)
{
  PetscInitializeFortran();
  return 0;
}

PETSC_EXTERN void PETSC_STDCALL slepcinitializefortran_(PetscErrorCode *info)
{
  *info = SlepcInitializeFortran();
}

