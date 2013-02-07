/*
  This file contains the Fortran version of SlepcInitialize().

   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
   SLEPc - Scalable Library for Eigenvalue Problem Computations
   Copyright (c) 2002-2012, Universitat Politecnica de Valencia, Spain

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

extern PetscBool SlepcBeganPetsc;
extern PetscBool SlepcInitializeCalled;
extern PetscErrorCode SlepcInitialize_DynamicLibraries(void);
extern PetscErrorCode SlepcInitialize_Packages(void);
extern PetscErrorCode SlepcInitialize_LogEvents(void);

#if defined(PETSC_HAVE_FORTRAN_CAPS)
#define petscinitialize_              PETSCINITIALIZE
#define slepcinitialize_              SLEPCINITIALIZE
#elif !defined(PETSC_HAVE_FORTRAN_UNDERSCORE)
#define petscinitialize_              petscinitialize
#define slepcinitialize_              slepcinitialize
#endif

EXTERN_C_BEGIN
extern void PETSC_STDCALL petscinitialize_(CHAR filename PETSC_MIXED_LEN(len),PetscErrorCode *ierr PETSC_END_LEN(len));
EXTERN_C_END

EXTERN_C_BEGIN
/*
    SlepcInitialize - Version called from Fortran.

    Notes:
    Since this routine is called from Fortran it does not return error codes.
*/
void PETSC_STDCALL slepcinitialize_(CHAR filename PETSC_MIXED_LEN(len),PetscErrorCode *ierr PETSC_END_LEN(len))
{
  PetscBool flg;
  *ierr = 1;
  if (SlepcInitializeCalled) { *ierr = 0; return; }

  *ierr = PetscInitialized(&flg);
  if (*ierr) { (*PetscErrorPrintf)("SlepcInitialize:PetscInitialized failed");return; }
  if (!flg) {
#if defined(PETSC_HAVE_FORTRAN_MIXED_STR_ARG)
    petscinitialize_(filename,len,ierr);
#else
    petscinitialize_(filename,ierr,len);
#endif
    if (*ierr) { (*PetscErrorPrintf)("SlepcInitialize:PetscInitialize failed");return; }
    SlepcBeganPetsc = PETSC_TRUE;
  }

#if defined(PETSC_USE_DYNAMIC_LIBRARIES)
  *ierr = SlepcInitialize_DynamicLibraries(); 
  if (*ierr) { (*PetscErrorPrintf)("SlepcInitialize:Initializing dynamic libraries\n");return; }
#else
  *ierr = SlepcInitialize_Packages(); 
  if (*ierr) { (*PetscErrorPrintf)("SlepcInitialize:Initializing packages\n");return; }
#endif

  *ierr = SlepcInitialize_LogEvents(); 
  if (*ierr) { (*PetscErrorPrintf)("SlepcInitialize:Initializing log events\n");return; }

#if defined(PETSC_HAVE_DRAND48)
  /* work-around for Cygwin drand48() initialization bug */
  srand48(0);
#endif

  SlepcInitializeCalled = PETSC_TRUE;
  *ierr = PetscInfo(0,"SLEPc successfully started from Fortran\n");
  if (*ierr) { (*PetscErrorPrintf)("SlepcInitialize:Calling PetscInfo()");return; }

}  

EXTERN_C_END
