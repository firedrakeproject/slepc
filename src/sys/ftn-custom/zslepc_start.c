/*
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
   SLEPc - Scalable Library for Eigenvalue Problem Computations
   Copyright (c) 2002-2019, Universitat Politecnica de Valencia, Spain

   This file is part of SLEPc.
   SLEPc is distributed under a 2-clause BSD license (see LICENSE).
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
*/
/*
   This file contains the Fortran version of SlepcInitialize()
*/

#include <slepc/private/slepcimpl.h>
#include <petsc/private/fortranimpl.h>

#if defined(PETSC_HAVE_FORTRAN_CAPS)
#define petscinitialize_              PETSCINITIALIZE
#define petscinitializenoarguments_   PETSCINITIALIZENOARGUMENTS
#define petscfinalize_                PETSCFINALIZE
#define slepcinitialize_              SLEPCINITIALIZE
#define slepcinitializenoarguments_   SLEPCINITIALIZENOARGUMENTS
#define slepcfinalize_                SLEPCFINALIZE
#elif !defined(PETSC_HAVE_FORTRAN_UNDERSCORE)
#define petscinitialize_              petscinitialize
#define petscinitializenoarguments_   petscinitializenoarguments
#define petscfinalize_                petscfinalize
#define slepcinitialize_              slepcinitialize
#define slepcinitializenoarguments_   slepcinitializenoarguments
#define slepcfinalize_                slepcfinalize
#endif

SLEPC_EXTERN void PETSC_STDCALL petscinitialize_(char *filename PETSC_MIXED_LEN(len),PetscErrorCode *ierr PETSC_END_LEN(len));
SLEPC_EXTERN void PETSC_STDCALL petscinitializenoarguments_(PetscErrorCode *ierr);
SLEPC_EXTERN void PETSC_STDCALL petscfinalize_(PetscErrorCode *ierr);

/*
    SlepcInitialize - Version called from Fortran.

    Notes:
    Since this routine is called from Fortran it does not return error codes.
*/
static void slepcinitialize_internal(char *filename,PetscInt len,PetscBool arguments,PetscErrorCode *ierr)
{
  PetscBool flg;
  *ierr = 1;
  if (SlepcInitializeCalled) { *ierr = 0; return; }

  *ierr = PetscInitialized(&flg);
  if (*ierr) { (*PetscErrorPrintf)("SlepcInitialize:PetscInitialized failed");return; }
  if (!flg) {
    if (arguments) {
#if defined(PETSC_HAVE_FORTRAN_MIXED_STR_ARG)
      petscinitialize_(filename,len,ierr);
#else
      petscinitialize_(filename,ierr,len);
#endif
    } else {
      petscinitializenoarguments_(ierr);
    }
    if (*ierr) { (*PetscErrorPrintf)("SlepcInitialize:PetscInitialize failed");return; }
    SlepcBeganPetsc = PETSC_TRUE;
  }

  *ierr = SlepcCitationsInitialize();
  if (*ierr) { (*PetscErrorPrintf)("SlepcInitialize:SlepcCitationsInitialize()\n");return; }
#if defined(PETSC_HAVE_DYNAMIC_LIBRARIES)
  *ierr = SlepcInitialize_DynamicLibraries();
  if (*ierr) { (*PetscErrorPrintf)("SlepcInitialize:Initializing dynamic libraries\n");return; }
#endif

#if defined(PETSC_HAVE_DRAND48)
  /* work-around for Cygwin drand48() initialization bug */
  srand48(0);
#endif

  SlepcInitializeCalled = PETSC_TRUE;
  *ierr = PetscInfo(0,"SLEPc successfully started from Fortran\n");
  if (*ierr) { (*PetscErrorPrintf)("SlepcInitialize:Calling PetscInfo()");return; }
}

SLEPC_EXTERN void PETSC_STDCALL slepcinitialize_(char *filename PETSC_MIXED_LEN(len),PetscErrorCode *ierr PETSC_END_LEN(len))
{
  slepcinitialize_internal(filename,len,PETSC_TRUE,ierr);
}

SLEPC_EXTERN void PETSC_STDCALL slepcinitializenoarguments_(PetscErrorCode *ierr)
{
  slepcinitialize_internal(NULL,(PetscInt)0,PETSC_FALSE,ierr);
}

SLEPC_EXTERN void PETSC_STDCALL slepcfinalize_(PetscErrorCode *ierr)
{
  *ierr = PetscInfo(0,"SlepcFinalize called from Fortran\n");
  if (*ierr) { (*PetscErrorPrintf)("SlepcFinalize:Calling PetscInfo()");return; }
  *ierr = 0;
  if (SlepcBeganPetsc) {
    petscfinalize_(ierr);
    if (*ierr) { (*PetscErrorPrintf)("SlepcFinalize:Calling petscfinalize_()");return; }
  }
  SlepcInitializeCalled = PETSC_FALSE;
}

