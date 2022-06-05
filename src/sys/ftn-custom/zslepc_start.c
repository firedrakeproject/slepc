/*
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
   SLEPc - Scalable Library for Eigenvalue Problem Computations
   Copyright (c) 2002-, Universitat Politecnica de Valencia, Spain

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
#define petscinitializef_             PETSCINITIALIZEF
#define petscfinalize_                PETSCFINALIZE
#define slepcinitializef_             SLEPCINITIALIZEF
#define slepcfinalize_                SLEPCFINALIZE
#elif !defined(PETSC_HAVE_FORTRAN_UNDERSCORE)
#define petscinitializef_             petscinitializef
#define petscfinalize_                petscfinalize
#define slepcinitializef_             slepcinitializef
#define slepcfinalize_                slepcfinalize
#endif

SLEPC_EXTERN void petscinitializef_(char *filename,char* help,PetscBool *readarguments,PetscErrorCode *ierr,PETSC_FORTRAN_CHARLEN_T len,PETSC_FORTRAN_CHARLEN_T helplen);
SLEPC_EXTERN void petscfinalize_(PetscErrorCode *ierr);

/*
    SlepcInitialize - Version called from Fortran.

    Notes:
    Since this routine is called from Fortran it does not return error codes.
*/
SLEPC_EXTERN void slepcinitializef_(char *filename,char* help,PetscBool *readarguments,PetscErrorCode *ierr,PETSC_FORTRAN_CHARLEN_T len,PETSC_FORTRAN_CHARLEN_T helplen)
{
  PetscBool flg;

  *ierr = 1;
  if (SlepcInitializeCalled) { *ierr = 0; return; }

  *ierr = PetscInitialized(&flg);
  if (*ierr) { (*PetscErrorPrintf)("SlepcInitialize:PetscInitialized failed");return; }
  if (!flg) {
    petscinitializef_(filename,help,readarguments,ierr,len,helplen);
    if (*ierr) { (*PetscErrorPrintf)("SlepcInitialize:PetscInitialize failed");return; }
    SlepcBeganPetsc = PETSC_TRUE;
  }

  *ierr = SlepcCitationsInitialize();
  if (*ierr) { (*PetscErrorPrintf)("SlepcInitialize:SlepcCitationsInitialize()\n");return; }

  *ierr = SlepcInitialize_DynamicLibraries();
  if (*ierr) { (*PetscErrorPrintf)("SlepcInitialize:Initializing dynamic libraries\n");return; }

  SlepcInitializeCalled = PETSC_TRUE;
  SlepcFinalizeCalled   = PETSC_FALSE;
  *ierr = PetscInfo(0,"SLEPc successfully started from Fortran\n");
  if (*ierr) { (*PetscErrorPrintf)("SlepcInitialize:Calling PetscInfo()");return; }
}

SLEPC_EXTERN void slepcfinalize_(PetscErrorCode *ierr)
{
  if (*ierr) { (*PetscErrorPrintf)("SlepcInitialize:PetscFinalized failed");return; }
  if (PetscUnlikely(!SlepcInitializeCalled)) {
    (*PetscErrorPrintf)("SlepcInitialize() must be called before SlepcFinalize()");
    return;
  }

  *ierr = PetscInfo(0,"SlepcFinalize() called from Fortran\n");
  if (*ierr) { (*PetscErrorPrintf)("SlepcFinalize:Calling PetscInfo()");return; }
  *ierr = 0;
  if (SlepcBeganPetsc) {
    petscfinalize_(ierr);
    if (*ierr) { (*PetscErrorPrintf)("SlepcFinalize:Calling petscfinalize_()");return; }
    SlepcBeganPetsc = PETSC_FALSE;
  }
  SlepcInitializeCalled = PETSC_FALSE;
  SlepcFinalizeCalled   = PETSC_TRUE;
}

