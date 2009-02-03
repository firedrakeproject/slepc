/*
  This file contains the Fortran version of SlepcInitialize().

   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
   SLEPc - Scalable Library for Eigenvalue Problem Computations
   Copyright (c) 2002-2009, Universidad Politecnica de Valencia, Spain

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

#include "private/fortranimpl.h" 
#include "slepc.h"
#include "slepcst.h"
#include "slepceps.h"
#include "slepcsvd.h"
#include "slepcip.h"

extern PetscTruth SlepcBeganPetsc;

extern PetscTruth SlepcInitializeCalled;

#ifdef PETSC_HAVE_FORTRAN_CAPS
#define petscinitialize_              PETSCINITIALIZE
#define slepcinitialize_              SLEPCINITIALIZE
#elif !defined(PETSC_HAVE_FORTRAN_UNDERSCORE)
#define petscinitialize_              petscinitialize
#define slepcinitialize_              slepcinitialize
#endif

EXTERN_C_BEGIN
extern void PETSC_STDCALL petscinitialize_(CHAR filename PETSC_MIXED_LEN(len),PetscErrorCode *ierr PETSC_END_LEN(len));
EXTERN_C_END

#if defined(PETSC_USE_DYNAMIC_LIBRARIES)
extern PetscDLLibrary DLLibrariesLoaded;
#endif

EXTERN_C_BEGIN
/*
    SlepcInitialize - Version called from Fortran.

    Notes:
    Since this routine is called from Fortran it does not return error codes.
*/
void PETSC_STDCALL slepcinitialize_(CHAR filename PETSC_MIXED_LEN(len),PetscErrorCode *ierr PETSC_END_LEN(len))
{
#if defined(PETSC_USE_DYNAMIC_LIBRARIES)
  char       libs[PETSC_MAX_PATH_LEN],dlib[PETSC_MAX_PATH_LEN];
  PetscTruth found;
#endif
  *ierr = 1;
  if (SlepcInitializeCalled) {*ierr = 0; return;}

  if (!PetscInitializeCalled) {
#if defined(PETSC_HAVE_FORTRAN_MIXED_STR_ARG)
    petscinitialize_(filename,len,ierr);
#else
    petscinitialize_(filename,ierr,len);
#endif
    if (*ierr) { (*PetscErrorPrintf)("SlepcInitialize:PetscInitialize failed");return;}
    SlepcBeganPetsc = PETSC_TRUE;
  }

#if defined(PETSC_USE_DYNAMIC_LIBRARIES)
  *ierr = PetscStrcpy(libs,SLEPC_LIB_DIR);if (*ierr) return;
  *ierr = PetscStrcat(libs,"/libslepc");if (*ierr) return;
  *ierr = PetscDLLibraryRetrieve(PETSC_COMM_WORLD,libs,dlib,1024,&found);if (*ierr) return;
  if (found) {
    *ierr = PetscDLLibraryAppend(PETSC_COMM_WORLD,&DLLibrariesLoaded,libs);if (*ierr) return;
  } else {
    *ierr = 1;
    (*PetscErrorPrintf)("Unable to locate SLEPc dynamic library %s \n",libs);
    return;
  }
#else
  *ierr = STInitializePackage(PETSC_NULL); if (*ierr) { (*PetscErrorPrintf)("SlepcInitialize:Initializing ST package");return;}
  *ierr = EPSInitializePackage(PETSC_NULL); if (*ierr) { (*PetscErrorPrintf)("SlepcInitialize:Initializing EPS package");return;}
  *ierr = SVDInitializePackage(PETSC_NULL); if (*ierr) { (*PetscErrorPrintf)("SlepcInitialize:Initializing SVD package");return;}
  *ierr = IPInitializePackage(PETSC_NULL); if (*ierr) { (*PetscErrorPrintf)("SlepcInitialize:Initializing IP package");return;}
#endif

  SlepcInitializeCalled = PETSC_TRUE;
  *ierr = PetscInfo(0,"SLEPc successfully started from Fortran\n");
  if (*ierr) { (*PetscErrorPrintf)("SlepcInitialize:Calling PetscInfo()");return;}

}  

EXTERN_C_END
