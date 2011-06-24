/*
   This include file contains definitions of system functions. It is included
   by all other SLEPc include files.

   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
   SLEPc - Scalable Library for Eigenvalue Problem Computations
   Copyright (c) 2002-2010, Universidad Politecnica de Valencia, Spain

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

#if !defined(__SLEPC_H)
#define __SLEPC_H

/* ========================================================================== */
/* 
   slepcconf.h is created by the configure script and placed in ${PETSC_ARCH}/include.
   It contains macro definitions set at configure time.
*/
#include "slepcconf.h"

/* ========================================================================== */
/* 
   Current SLEPc version number and release date
*/
#include "slepcversion.h"
#define SLEPC_AUTHOR_INFO        "       The SLEPc Team\n    slepc-maint@grycap.upv.es\n http://www.grycap.upv.es/slepc\n"
#if (SLEPC_VERSION_RELEASE == 1)
#define SlepcGetVersion(version,len) PetscSNPrintf(version,len,"SLEPc Release Version %d.%d, Patch %d, %s", \
                                         SLEPC_VERSION_MAJOR,SLEPC_VERSION_MINOR, \
                                         SLEPC_VERSION_PATCH,SLEPC_VERSION_PATCH_DATE)
#else
#define SlepcGetVersion(version,len) PetscSNPrintf(version,len,"SLEPc Development SVN revision: %d  SVN Date: %s", \
                                        SLEPC_VERSION_SVN, SLEPC_VERSION_DATE_SVN)
#endif
/*MC
    SlepcGetVersion - Gets the SLEPc version information in a string.

    Input Parameter:
.   len - length of the string

    Output Parameter:
.   version - version string

    Level: developer

    Usage:
    char version[256];
    ierr = SlepcGetVersion(version,256);CHKERRQ(ierr)

    Fortran Note:
    This routine is not supported in Fortran.
M*/

/* ========================================================================== */
/* 
   The PETSc include files. 
*/
#include "petscsys.h"
#include "petscvec.h"
#include "petscmat.h"
/*
    slepcvec.h contains extensions to PETSc Vec's
*/
#include "slepcvec.h"
/*
    slepcimpl.h contains definitions common to all SLEPc objects
*/
#include "private/slepcimpl.h"

PETSC_EXTERN_CXX_BEGIN
/*
    Initialization of SLEPc and other system routines
*/
extern PetscErrorCode SlepcInitialize(int*,char***,const char[],const char[]);
extern PetscErrorCode SlepcFinalize(void);
extern PetscErrorCode SlepcInitializeFortran(void);
extern PetscErrorCode SlepcInitialized(PetscBool*);

#if !defined(PETSC_USE_COMPLEX)
extern PetscReal SlepcAbsEigenvalue(PetscScalar,PetscScalar);
#else
#define SlepcAbsEigenvalue(x,y) PetscAbsScalar(x)
#endif
extern PetscErrorCode SlepcMatConvertSeqDense(Mat,Mat*);
extern PetscErrorCode SlepcMatTile(PetscScalar,Mat,PetscScalar,Mat,PetscScalar,Mat,PetscScalar,Mat,Mat*);
extern PetscErrorCode SlepcCheckOrthogonality(Vec*,PetscInt,Vec *,PetscInt,Mat,PetscScalar*);
 
extern PetscBool SlepcInitializeCalled;

PETSC_EXTERN_CXX_END
#endif

