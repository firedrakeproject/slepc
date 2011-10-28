/*
   This include file contains definitions of system functions. It is included
   by all other SLEPc include files.

   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
   SLEPc - Scalable Library for Eigenvalue Problem Computations
   Copyright (c) 2002-2011, Universitat Politecnica de Valencia, Spain

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

/*
    Default tolerance for the different solvers, depending on the precision
*/
#if defined(PETSC_USE_REAL_SINGLE)
#  define SLEPC_DEFAULT_TOL   1e-6
#elif defined(PETSC_USE_REAL_DOUBLE)
#  define SLEPC_DEFAULT_TOL   1e-8
#elif defined(PETSC_USE_REAL___FLOAT128)
#  define SLEPC_DEFAULT_TOL   1e-16
#else
#  define SLEPC_DEFAULT_TOL   1e-7
#endif


PETSC_EXTERN_CXX_BEGIN
/*
    Initialization of SLEPc and other system routines
*/
extern PetscErrorCode SlepcInitialize(int*,char***,const char[],const char[]);
extern PetscErrorCode SlepcFinalize(void);
extern PetscErrorCode SlepcInitializeFortran(void);
extern PetscErrorCode SlepcInitialized(PetscBool*);

#undef __FUNCT__ 
#define __FUNCT__ "SlepcAbs"
/*@C
   SlepcAbs - Returns sqrt(x**2+y**2), taking care not to cause unnecessary
   overflow. It is based on LAPACK's DLAPY2.

   Not Collective

   Input parameters:
.  x,y - the real numbers

   Output parameter:
.  return - the result

   Level: developer
@*/
PETSC_STATIC_INLINE PetscReal SlepcAbs(PetscReal x,PetscReal y)
{
  PetscReal w,z,t,xabs=PetscAbs(x),yabs=PetscAbs(y);

  w = PetscMax(xabs,yabs);
  z = PetscMin(xabs,yabs);
  if (z == 0.0) return w;
  t = z/w;
  return w*PetscSqrtReal(1.0+t*t);
}


/*MC
   SlepcAbsEigenvalue - Returns the absolute value of a complex number given
   its real and imaginary parts.

  Synopsis:
   PetscReal SlepcAbsEigenvalue(PetscScalar x,PetscScalar y)
   Not Collective

   Input parameters:
+  x  - the real part of the complex number
-  y  - the imaginary part of the complex number

   Notes: 
   This function computes sqrt(x**2+y**2), taking care not to cause unnecessary
   overflow. It is based on LAPACK's DLAPY2.

   Level: developer

M*/
#if !defined(PETSC_USE_COMPLEX)
#define SlepcAbsEigenvalue(x,y) SlepcAbs(x,y)
#else
#define SlepcAbsEigenvalue(x,y) PetscAbsScalar(x)
#endif
extern PetscErrorCode SlepcMatConvertSeqDense(Mat,Mat*);
extern PetscErrorCode SlepcMatTile(PetscScalar,Mat,PetscScalar,Mat,PetscScalar,Mat,PetscScalar,Mat,Mat*);
extern PetscErrorCode SlepcCheckOrthogonality(Vec*,PetscInt,Vec *,PetscInt,Mat,PetscReal*);
 
extern PetscBool SlepcInitializeCalled;

PETSC_EXTERN_CXX_END
#endif

