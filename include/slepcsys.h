/*
   This include file contains definitions of system functions. It is included
   by all other SLEPc include files.

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

#if !defined(__SLEPCSYS_H)
#define __SLEPCSYS_H

/* ========================================================================== */
/* 
   slepcconf.h is created by the configure script and placed in ${PETSC_ARCH}/include.
   It contains macro definitions set at configure time.
*/
#include "slepcconf.h"
/*
    slepcversion.h contains version info
*/
#include "slepcversion.h"
#define SLEPC_AUTHOR_INFO "       The SLEPc Team\n    slepc-maint@grycap.upv.es\n http://www.grycap.upv.es/slepc\n"

/* ========================================================================== */
/* 
   The PETSc include files. 
*/
#include "petscsys.h"
#include "petscvec.h"
#include "petscmat.h"
/*
    slepcmath.h contains definition of basic math functions
*/
#include "slepcmath.h"
/*
    slepcvec.h contains extensions to PETSc Vec's
*/
#include "slepcvec.h"
/*
    slepcimpl.h contains definitions common to all SLEPc objects
*/
#include "slepc-private/slepcimpl.h"

/*
    Initialization of SLEPc and other system routines
*/
PETSC_EXTERN PetscErrorCode SlepcInitialize(int*,char***,const char[],const char[]);
PETSC_EXTERN PetscErrorCode SlepcInitializeNoPointers(int,char**,const char[],const char[]);
PETSC_EXTERN PetscErrorCode SlepcInitializeNoArguments(void);
PETSC_EXTERN PetscErrorCode SlepcFinalize(void);
PETSC_EXTERN PetscErrorCode SlepcInitializeFortran(void);
PETSC_EXTERN PetscErrorCode SlepcInitialized(PetscBool*);
PETSC_EXTERN PetscErrorCode SlepcGetVersion(char[],size_t);

PETSC_EXTERN PetscErrorCode SlepcMatConvertSeqDense(Mat,Mat*);
PETSC_EXTERN PetscErrorCode SlepcMatTile(PetscScalar,Mat,PetscScalar,Mat,PetscScalar,Mat,PetscScalar,Mat,Mat*);
PETSC_EXTERN PetscErrorCode SlepcCheckOrthogonality(Vec*,PetscInt,Vec *,PetscInt,Mat,PetscReal*);
 
PETSC_EXTERN PetscBool SlepcInitializeCalled;

#endif

