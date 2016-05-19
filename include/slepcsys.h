/*
   This include file contains definitions of system functions. It is included
   by all other SLEPc include files.

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

#if !defined(__SLEPCSYS_H)
#define __SLEPCSYS_H

/* ========================================================================== */
/*
   slepcconf.h is created by the configure script and placed in ${PETSC_ARCH}/include.
   It contains macro definitions set at configure time.
*/
#include <slepcconf.h>
/*
    slepcversion.h contains version info
*/
#include <slepcversion.h>
#define SLEPC_AUTHOR_INFO "       The SLEPc Team\n    slepc-maint@upv.es\n http://slepc.upv.es\n"

/* ========================================================================== */
/*
   The PETSc include files.
*/
#include <petscmat.h>
/*
    slepcmath.h contains definition of basic math functions
*/
#include <slepcmath.h>
/*
    slepcsc.h contains definition of sorting criterion
*/
#include <slepcsc.h>

/*
    Creation and destruction of context for monitors of type XXXMonitorConverged
*/
typedef struct _n_SlepcConvMonitor* SlepcConvMonitor;
PETSC_EXTERN PetscErrorCode SlepcConvMonitorCreate(PetscViewer,PetscViewerFormat,SlepcConvMonitor*);
PETSC_EXTERN PetscErrorCode SlepcConvMonitorDestroy(SlepcConvMonitor*);

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
PETSC_EXTERN PetscErrorCode SlepcCheckOrthogonality(Vec*,PetscInt,Vec*,PetscInt,Mat,PetscViewer,PetscReal*);
PETSC_EXTERN PetscErrorCode SlepcSNPrintfScalar(char*,size_t,PetscScalar,PetscBool);
PETSC_EXTERN PetscErrorCode SlepcVecNormalize(Vec,Vec,PetscBool,PetscReal*);

PETSC_EXTERN PetscBool SlepcInitializeCalled;

#endif

