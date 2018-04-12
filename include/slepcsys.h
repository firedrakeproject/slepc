/*
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
   SLEPc - Scalable Library for Eigenvalue Problem Computations
   Copyright (c) 2002-2018, Universitat Politecnica de Valencia, Spain

   This file is part of SLEPc.
   SLEPc is distributed under a 2-clause BSD license (see LICENSE).
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
*/
/*
   This include file contains definitions of system functions. It is included
   by all other SLEPc include files.
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
    slepcmat.h, slepcvec.h contain utilities related to Mat and Vec, extend functionality in PETSc
*/
#include <slepcmat.h>
#include <slepcvec.h>

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
PETSC_EXTERN PetscErrorCode SlepcGetVersionNumber(PetscInt*,PetscInt*,PetscInt*,PetscInt*);

PETSC_EXTERN PetscErrorCode SlepcSNPrintfScalar(char*,size_t,PetscScalar,PetscBool);

PETSC_DEPRECATED("Use MatCreateRedundantMatrix() followed by MatConvert()") PETSC_STATIC_INLINE PetscErrorCode SlepcMatConvertSeqDense(Mat mat,Mat *newmat) {
  PetscErrorCode ierr; Mat Ar; 
  ierr = MatCreateRedundantMatrix(mat,0,PETSC_COMM_SELF,MAT_INITIAL_MATRIX,&Ar);CHKERRQ(ierr);
  ierr = MatConvert(Ar,MATSEQDENSE,MAT_INITIAL_MATRIX,newmat);CHKERRQ(ierr);
  ierr = MatDestroy(&Ar);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
PETSC_DEPRECATED("Use VecNormalizeComplex()") PETSC_STATIC_INLINE PetscErrorCode SlepcVecNormalize(Vec xr,Vec xi,PetscBool c,PetscReal *nrm) {return VecNormalizeComplex(xr,xi,c,nrm);}
PETSC_DEPRECATED("Use VecCheckOrthogonality()") PETSC_STATIC_INLINE PetscErrorCode SlepcCheckOrthogonality(Vec *V,PetscInt nv,Vec *W,PetscInt nw,Mat B,PetscViewer viewer,PetscReal *lev) {return VecCheckOrthogonality(V,nv,W,nw,B,viewer,lev);}
PETSC_DEPRECATED("Use MatCreateTile()") PETSC_STATIC_INLINE PetscErrorCode SlepcMatTile(PetscScalar a,Mat A,PetscScalar b,Mat B,PetscScalar c,Mat C,PetscScalar d,Mat D,Mat *G) {return MatCreateTile(a,A,b,B,c,C,d,D,G);}

PETSC_EXTERN PetscBool SlepcInitializeCalled;

#endif

