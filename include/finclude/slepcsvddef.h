!
!  Include file for Fortran use of the SVD object in SLEPc
!
!
!  - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
!     SLEPc - Scalable Library for Eigenvalue Problem Computations
!     Copyright (c) 2002-2007, Universidad Politecnica de Valencia, Spain
!
!     This file is part of SLEPc. See the README file for conditions of use
!     and additional information.
!  - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
!
#if !defined(__SLEPCSVD_H)
#define __SLEPCSVD_H

#include "finclude/slepcipdef.h"
#include "finclude/slepcepsdef.h"

#if !defined(PETSC_USE_FORTRAN_DATATYPES)
#define SVD                PetscFortranAddr
#endif

#define SVDType            character*(80)
#define SVDTransposeMode   PetscEnum
#define SVDWhich           PetscEnum
#define SVDConvergedReason PetscEnum

#define SVDCROSS     'cross'
#define SVDCYCLIC    'cyclic'
#define SVDLAPACK    'lapack'
#define SVDLANCZOS   'lanczos'
#define SVDTRLANCZOS 'trlanczos'

#endif
