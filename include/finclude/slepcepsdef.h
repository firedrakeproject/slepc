!
!  Include file for Fortran use of the EPS object in SLEPc
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
#if !defined(__SLEPCEPS_H)
#define __SLEPCEPS_H

#include "finclude/slepcstdef.h"
#include "finclude/slepcipdef.h"

#if !defined(PETSC_USE_FORTRAN_DATATYPES)
#define EPS                    PetscFortranAddr
#endif

#define EPSType                character*(80)
#define EPSConvergedReason     PetscEnum
#define EPSProblemType         PetscEnum
#define EPSWhich               PetscEnum
#define EPSClass               PetscEnum
#define EPSExtraction          PetscEnum
#define EPSPowerShiftType      PetscEnum
#define EPSLanczosReorthogType PetscEnum
#define EPSPRIMMEMethod        PetscEnum
#define EPSPRIMMEPrecond       PetscEnum


#define EPSPOWER       'power'
#define EPSSUBSPACE    'subspace'
#define EPSARNOLDI     'arnoldi'
#define EPSLANCZOS     'lanczos'
#define EPSKRYLOVSCHUR 'krylovschur'
#define EPSLAPACK      'lapack'
#define EPSARPACK      'arpack'
#define EPSBLZPACK     'blzpack'
#define EPSTRLAN       'trlan'
#define EPSBLOPEX      'blopex'
#define EPSPRIMME      'primme'

#endif
