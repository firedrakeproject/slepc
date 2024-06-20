!
!  - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
!  SLEPc - Scalable Library for Eigenvalue Problem Computations
!  Copyright (c) 2002-, Universitat Politecnica de Valencia, Spain
!
!  This file is part of SLEPc.
!  SLEPc is distributed under a 2-clause BSD license (see LICENSE).
!  - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
!
!  Used by slepcnepmod.F90 to create Fortran module file
!
#include "slepc/finclude/slepcnep.h"

      type, extends(tPetscObject) :: tNEP
      end type tNEP

      NEP, parameter :: SLEPC_NULL_NEP = tNEP(0)

!  Convergence flags.
!  They should match the flags in $SLEPC_DIR/include/slepcnep.h

      PetscEnum, parameter :: NEP_REFINE_NONE            =  0
      PetscEnum, parameter :: NEP_REFINE_SIMPLE          =  1
      PetscEnum, parameter :: NEP_REFINE_MULTIPLE        =  2

      PetscEnum, parameter :: NEP_REFINE_SCHEME_SCHUR    =  1
      PetscEnum, parameter :: NEP_REFINE_SCHEME_MBE      =  2
      PetscEnum, parameter :: NEP_REFINE_SCHEME_EXPLICIT =  3

      PetscEnum, parameter :: NEP_CONV_ABS               =  0
      PetscEnum, parameter :: NEP_CONV_REL               =  1
      PetscEnum, parameter :: NEP_CONV_NORM              =  2
      PetscEnum, parameter :: NEP_CONV_USER              =  3

      PetscEnum, parameter :: NEP_STOP_BASIC             =  0
      PetscEnum, parameter :: NEP_STOP_USER              =  1

      PetscEnum, parameter :: NEP_CONVERGED_TOL          =  1
      PetscEnum, parameter :: NEP_CONVERGED_USER         =  2
      PetscEnum, parameter :: NEP_DIVERGED_ITS           = -1
      PetscEnum, parameter :: NEP_DIVERGED_BREAKDOWN     = -2
      PetscEnum, parameter :: NEP_DIVERGED_LINEAR_SOLVE  = -4
      PetscEnum, parameter :: NEP_DIVERGED_SUBSPACE_EXHAUSTED = -5
      PetscEnum, parameter :: NEP_CONVERGED_ITERATING    =  0

      PetscEnum, parameter :: NEP_GENERAL                =  1
      PetscEnum, parameter :: NEP_RATIONAL               =  2

      PetscEnum, parameter :: NEP_LARGEST_MAGNITUDE      =  1
      PetscEnum, parameter :: NEP_SMALLEST_MAGNITUDE     =  2
      PetscEnum, parameter :: NEP_LARGEST_REAL           =  3
      PetscEnum, parameter :: NEP_SMALLEST_REAL          =  4
      PetscEnum, parameter :: NEP_LARGEST_IMAGINARY      =  5
      PetscEnum, parameter :: NEP_SMALLEST_IMAGINARY     =  6
      PetscEnum, parameter :: NEP_TARGET_MAGNITUDE       =  7
      PetscEnum, parameter :: NEP_TARGET_REAL            =  8
      PetscEnum, parameter :: NEP_TARGET_IMAGINARY       =  9
      PetscEnum, parameter :: NEP_ALL                    = 10
      PetscEnum, parameter :: NEP_WHICH_USER             = 11

      PetscEnum, parameter :: NEP_ERROR_ABSOLUTE         =  0
      PetscEnum, parameter :: NEP_ERROR_RELATIVE         =  1
      PetscEnum, parameter :: NEP_ERROR_BACKWARD         =  2

      PetscEnum, parameter :: NEP_CISS_EXTRACTION_RITZ   =  0
      PetscEnum, parameter :: NEP_CISS_EXTRACTION_HANKEL =  1
      PetscEnum, parameter :: NEP_CISS_EXTRACTION_CAA    =  2

!
!   Possible arguments to NEPMonitorSet()
!
      external NEPMONITORFIRST
      external NEPMONITORALL
      external NEPMONITORCONVERGED

      external NEPMonitorConvergedDestroy

#if defined(_WIN32) && defined(PETSC_USE_SHARED_LIBRARIES)
!DEC$ ATTRIBUTES DLLEXPORT::SLEPC_NULL_NEP
#endif
