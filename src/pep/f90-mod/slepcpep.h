!
!  - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
!  SLEPc - Scalable Library for Eigenvalue Problem Computations
!  Copyright (c) 2002-, Universitat Politecnica de Valencia, Spain
!
!  This file is part of SLEPc.
!  SLEPc is distributed under a 2-clause BSD license (see LICENSE).
!  - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
!
!  Used by slepcpepmod.F90 to create Fortran module file
!
#include "slepc/finclude/slepcpep.h"

      type, extends(tPetscObject) :: tPEP
      end type tPEP

      PEP, parameter :: SLEPC_NULL_PEP = tPEP(0)

!  Convergence flags.
!  They should match the flags in $SLEPC_DIR/include/slepcpep.h

      PetscEnum, parameter :: PEP_CONVERGED_TOL          =  1
      PetscEnum, parameter :: PEP_CONVERGED_USER         =  2
      PetscEnum, parameter :: PEP_DIVERGED_ITS           = -1
      PetscEnum, parameter :: PEP_DIVERGED_BREAKDOWN     = -2
      PetscEnum, parameter :: PEP_DIVERGED_SYMMETRY_LOST = -3
      PetscEnum, parameter :: PEP_CONVERGED_ITERATING    =  0

      PetscEnum, parameter :: PEP_GENERAL                =  1
      PetscEnum, parameter :: PEP_HERMITIAN              =  2
      PetscEnum, parameter :: PEP_HYPERBOLIC             =  3
      PetscEnum, parameter :: PEP_GYROSCOPIC             =  4

      PetscEnum, parameter :: PEP_LARGEST_MAGNITUDE      =  1
      PetscEnum, parameter :: PEP_SMALLEST_MAGNITUDE     =  2
      PetscEnum, parameter :: PEP_LARGEST_REAL           =  3
      PetscEnum, parameter :: PEP_SMALLEST_REAL          =  4
      PetscEnum, parameter :: PEP_LARGEST_IMAGINARY      =  5
      PetscEnum, parameter :: PEP_SMALLEST_IMAGINARY     =  6
      PetscEnum, parameter :: PEP_TARGET_MAGNITUDE       =  7
      PetscEnum, parameter :: PEP_TARGET_REAL            =  8
      PetscEnum, parameter :: PEP_TARGET_IMAGINARY       =  9
      PetscEnum, parameter :: PEP_WHICH_USER             = 10

      PetscEnum, parameter :: PEP_BASIS_MONOMIAL         =  0
      PetscEnum, parameter :: PEP_BASIS_CHEBYSHEV1       =  1
      PetscEnum, parameter :: PEP_BASIS_CHEBYSHEV2       =  2
      PetscEnum, parameter :: PEP_BASIS_LEGENDRE         =  3
      PetscEnum, parameter :: PEP_BASIS_LAGUERRE         =  4
      PetscEnum, parameter :: PEP_BASIS_HERMITE          =  5

      PetscEnum, parameter :: PEP_SCALE_NONE             =  0
      PetscEnum, parameter :: PEP_SCALE_SCALAR           =  1
      PetscEnum, parameter :: PEP_SCALE_DIAGONAL         =  2
      PetscEnum, parameter :: PEP_SCALE_BOTH             =  3

      PetscEnum, parameter :: PEP_REFINE_NONE            =  0
      PetscEnum, parameter :: PEP_REFINE_SIMPLE          =  1
      PetscEnum, parameter :: PEP_REFINE_MULTIPLE        =  2

      PetscEnum, parameter :: PEP_REFINE_SCHEME_SCHUR    =  1
      PetscEnum, parameter :: PEP_REFINE_SCHEME_MBE      =  2
      PetscEnum, parameter :: PEP_REFINE_SCHEME_EXPLICIT =  3

      PetscEnum, parameter :: PEP_EXTRACT_NONE           =  1
      PetscEnum, parameter :: PEP_EXTRACT_NORM           =  2
      PetscEnum, parameter :: PEP_EXTRACT_RESIDUAL       =  3
      PetscEnum, parameter :: PEP_EXTRACT_STRUCTURED     =  4

      PetscEnum, parameter :: PEP_ERROR_ABSOLUTE         =  0
      PetscEnum, parameter :: PEP_ERROR_RELATIVE         =  1
      PetscEnum, parameter :: PEP_ERROR_BACKWARD         =  2

      PetscEnum, parameter :: PEP_CONV_ABS               =  0
      PetscEnum, parameter :: PEP_CONV_REL               =  1
      PetscEnum, parameter :: PEP_CONV_NORM              =  2
      PetscEnum, parameter :: PEP_CONV_USER              =  3

      PetscEnum, parameter :: PEP_STOP_BASIC             =  0
      PetscEnum, parameter :: PEP_STOP_USER              =  1

      PetscEnum, parameter :: PEP_JD_PROJECTION_HARMONIC   =  0
      PetscEnum, parameter :: PEP_JD_PROJECTION_ORTHOGONAL =  1

      PetscEnum, parameter :: PEP_CISS_EXTRACTION_RITZ   =  0
      PetscEnum, parameter :: PEP_CISS_EXTRACTION_HANKEL =  1
      PetscEnum, parameter :: PEP_CISS_EXTRACTION_CAA    =  2

!
!   Possible arguments to PEPMonitorSet()
!
      external PEPMONITORFIRST
      external PEPMONITORALL
      external PEPMONITORCONVERGED

      external PEPMonitorConvergedDestroy

#if defined(_WIN32) && defined(PETSC_USE_SHARED_LIBRARIES)
!DEC$ ATTRIBUTES DLLEXPORT::SLEPC_NULL_PEP
#endif
