!
!  - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
!  SLEPc - Scalable Library for Eigenvalue Problem Computations
!  Copyright (c) 2002-, Universitat Politecnica de Valencia, Spain
!
!  This file is part of SLEPc.
!  SLEPc is distributed under a 2-clause BSD license (see LICENSE).
!  - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
!
!  Used by slepcrgmod.F90 to create Fortran module file
!
#include "slepc/finclude/slepcrg.h"

      type tRG
        PetscFortranAddr:: v PETSC_FORTRAN_TYPE_INITIALIZE
      end type tRG

      RG, parameter :: SLEPC_NULL_RG = tRG(0)

      PetscEnum, parameter :: RG_QUADRULE_TRAPEZOIDAL = 1
      PetscEnum, parameter :: RG_QUADRULE_CHEBYSHEV   = 2

#if defined(_WIN32) && defined(PETSC_USE_SHARED_LIBRARIES)
!DEC$ ATTRIBUTES DLLEXPORT::SLEPC_NULL_RG
#endif
