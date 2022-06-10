!
!  - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
!  SLEPc - Scalable Library for Eigenvalue Problem Computations
!  Copyright (c) 2002-, Universitat Politecnica de Valencia, Spain
!
!  This file is part of SLEPc.
!  SLEPc is distributed under a 2-clause BSD license (see LICENSE).
!  - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
!
!  Used by slepcfnmod.F90 to create Fortran module file
!
#include "slepc/finclude/slepcfn.h"

      type tFN
        PetscFortranAddr:: v PETSC_FORTRAN_TYPE_INITIALIZE
      end type tFN

      FN, parameter :: SLEPC_NULL_FN = tFN(0)

      PetscEnum, parameter :: FN_COMBINE_ADD           =  0
      PetscEnum, parameter :: FN_COMBINE_MULTIPLY      =  1
      PetscEnum, parameter :: FN_COMBINE_DIVIDE        =  2
      PetscEnum, parameter :: FN_COMBINE_COMPOSE       =  3

      PetscEnum, parameter :: FN_PARALLEL_REDUNDANT    =  0
      PetscEnum, parameter :: FN_PARALLEL_SYNCHRONIZED =  1

#if defined(_WIN32) && defined(PETSC_USE_SHARED_LIBRARIES)
!DEC$ ATTRIBUTES DLLEXPORT::SLEPC_NULL_FN
#endif
