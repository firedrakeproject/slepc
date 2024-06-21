!
!  - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
!  SLEPc - Scalable Library for Eigenvalue Problem Computations
!  Copyright (c) 2002-, Universitat Politecnica de Valencia, Spain
!
!  This file is part of SLEPc.
!  SLEPc is distributed under a 2-clause BSD license (see LICENSE).
!  - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
!
!  Used by slepcdsmod.F90 to create Fortran module file
!
#include "slepc/finclude/slepcds.h"

      type, extends(tPetscObject) :: tDS
      end type tDS

      DS, parameter :: SLEPC_NULL_DS = tDS(0)

      PetscEnum, parameter :: DS_STATE_RAW             =  0
      PetscEnum, parameter :: DS_STATE_INTERMEDIATE    =  1
      PetscEnum, parameter :: DS_STATE_CONDENSED       =  2
      PetscEnum, parameter :: DS_STATE_TRUNCATED       =  3

      PetscEnum, parameter :: DS_MAT_A         =  0
      PetscEnum, parameter :: DS_MAT_B         =  1
      PetscEnum, parameter :: DS_MAT_C         =  2
      PetscEnum, parameter :: DS_MAT_T         =  3
      PetscEnum, parameter :: DS_MAT_D         =  4
      PetscEnum, parameter :: DS_MAT_F         =  5
      PetscEnum, parameter :: DS_MAT_Q         =  6
      PetscEnum, parameter :: DS_MAT_Z         =  7
      PetscEnum, parameter :: DS_MAT_X         =  8
      PetscEnum, parameter :: DS_MAT_Y         =  9
      PetscEnum, parameter :: DS_MAT_U         = 10
      PetscEnum, parameter :: DS_MAT_V         = 11
      PetscEnum, parameter :: DS_MAT_W         = 12
      PetscEnum, parameter :: DS_MAT_E0        = 13
      PetscEnum, parameter :: DS_MAT_E1        = 14
      PetscEnum, parameter :: DS_MAT_E2        = 15
      PetscEnum, parameter :: DS_MAT_E3        = 16
      PetscEnum, parameter :: DS_MAT_E4        = 17
      PetscEnum, parameter :: DS_MAT_E5        = 18
      PetscEnum, parameter :: DS_MAT_E6        = 19
      PetscEnum, parameter :: DS_MAT_E7        = 20
      PetscEnum, parameter :: DS_MAT_E8        = 21
      PetscEnum, parameter :: DS_MAT_E9        = 22
      PetscEnum, parameter :: DS_NUM_MAT       = 23

      PetscEnum, parameter :: DS_PARALLEL_REDUNDANT    = 0
      PetscEnum, parameter :: DS_PARALLEL_SYNCHRONIZED = 1
      PetscEnum, parameter :: DS_PARALLEL_DISTRIBUTED  = 2

#if defined(_WIN32) && defined(PETSC_USE_SHARED_LIBRARIES)
!DEC$ ATTRIBUTES DLLEXPORT::SLEPC_NULL_DS
#endif
