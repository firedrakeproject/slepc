!
!  - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
!  SLEPc - Scalable Library for Eigenvalue Problem Computations
!  Copyright (c) 2002-2018, Universitat Politecnica de Valencia, Spain
!
!  This file is part of SLEPc.
!  SLEPc is distributed under a 2-clause BSD license (see LICENSE).
!  - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
!
!  Include file for Fortran use of the DS object in SLEPc
!
#include "slepc/finclude/slepcds.h"

      type tDS
        PetscFortranAddr:: v
      end type tDS

      PetscEnum DS_STATE_RAW
      PetscEnum DS_STATE_INTERMEDIATE
      PetscEnum DS_STATE_CONDENSED
      PetscEnum DS_STATE_TRUNCATED

      parameter (DS_STATE_RAW             =  0)
      parameter (DS_STATE_INTERMEDIATE    =  1)
      parameter (DS_STATE_CONDENSED       =  2)
      parameter (DS_STATE_TRUNCATED       =  3)

      PetscEnum DS_MAT_A
      PetscEnum DS_MAT_B
      PetscEnum DS_MAT_C
      PetscEnum DS_MAT_T
      PetscEnum DS_MAT_D
      PetscEnum DS_MAT_F
      PetscEnum DS_MAT_Q
      PetscEnum DS_MAT_Z
      PetscEnum DS_MAT_X
      PetscEnum DS_MAT_Y
      PetscEnum DS_MAT_U
      PetscEnum DS_MAT_VT
      PetscEnum DS_MAT_W
      PetscEnum DS_MAT_E0
      PetscEnum DS_MAT_E1
      PetscEnum DS_MAT_E2
      PetscEnum DS_MAT_E3
      PetscEnum DS_MAT_E4
      PetscEnum DS_MAT_E5
      PetscEnum DS_MAT_E6
      PetscEnum DS_MAT_E7
      PetscEnum DS_MAT_E8
      PetscEnum DS_MAT_E9
      PetscEnum DS_NUM_MAT

      parameter (DS_MAT_A         =  0)
      parameter (DS_MAT_B         =  1)
      parameter (DS_MAT_C         =  2)
      parameter (DS_MAT_T         =  3)
      parameter (DS_MAT_D         =  4)
      parameter (DS_MAT_F         =  5)
      parameter (DS_MAT_Q         =  6)
      parameter (DS_MAT_Z         =  7)
      parameter (DS_MAT_X         =  8)
      parameter (DS_MAT_Y         =  9)
      parameter (DS_MAT_U         = 10)
      parameter (DS_MAT_VT        = 11)
      parameter (DS_MAT_W         = 12)
      parameter (DS_MAT_E0        = 13)
      parameter (DS_MAT_E1        = 14)
      parameter (DS_MAT_E2        = 15)
      parameter (DS_MAT_E3        = 16)
      parameter (DS_MAT_E4        = 17)
      parameter (DS_MAT_E5        = 18)
      parameter (DS_MAT_E6        = 19)
      parameter (DS_MAT_E7        = 20)
      parameter (DS_MAT_E8        = 21)
      parameter (DS_MAT_E9        = 22)
      parameter (DS_NUM_MAT       = 23)

      PetscEnum DS_PARALLEL_REDUNDANT
      PetscEnum DS_PARALLEL_SYNCHRONIZED

      parameter (DS_PARALLEL_REDUNDANT    = 0)
      parameter (DS_PARALLEL_SYNCHRONIZED = 1)

!
!  End of Fortran include file for the DS package in SLEPc
!
