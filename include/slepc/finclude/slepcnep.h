!
!  Include file for Fortran use of the NEP object in SLEPc
!
!
!  - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
!  SLEPc - Scalable Library for Eigenvalue Problem Computations
!  Copyright (c) 2002-2016, Universitat Politecnica de Valencia, Spain
!
!  This file is part of SLEPc.
!
!  SLEPc is free software: you can redistribute it and/or modify it under  the
!  terms of version 3 of the GNU Lesser General Public License as published by
!  the Free Software Foundation.
!
!  SLEPc  is  distributed in the hope that it will be useful, but WITHOUT  ANY
!  WARRANTY;  without even the implied warranty of MERCHANTABILITY or  FITNESS
!  FOR  A  PARTICULAR PURPOSE. See the GNU Lesser General Public  License  for
!  more details.
!
!  You  should have received a copy of the GNU Lesser General  Public  License
!  along with SLEPc. If not, see <http://www.gnu.org/licenses/>.
!  - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
!
#include "slepc/finclude/slepcnepdef.h"

!  Convergence flags.
!  They should match the flags in $SLEPC_DIR/include/slepcnep.h

      PetscEnum NEP_REFINE_NONE
      PetscEnum NEP_REFINE_SIMPLE
      PetscEnum NEP_REFINE_MULTIPLE

      parameter (NEP_REFINE_NONE            =  0)
      parameter (NEP_REFINE_SIMPLE          =  1)
      parameter (NEP_REFINE_MULTIPLE        =  2)

      PetscEnum NEP_REFINE_SCHEME_SCHUR
      PetscEnum NEP_REFINE_SCHEME_MBE
      PetscEnum NEP_REFINE_SCHEME_EXPLICIT

      parameter (NEP_REFINE_SCHEME_SCHUR    =  1)
      parameter (NEP_REFINE_SCHEME_MBE      =  2)
      parameter (NEP_REFINE_SCHEME_EXPLICIT =  3)

      PetscEnum NEP_CONV_ABS
      PetscEnum NEP_CONV_REL
      PetscEnum NEP_CONV_NORM
      PetscEnum NEP_CONV_USER

      parameter (NEP_CONV_ABS               =  0)
      parameter (NEP_CONV_REL               =  1)
      parameter (NEP_CONV_NORM              =  2)
      parameter (NEP_CONV_USER              =  3)

      PetscEnum NEP_STOP_BASIC
      PetscEnum NEP_STOP_USER

      parameter (NEP_STOP_BASIC             =  0)
      parameter (NEP_STOP_USER              =  1)

      PetscEnum NEP_CONVERGED_TOL
      PetscEnum NEP_CONVERGED_USER
      PetscEnum NEP_DIVERGED_ITS
      PetscEnum NEP_DIVERGED_BREAKDOWN
      PetscEnum NEP_DIVERGED_LINEAR_SOLVE
      PetscEnum NEP_CONVERGED_ITERATING

      parameter (NEP_CONVERGED_TOL            =  1)
      parameter (NEP_CONVERGED_USER           =  2)
      parameter (NEP_DIVERGED_ITS             = -1)
      parameter (NEP_DIVERGED_BREAKDOWN       = -2)
      parameter (NEP_DIVERGED_LINEAR_SOLVE    = -4)
      parameter (NEP_CONVERGED_ITERATING      =  0)

      PetscEnum NEP_LARGEST_MAGNITUDE
      PetscEnum NEP_SMALLEST_MAGNITUDE
      PetscEnum NEP_LARGEST_REAL
      PetscEnum NEP_SMALLEST_REAL
      PetscEnum NEP_LARGEST_IMAGINARY
      PetscEnum NEP_SMALLEST_IMAGINARY
      PetscEnum NEP_TARGET_MAGNITUDE
      PetscEnum NEP_TARGET_REAL
      PetscEnum NEP_TARGET_IMAGINARY
      PetscEnum NEP_ALL
      PetscEnum NEP_WHICH_USER

      parameter (NEP_LARGEST_MAGNITUDE      =  1)
      parameter (NEP_SMALLEST_MAGNITUDE     =  2)
      parameter (NEP_LARGEST_REAL           =  3)
      parameter (NEP_SMALLEST_REAL          =  4)
      parameter (NEP_LARGEST_IMAGINARY      =  5)
      parameter (NEP_SMALLEST_IMAGINARY     =  6)
      parameter (NEP_TARGET_MAGNITUDE       =  7)
      parameter (NEP_TARGET_REAL            =  8)
      parameter (NEP_TARGET_IMAGINARY       =  9)
      parameter (NEP_ALL                    = 10)
      parameter (NEP_WHICH_USER             = 11)

      PetscEnum NEP_ERROR_ABSOLUTE
      PetscEnum NEP_ERROR_RELATIVE
      PetscEnum NEP_ERROR_BACKWARD

      parameter (NEP_ERROR_ABSOLUTE         =  0)
      parameter (NEP_ERROR_RELATIVE         =  1)
      parameter (NEP_ERROR_BACKWARD         =  2)

!
!   Possible arguments to NEPMonitorSet()
!
      external NEPMONITORALL
      external NEPMONITORLG
      external NEPMONITORLGALL
      external NEPMONITORCONVERGED
      external NEPMONITORFIRST

!
!  End of Fortran include file for the NEP package in SLEPc
!
