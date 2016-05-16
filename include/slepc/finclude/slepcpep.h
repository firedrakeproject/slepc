!
!  Include file for Fortran use of the PEP object in SLEPc
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
#include "slepc/finclude/slepcpepdef.h"

!  Convergence flags.
!  They should match the flags in $SLEPC_DIR/include/slepcpep.h

      PetscEnum PEP_CONVERGED_TOL
      PetscEnum PEP_CONVERGED_USER
      PetscEnum PEP_DIVERGED_ITS
      PetscEnum PEP_DIVERGED_BREAKDOWN
      PetscEnum PEP_DIVERGED_SYMMETRY_LOST
      PetscEnum PEP_CONVERGED_ITERATING

      parameter (PEP_CONVERGED_TOL          =  1)
      parameter (PEP_CONVERGED_USER         =  2)
      parameter (PEP_DIVERGED_ITS           = -1)
      parameter (PEP_DIVERGED_BREAKDOWN     = -2)
      parameter (PEP_DIVERGED_SYMMETRY_LOST = -3)
      parameter (PEP_CONVERGED_ITERATING    =  0)

      PetscEnum PEP_GENERAL
      PetscEnum PEP_HERMITIAN
      PetscEnum PEP_GYROSCOPIC

      parameter (PEP_GENERAL                =  1)
      parameter (PEP_HERMITIAN              =  2)
      parameter (PEP_GYROSCOPIC             =  3)

      PetscEnum PEP_LARGEST_MAGNITUDE
      PetscEnum PEP_SMALLEST_MAGNITUDE
      PetscEnum PEP_LARGEST_REAL
      PetscEnum PEP_SMALLEST_REAL
      PetscEnum PEP_LARGEST_IMAGINARY
      PetscEnum PEP_SMALLEST_IMAGINARY
      PetscEnum PEP_TARGET_MAGNITUDE
      PetscEnum PEP_TARGET_REAL
      PetscEnum PEP_TARGET_IMAGINARY
      PetscEnum PEP_WHICH_USER

      parameter (PEP_LARGEST_MAGNITUDE      =  1)
      parameter (PEP_SMALLEST_MAGNITUDE     =  2)
      parameter (PEP_LARGEST_REAL           =  3)
      parameter (PEP_SMALLEST_REAL          =  4)
      parameter (PEP_LARGEST_IMAGINARY      =  5)
      parameter (PEP_SMALLEST_IMAGINARY     =  6)
      parameter (PEP_TARGET_MAGNITUDE       =  7)
      parameter (PEP_TARGET_REAL            =  8)
      parameter (PEP_TARGET_IMAGINARY       =  9)
      parameter (PEP_WHICH_USER             = 10)

      PetscEnum PEP_BASIS_MONOMIAL
      PetscEnum PEP_BASIS_CHEBYSHEV1
      PetscEnum PEP_BASIS_CHEBYSHEV2
      PetscEnum PEP_BASIS_LEGENDRE
      PetscEnum PEP_BASIS_LAGUERRE
      PetscEnum PEP_BASIS_HERMITE

      parameter (PEP_BASIS_MONOMIAL         =  0)
      parameter (PEP_BASIS_CHEBYSHEV1       =  1)
      parameter (PEP_BASIS_CHEBYSHEV2       =  2)
      parameter (PEP_BASIS_LEGENDRE         =  3)
      parameter (PEP_BASIS_LAGUERRE         =  4)
      parameter (PEP_BASIS_HERMITE          =  5)

      PetscEnum PEP_SCALE_NONE
      PetscEnum PEP_SCALE_SCALAR
      PetscEnum PEP_SCALE_DIAGONAL
      PetscEnum PEP_SCALE_BOTH

      parameter (PEP_SCALE_NONE             =  0)
      parameter (PEP_SCALE_SCALAR           =  1)
      parameter (PEP_SCALE_DIAGONAL         =  2)
      parameter (PEP_SCALE_BOTH             =  3)

      PetscEnum PEP_REFINE_NONE
      PetscEnum PEP_REFINE_SIMPLE
      PetscEnum PEP_REFINE_MULTIPLE

      parameter (PEP_REFINE_NONE            =  0)
      parameter (PEP_REFINE_SIMPLE          =  1)
      parameter (PEP_REFINE_MULTIPLE        =  2)

      PetscEnum PEP_REFINE_SCHEME_SCHUR
      PetscEnum PEP_REFINE_SCHEME_MBE
      PetscEnum PEP_REFINE_SCHEME_EXPLICIT

      parameter (PEP_REFINE_SCHEME_SCHUR    =  1)
      parameter (PEP_REFINE_SCHEME_MBE      =  2)
      parameter (PEP_REFINE_SCHEME_EXPLICIT =  3)

      PetscEnum PEP_EXTRACT_NONE
      PetscEnum PEP_EXTRACT_NORM
      PetscEnum PEP_EXTRACT_RESIDUAL
      PetscEnum PEP_EXTRACT_STRUCTURED

      parameter (PEP_EXTRACT_NONE           =  1)
      parameter (PEP_EXTRACT_NORM           =  2)
      parameter (PEP_EXTRACT_RESIDUAL       =  3)
      parameter (PEP_EXTRACT_STRUCTURED     =  4)

      PetscEnum PEP_ERROR_ABSOLUTE
      PetscEnum PEP_ERROR_RELATIVE
      PetscEnum PEP_ERROR_BACKWARD

      parameter (PEP_ERROR_ABSOLUTE         =  0)
      parameter (PEP_ERROR_RELATIVE         =  1)
      parameter (PEP_ERROR_BACKWARD         =  2)

      PetscEnum PEP_CONV_ABS
      PetscEnum PEP_CONV_REL
      PetscEnum PEP_CONV_NORM
      PetscEnum PEP_CONV_USER

      parameter (PEP_CONV_ABS               =  0)
      parameter (PEP_CONV_REL               =  1)
      parameter (PEP_CONV_NORM              =  2)
      parameter (PEP_CONV_USER              =  3)

      PetscEnum PEP_STOP_BASIC
      PetscEnum PEP_STOP_USER

      parameter (PEP_STOP_BASIC             =  0)
      parameter (PEP_STOP_USER              =  1)

!
!   Possible arguments to PEPMonitorSet()
!
      external PEPMONITORALL
      external PEPMONITORLG
      external PEPMONITORLGALL
      external PEPMONITORCONVERGED
      external PEPMONITORFIRST

!
!  End of Fortran include file for the PEP package in SLEPc
!
