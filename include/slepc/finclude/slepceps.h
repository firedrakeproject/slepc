!
!  Include file for Fortran use of the EPS object in SLEPc
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
#include "slepc/finclude/slepcepsdef.h"

      PetscEnum EPS_CONVERGED_TOL
      PetscEnum EPS_CONVERGED_USER
      PetscEnum EPS_DIVERGED_ITS
      PetscEnum EPS_DIVERGED_BREAKDOWN
      PetscEnum EPS_DIVERGED_SYMMETRY_LOST
      PetscEnum EPS_CONVERGED_ITERATING

      parameter (EPS_CONVERGED_TOL          =  1)
      parameter (EPS_CONVERGED_USER         =  2)
      parameter (EPS_DIVERGED_ITS           = -1)
      parameter (EPS_DIVERGED_BREAKDOWN     = -2)
      parameter (EPS_DIVERGED_SYMMETRY_LOST = -3)
      parameter (EPS_CONVERGED_ITERATING    =  0)

      PetscEnum EPS_HEP
      PetscEnum EPS_GHEP
      PetscEnum EPS_NHEP
      PetscEnum EPS_GNHEP
      PetscEnum EPS_PGNHEP
      PetscEnum EPS_GHIEP

      parameter (EPS_HEP                    =  1)
      parameter (EPS_GHEP                   =  2)
      parameter (EPS_NHEP                   =  3)
      parameter (EPS_GNHEP                  =  4)
      parameter (EPS_PGNHEP                 =  5)
      parameter (EPS_GHIEP                  =  6)

      PetscEnum EPS_LARGEST_MAGNITUDE
      PetscEnum EPS_SMALLEST_MAGNITUDE
      PetscEnum EPS_LARGEST_REAL
      PetscEnum EPS_SMALLEST_REAL
      PetscEnum EPS_LARGEST_IMAGINARY
      PetscEnum EPS_SMALLEST_IMAGINARY
      PetscEnum EPS_TARGET_MAGNITUDE
      PetscEnum EPS_TARGET_REAL
      PetscEnum EPS_TARGET_IMAGINARY
      PetscEnum EPS_ALL
      PetscEnum EPS_WHICH_USER

      parameter (EPS_LARGEST_MAGNITUDE      =  1)
      parameter (EPS_SMALLEST_MAGNITUDE     =  2)
      parameter (EPS_LARGEST_REAL           =  3)
      parameter (EPS_SMALLEST_REAL          =  4)
      parameter (EPS_LARGEST_IMAGINARY      =  5)
      parameter (EPS_SMALLEST_IMAGINARY     =  6)
      parameter (EPS_TARGET_MAGNITUDE       =  7)
      parameter (EPS_TARGET_REAL            =  8)
      parameter (EPS_TARGET_IMAGINARY       =  9)
      parameter (EPS_ALL                    = 10)
      parameter (EPS_WHICH_USER             = 11)

      PetscEnum EPS_BALANCE_NONE
      PetscEnum EPS_BALANCE_ONESIDE
      PetscEnum EPS_BALANCE_TWOSIDE
      PetscEnum EPS_BALANCE_USER

      parameter (EPS_BALANCE_NONE           =  0)
      parameter (EPS_BALANCE_ONESIDE        =  1)
      parameter (EPS_BALANCE_TWOSIDE        =  2)
      parameter (EPS_BALANCE_USER           =  3)

      PetscEnum EPS_RITZ
      PetscEnum EPS_HARMONIC
      PetscEnum EPS_HARMONIC_RELATIVE
      PetscEnum EPS_HARMONIC_RIGHT
      PetscEnum EPS_HARMONIC_LARGEST
      PetscEnum EPS_REFINED
      PetscEnum EPS_REFINED_HARMONIC

      parameter (EPS_RITZ                   =  0)
      parameter (EPS_HARMONIC               =  1)
      parameter (EPS_HARMONIC_RELATIVE      =  2)
      parameter (EPS_HARMONIC_RIGHT         =  3)
      parameter (EPS_HARMONIC_LARGEST       =  4)
      parameter (EPS_REFINED                =  5)
      parameter (EPS_REFINED_HARMONIC       =  6)

      PetscEnum EPS_ERROR_ABSOLUTE
      PetscEnum EPS_ERROR_RELATIVE
      PetscEnum EPS_ERROR_BACKWARD

      parameter (EPS_ERROR_ABSOLUTE         =  0)
      parameter (EPS_ERROR_RELATIVE         =  1)
      parameter (EPS_ERROR_BACKWARD         =  2)

      PetscEnum EPS_CONV_ABS
      PetscEnum EPS_CONV_REL
      PetscEnum EPS_CONV_NORM
      PetscEnum EPS_CONV_USER

      parameter (EPS_CONV_ABS               =  0)
      parameter (EPS_CONV_REL               =  1)
      parameter (EPS_CONV_NORM              =  2)
      parameter (EPS_CONV_USER              =  3)

      PetscEnum EPS_STOP_BASIC
      PetscEnum EPS_STOP_USER

      parameter (EPS_STOP_BASIC             =  0)
      parameter (EPS_STOP_USER              =  1)

      PetscEnum EPS_POWER_SHIFT_CONSTANT
      PetscEnum EPS_POWER_SHIFT_RAYLEIGH
      PetscEnum EPS_POWER_SHIFT_WILKINSON

      parameter (EPS_POWER_SHIFT_CONSTANT   =  0)
      parameter (EPS_POWER_SHIFT_RAYLEIGH   =  1)
      parameter (EPS_POWER_SHIFT_WILKINSON  =  2)

      PetscEnum EPS_LANCZOS_REORTHOG_LOCAL
      PetscEnum EPS_LANCZOS_REORTHOG_FULL
      PetscEnum EPS_LANCZOS_REORTHOG_SELECTIVE
      PetscEnum EPS_LANCZOS_REORTHOG_PERIODIC
      PetscEnum EPS_LANCZOS_REORTHOG_PARTIAL
      PetscEnum EPS_LANCZOS_REORTHOG_DELAYED

      parameter (EPS_LANCZOS_REORTHOG_LOCAL     =  0)
      parameter (EPS_LANCZOS_REORTHOG_FULL      =  1)
      parameter (EPS_LANCZOS_REORTHOG_SELECTIVE =  2)
      parameter (EPS_LANCZOS_REORTHOG_PERIODIC  =  3)
      parameter (EPS_LANCZOS_REORTHOG_PARTIAL   =  4)
      parameter (EPS_LANCZOS_REORTHOG_DELAYED   =  5)

      PetscEnum EPS_PRIMME_DYNAMIC
      PetscEnum EPS_PRIMME_DEFAULT_MIN_TIME
      PetscEnum EPS_PRIMME_DEFAULT_MIN_MATVECS
      PetscEnum EPS_PRIMME_ARNOLDI
      PetscEnum EPS_PRIMME_GD
      PetscEnum EPS_PRIMME_GD_PLUSK
      PetscEnum EPS_PRIMME_GD_OLSEN_PLUSK
      PetscEnum EPS_PRIMME_JD_OLSEN_PLUSK
      PetscEnum EPS_PRIMME_RQI
      PetscEnum EPS_PRIMME_JDQR
      PetscEnum EPS_PRIMME_JDQMR
      PetscEnum EPS_PRIMME_JDQMR_ETOL
      PetscEnum EPS_PRIMME_SUBSPACE_ITERATION
      PetscEnum EPS_PRIMME_LOBPCG_ORTHOBASIS
      PetscEnum EPS_PRIMME_LOBPCG_ORTHOBASISW

      parameter (EPS_PRIMME_DYNAMIC             =  0)
      parameter (EPS_PRIMME_DEFAULT_MIN_TIME    =  1)
      parameter (EPS_PRIMME_DEFAULT_MIN_MATVECS =  2)
      parameter (EPS_PRIMME_ARNOLDI             =  3)
      parameter (EPS_PRIMME_GD                  =  4)
      parameter (EPS_PRIMME_GD_PLUSK            =  5)
      parameter (EPS_PRIMME_GD_OLSEN_PLUSK      =  7)
      parameter (EPS_PRIMME_JD_OLSEN_PLUSK      =  8)
      parameter (EPS_PRIMME_RQI                 =  9)
      parameter (EPS_PRIMME_JDQR                = 10)
      parameter (EPS_PRIMME_JDQMR               = 11)
      parameter (EPS_PRIMME_JDQMR_ETOL          = 12)
      parameter (EPS_PRIMME_SUBSPACE_ITERATION  = 13)
      parameter (EPS_PRIMME_LOBPCG_ORTHOBASIS   = 14)
      parameter (EPS_PRIMME_LOBPCG_ORTHOBASISW  = 15)

      PetscEnum EPS_CISS_QUADRULE_TRAPEZOIDAL
      PetscEnum EPS_CISS_QUADRULE_CHEBYSHEV

      parameter (EPS_CISS_QUADRULE_TRAPEZOIDAL  =  1)
      parameter (EPS_CISS_QUADRULE_CHEBYSHEV    =  2)

      PetscEnum EPS_CISS_EXTRACTION_RITZ
      PetscEnum EPS_CISS_EXTRACTION_HANKEL

      parameter (EPS_CISS_EXTRACTION_RITZ       =  0)
      parameter (EPS_CISS_EXTRACTION_HANKEL     =  1)

!
!   Possible arguments to EPSMonitorSet()
!
      external EPSMONITORALL
      external EPSMONITORLG
      external EPSMONITORLGALL
      external EPSMONITORCONVERGED
      external EPSMONITORFIRST

!
!  End of Fortran include file for the EPS package in SLEPc
!
