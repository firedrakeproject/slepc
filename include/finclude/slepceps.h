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
#include "finclude/slepcepsdef.h"

!  Convergence flags.
!  They sould match the flags in $SLEPC_DIR/include/slepceps.h

      PetscEnum EPS_CONVERGED_TOL        
      PetscEnum EPS_DIVERGED_ITS
      PetscEnum EPS_DIVERGED_BREAKDOWN
      PetscEnum EPS_DIVERGED_NONSYMMETRIC
      PetscEnum EPS_CONVERGED_ITERATING

      parameter (EPS_CONVERGED_TOL          =  2)
      parameter (EPS_DIVERGED_ITS           = -3)
      parameter (EPS_DIVERGED_BREAKDOWN     = -4)
      parameter (EPS_DIVERGED_NONSYMMETRIC  = -5)
      parameter (EPS_CONVERGED_ITERATING    =  0)

      PetscEnum EPS_HEP
      PetscEnum EPS_GHEP
      PetscEnum EPS_NHEP
      PetscEnum EPS_GNHEP

      parameter (EPS_HEP                    =  1)
      parameter (EPS_GHEP                   =  2)
      parameter (EPS_NHEP                   =  3)
      parameter (EPS_GNHEP                  =  4)
      
      PetscEnum EPS_LARGEST_MAGNITUDE
      PetscEnum EPS_SMALLEST_MAGNITUDE
      PetscEnum EPS_LARGEST_REAL
      PetscEnum EPS_SMALLEST_REAL
      PetscEnum EPS_LARGEST_IMAGINARY
      PetscEnum EPS_SMALLEST_IMAGINARY

      parameter (EPS_LARGEST_MAGNITUDE      =  0)
      parameter (EPS_SMALLEST_MAGNITUDE     =  1)
      parameter (EPS_LARGEST_REAL           =  2)
      parameter (EPS_SMALLEST_REAL          =  3)
      parameter (EPS_LARGEST_IMAGINARY      =  4)
      parameter (EPS_SMALLEST_IMAGINARY     =  5)
       
      PetscEnum EPSPOWER_SHIFT_CONSTANT
      PetscEnum EPSPOWER_SHIFT_RAYLEIGH
      PetscEnum EPSPOWER_SHIFT_WILKINSON

      parameter (EPSPOWER_SHIFT_CONSTANT    =  0)
      parameter (EPSPOWER_SHIFT_RAYLEIGH    =  1)
      parameter (EPSPOWER_SHIFT_WILKINSON   =  2)

      PetscEnum EPS_ONE_SIDE
      PetscEnum EPS_TWO_SIDE

      parameter (EPS_ONE_SIDE               =  0)
      parameter (EPS_TWO_SIDE               =  1)
      
      PetscEnum EPS_RITZ
      PetscEnum EPS_HARMONIC
      PetscEnum EPS_REFINED
      PetscEnum EPS_REFINED_HARMONIC

      parameter (EPS_RITZ                   =  1)
      parameter (EPS_HARMONIC               =  2)
      parameter (EPS_REFINED                =  3)
      parameter (EPS_REFINED_HARMONIC       =  4)

      PetscEnum EPSLANCZOS_REORTHOG_LOCAL
      PetscEnum EPSLANCZOS_REORTHOG_FULL
      PetscEnum EPSLANCZOS_REORTHOG_SELECTIVE
      PetscEnum EPSLANCZOS_REORTHOG_PERIODIC
      PetscEnum EPSLANCZOS_REORTHOG_PARTIAL

      parameter (EPSLANCZOS_REORTHOG_LOCAL     =  0)
      parameter (EPSLANCZOS_REORTHOG_FULL      =  1)
      parameter (EPSLANCZOS_REORTHOG_SELECTIVE =  2)
      parameter (EPSLANCZOS_REORTHOG_PERIODIC  =  3)
      parameter (EPSLANCZOS_REORTHOG_PARTIAL   =  4)

      PetscEnum EPSPRIMME_DYNAMIC
      PetscEnum EPSPRIMME_DEFAULT_MIN_TIME
      PetscEnum EPSPRIMME_DEFAULT_MIN_MATVECS
      PetscEnum EPSPRIMME_ARNOLDI
      PetscEnum EPSPRIMME_GD
      PetscEnum EPSPRIMME_GD_PLUSK
      PetscEnum EPSPRIMME_GD_OLSEN_PLUSK
      PetscEnum EPSPRIMME_JD_OLSEN_PLUSK
      PetscEnum EPSPRIMME_RQI
      PetscEnum EPSPRIMME_JDQR
      PetscEnum EPSPRIMME_JDQMR
      PetscEnum EPSPRIMME_JDQMR_ETOL
      PetscEnum EPSPRIMME_SUBSPACE_ITERATION
      PetscEnum EPSPRIMME_LOBPCG_ORTHOBASIS
      PetscEnum EPSPRIMME_LOBPCG_ORTHOBASISW

      parameter (EPSPRIMME_DYNAMIC                  =  0)
      parameter (EPSPRIMME_DEFAULT_MIN_TIME         =  1)
      parameter (EPSPRIMME_DEFAULT_MIN_MATVECS      =  2)
      parameter (EPSPRIMME_ARNOLDI                  =  3)
      parameter (EPSPRIMME_GD                       =  4)
      parameter (EPSPRIMME_GD_PLUSK                 =  5)
      parameter (EPSPRIMME_GD_OLSEN_PLUSK           =  7)
      parameter (EPSPRIMME_JD_OLSEN_PLUSK           =  8)
      parameter (EPSPRIMME_RQI                      =  9)
      parameter (EPSPRIMME_JDQR                     = 10)
      parameter (EPSPRIMME_JDQMR                    = 11)
      parameter (EPSPRIMME_JDQMR_ETOL               = 12)
      parameter (EPSPRIMME_SUBSPACE_ITERATION       = 13)
      parameter (EPSPRIMME_LOBPCG_ORTHOBASIS        = 14)
      parameter (EPSPRIMME_LOBPCG_ORTHOBASISW       = 15)

      PetscEnum EPSPRIMME_NONE
      PetscEnum EPSPRIMME_DIAGONAL

      parameter (EPSPRIMME_NONE               =  0)
      parameter (EPSPRIMME_DIAGONAL           =  1)
