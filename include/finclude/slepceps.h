!
!  Include file for Fortran use of the EPS object in SLEPc
!
!
!  - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
!  SLEPc - Scalable Library for Eigenvalue Problem Computations
!  Copyright (c) 2002-2009, Universidad Politecnica de Valencia, Spain
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
      PetscEnum EPS_PGNHEP

      parameter (EPS_HEP                    =  1)
      parameter (EPS_GHEP                   =  2)
      parameter (EPS_NHEP                   =  3)
      parameter (EPS_GNHEP                  =  4)
      parameter (EPS_PGNHEP                 =  5)
      
      PetscEnum EPS_LARGEST_MAGNITUDE
      PetscEnum EPS_SMALLEST_MAGNITUDE
      PetscEnum EPS_LARGEST_REAL
      PetscEnum EPS_SMALLEST_REAL
      PetscEnum EPS_LARGEST_IMAGINARY
      PetscEnum EPS_SMALLEST_IMAGINARY
      PetscEnum EPS_TARGET_MAGNITUDE
      PetscEnum EPS_TARGET_REAL
      PetscEnum EPS_TARGET_IMAGINARY
      PetscEnum EPS_USER

      parameter (EPS_LARGEST_MAGNITUDE      =  0)
      parameter (EPS_SMALLEST_MAGNITUDE     =  1)
      parameter (EPS_LARGEST_REAL           =  2)
      parameter (EPS_SMALLEST_REAL          =  3)
      parameter (EPS_LARGEST_IMAGINARY      =  4)
      parameter (EPS_SMALLEST_IMAGINARY     =  5)
      parameter (EPS_TARGET_MAGNITUDE       =  6)
      parameter (EPS_TARGET_REAL            =  7)
      parameter (EPS_TARGET_IMAGINARY       =  8)
      parameter (EPS_USER                   =  9)
       
      PetscEnum EPS_BALANCE_NONE
      PetscEnum EPS_BALANCE_ONESIDE
      PetscEnum EPS_BALANCE_TWOSIDE
      PetscEnum EPS_BALANCE_USER

      parameter (EPS_BALANCE_NONE           =  1)
      parameter (EPS_BALANCE_ONESIDE        =  2)
      parameter (EPS_BALANCE_TWOSIDE        =  3)
      parameter (EPS_BALANCE_USER           =  4)

      PetscEnum EPS_POWER_SHIFT_CONSTANT
      PetscEnum EPS_POWER_SHIFT_RAYLEIGH
      PetscEnum EPS_POWER_SHIFT_WILKINSON

      parameter (EPS_POWER_SHIFT_CONSTANT   =  0)
      parameter (EPS_POWER_SHIFT_RAYLEIGH   =  1)
      parameter (EPS_POWER_SHIFT_WILKINSON  =  2)

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

      PetscEnum EPS_LANCZOS_REORTHOG_LOCAL
      PetscEnum EPS_LANCZOS_REORTHOG_FULL
      PetscEnum EPS_LANCZOS_REORTHOG_SELECTIVE
      PetscEnum EPS_LANCZOS_REORTHOG_PERIODIC
      PetscEnum EPS_LANCZOS_REORTHOG_PARTIAL

      parameter (EPS_LANCZOS_REORTHOG_LOCAL     =  0)
      parameter (EPS_LANCZOS_REORTHOG_FULL      =  1)
      parameter (EPS_LANCZOS_REORTHOG_SELECTIVE =  2)
      parameter (EPS_LANCZOS_REORTHOG_PERIODIC  =  3)
      parameter (EPS_LANCZOS_REORTHOG_PARTIAL   =  4)

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

      PetscEnum EPS_PRIMME_PRECOND_NONE
      PetscEnum EPS_PRIMME_PRECOND_DIAGONAL

      parameter (EPS_PRIMME_PRECOND_NONE        =  0)
      parameter (EPS_PRIMME_PRECOND_DIAGONAL    =  1)

      external EPSMONITORDEFAULT
      external EPSMONITORLG
      external EPSMONITORCONVERGED
      external EPSMONITORFIRST

!PETSC_DEC_ATTRIBUTES(EPSMONITORDEFAULT,'_EPSMONITORDEFAULT')
!PETSC_DEC_ATTRIBUTES(EPSMONITORLG,'_EPSMONITORLG')
!PETSC_DEC_ATTRIBUTES(EPSMONITORCONVERGED,'_EPSMONITORCONVERGED')
!PETSC_DEC_ATTRIBUTES(EPSMONITORFIRST,'_EPSMONITORFIRST')
