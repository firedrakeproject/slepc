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
#if !defined(__SLEPCEPS_H)
#define __SLEPCEPS_H

#define EPS                PetscFortranAddr
#define EPSType            character*(80)
#define EPSConvergedReason integer

#define EPSPOWER     'power'
#define EPSSUBSPACE  'subspace'
#define EPSARNOLDI   'arnoldi'
#define EPSLANCZOS   'lanczos'
#define EPSKRYLOVSCHUR 'krylovschur'
#define EPSLAPACK    'lapack'
#define EPSARPACK    'arpack'
#define EPSBLZPACK   'blzpack'
#define EPSTRLAN     'trlan'
#define EPSBLOPEX    'blopex'
#define EPSPRIMME    'primme'

!  Convergence flags.
!  They sould match the flags in $SLEPC_DIR/include/slepceps.h

      integer EPS_CONVERGED_TOL        
      integer EPS_DIVERGED_ITS
      integer EPS_DIVERGED_BREAKDOWN
      integer EPS_DIVERGED_NONSYMMETRIC
      integer EPS_CONVERGED_ITERATING

      parameter (EPS_CONVERGED_TOL          =  2)
      parameter (EPS_DIVERGED_ITS           = -3)
      parameter (EPS_DIVERGED_BREAKDOWN     = -4)
      parameter (EPS_DIVERGED_NONSYMMETRIC  = -5)
      parameter (EPS_CONVERGED_ITERATING    =  0)

      integer EPS_HEP
      integer EPS_GHEP
      integer EPS_NHEP
      integer EPS_GNHEP

      parameter (EPS_HEP                    =  1)
      parameter (EPS_GHEP                   =  2)
      parameter (EPS_NHEP                   =  3)
      parameter (EPS_GNHEP                  =  4)
      
      integer EPS_LARGEST_MAGNITUDE
      integer EPS_SMALLEST_MAGNITUDE
      integer EPS_LARGEST_REAL
      integer EPS_SMALLEST_REAL
      integer EPS_LARGEST_IMAGINARY
      integer EPS_SMALLEST_IMAGINARY

      parameter (EPS_LARGEST_MAGNITUDE      =  0)
      parameter (EPS_SMALLEST_MAGNITUDE     =  1)
      parameter (EPS_LARGEST_REAL           =  2)
      parameter (EPS_SMALLEST_REAL          =  3)
      parameter (EPS_LARGEST_IMAGINARY      =  4)
      parameter (EPS_SMALLEST_IMAGINARY     =  5)
       
      integer EPSPOWER_SHIFT_CONSTANT
      integer EPSPOWER_SHIFT_RAYLEIGH
      integer EPSPOWER_SHIFT_WILKINSON

      parameter (EPSPOWER_SHIFT_CONSTANT    =  0)
      parameter (EPSPOWER_SHIFT_RAYLEIGH    =  1)
      parameter (EPSPOWER_SHIFT_WILKINSON   =  2)

      integer EPS_ONE_SIDE
      integer EPS_TWO_SIDE

      parameter (EPS_ONE_SIDE               =  0)
      parameter (EPS_TWO_SIDE               =  1)
      
      integer EPS_RITZ
      integer EPS_HARMONIC
      integer EPS_REFINED
      integer EPS_REFINED_HARMONIC

      parameter (EPS_RITZ                   =  1)
      parameter (EPS_HARMONIC               =  2)
      parameter (EPS_REFINED                =  3)
      parameter (EPS_REFINED_HARMONIC       =  4)

      integer EPSLANCZOS_REORTHOG_NONE
      integer EPSLANCZOS_REORTHOG_FULL
      integer EPSLANCZOS_REORTHOG_SELECTIVE
      integer EPSLANCZOS_REORTHOG_PERIODIC
      integer EPSLANCZOS_REORTHOG_PARTIAL

      parameter (EPSLANCZOS_REORTHOG_NONE      =  0)
      parameter (EPSLANCZOS_REORTHOG_FULL      =  1)
      parameter (EPSLANCZOS_REORTHOG_SELECTIVE =  2)
      parameter (EPSLANCZOS_REORTHOG_PERIODIC  =  3)
      parameter (EPSLANCZOS_REORTHOG_PARTIAL   =  4)

      integer EPSPRIMME_DYNAMIC
      integer EPSPRIMME_DEFAULT_MIN_TIME
      integer EPSPRIMME_DEFAULT_MIN_MATVECS
      integer EPSPRIMME_ARNOLDI
      integer EPSPRIMME_GD
      integer EPSPRIMME_GD_PLUSK
      integer EPSPRIMME_GD_OLSEN_PLUSK
      integer EPSPRIMME_JD_OLSEN_PLUSK
      integer EPSPRIMME_RQI
      integer EPSPRIMME_JDQR
      integer EPSPRIMME_JDQMR
      integer EPSPRIMME_JDQMR_ETOL
      integer EPSPRIMME_SUBSPACE_ITERATION
      integer EPSPRIMME_LOBPCG_ORTHOBASIS
      integer EPSPRIMME_LOBPCG_ORTHOBASIS_WINDOW

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
      parameter (EPSPRIMME_LOBPCG_ORTHOBASIS_WINDOW = 15)

      integer EPSPRIMME_NONE
      integer EPSPRIMME_DIAGONAL

      parameter (EPSPRIMME_NONE               =  0)
      parameter (EPSPRIMME_DIAGONAL           =  1)

#endif
