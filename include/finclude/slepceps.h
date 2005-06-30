
!
!  Include file for Fortran use of the EPS object in SLEPc
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
#define EPSLAPACK    'lapack'
#define EPSARPACK    'arpack'
#define EPSBLZPACK   'blzpack'
#define EPSPLANSO    'planso'
#define EPSTRLAN     'trlan'
#define EPSLOBPCG    'lobpcg'

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

      integer EPS_MGS_ORTH
      integer EPS_CGS_ORTH
      
      parameter (EPS_MGS_ORTH               =  0)
      parameter (EPS_CGS_ORTH               =  1)

      integer EPS_ORTH_REFINE_NEVER
      integer EPS_ORTH_REFINE_IFNEEDED
      integer EPS_ORTH_REFINE_ALWAYS 

      parameter (EPS_ORTH_REFINE_NEVER      =  0)  
      parameter (EPS_ORTH_REFINE_IFNEEDED   =  1)  
      parameter (EPS_ORTH_REFINE_ALWAYS     =  2)  
       
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

#endif
