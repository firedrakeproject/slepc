
!
!  Include file for Fortran use of the EPS object in SLEPc
!
#if !defined(__SLEPCEPS_H)
#define __SLEPCEPS_H

#define EPS                PetscFortranAddr
#define EPSType            character*(80)
#define EPSConvergedReason integer

#define EPSPOWER     'power'
#define EPSRQI       'rqi'
#define EPSSUBSPACE  'subspace'
#define EPSARNOLDI   'arnoldi'
#define EPSLAPACK    'lapack'
#define EPSARPACK    'arpack'
#define EPSBLZPACK   'blzpack'
#define EPSPLANSO    'planso'
#define EPSTRLAN     'trlan'

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

#endif
