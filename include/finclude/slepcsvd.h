!
!  Include file for Fortran use of the SVD object in SLEPc
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
#if !defined(__SLEPCSVD_H)
#define __SLEPCSVD_H

#define SVD                PetscFortranAddr
#define SVDType            character*(80)
#define SVDConvergedReason integer

#define SVDCROSS     'cross'
#define SVDCYCLIC    'cyclic'
#define SVDLAPACK    'lapack'
#define SVDLANCZOS   'lanczos'
#define SVDTRLANCZOS 'trlanczos'

!  Convergence flags.
!  They sould match the flags in $SLEPC_DIR/include/slepcsvd.h

      integer EPS_CONVERGED_TOL        
      integer EPS_DIVERGED_ITS
      integer EPS_DIVERGED_BREAKDOWN
      integer EPS_DIVERGED_NONSYMMETRIC
      integer EPS_CONVERGED_ITERATING

      parameter (EPS_CONVERGED_TOL          =  2)
      parameter (EPS_DIVERGED_ITS           = -3)
      parameter (EPS_DIVERGED_BREAKDOWN     = -4)
      parameter (EPS_CONVERGED_ITERATING    =  0)

      integer SVD_TRANSPOSE_EXPLICIT
      integer SVD_TRANSPOSE_IMPLICIT 

      parameter (SVD_TRANSPOSE_EXPLICIT     =  1)
      parameter (SVD_TRANSPOSE_IMPLICIT     =  2)
      
      integer SVD_LARGEST
      integer SVD_SMALLEST

      parameter (SVD_LARGEST                =  1)
      parameter (SVD_SMALLEST               =  2)

#endif
