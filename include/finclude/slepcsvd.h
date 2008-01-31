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

      integer SVD_CONVERGED_TOL        
      integer SVD_DIVERGED_ITS
      integer SVD_DIVERGED_BREAKDOWN
      integer SVD_CONVERGED_ITERATING

      parameter (SVD_CONVERGED_TOL          =  2)
      parameter (SVD_DIVERGED_ITS           = -3)
      parameter (SVD_DIVERGED_BREAKDOWN     = -4)
      parameter (SVD_CONVERGED_ITERATING    =  0)

      integer SVD_TRANSPOSE_EXPLICIT
      integer SVD_TRANSPOSE_IMPLICIT 

      parameter (SVD_TRANSPOSE_EXPLICIT     =  0)
      parameter (SVD_TRANSPOSE_IMPLICIT     =  1)
      
      integer SVD_LARGEST
      integer SVD_SMALLEST

      parameter (SVD_LARGEST                =  0)
      parameter (SVD_SMALLEST               =  1)

#endif
