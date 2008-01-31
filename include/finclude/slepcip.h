!
!  Include file for Fortran use of the IP object in SLEPc
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
#if !defined(__SLEPCIP_H)
#define __SLEPCIP_H

#define IP                PetscFortranAddr

      integer IP_MGS_ORTH
      integer IP_CGS_ORTH
      
      parameter (IP_MGS_ORTH               =  0)
      parameter (IP_CGS_ORTH               =  1)

      integer IP_ORTH_REFINE_NEVER
      integer IP_ORTH_REFINE_IFNEEDED
      integer IP_ORTH_REFINE_ALWAYS 

      parameter (IP_ORTH_REFINE_NEVER      =  0)  
      parameter (IP_ORTH_REFINE_IFNEEDED   =  1)  
      parameter (IP_ORTH_REFINE_ALWAYS     =  2)  

      integer IP_MGS_ORTH
      integer IP_CGS_ORTH

      parameter (IP_MGS_ORTH               =  0)
      parameter (IP_CGS_ORTH               =  1)

#endif
