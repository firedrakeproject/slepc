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
#include "finclude/slepcipdef.h"

      PetscEnum IP_MGS_ORTH
      PetscEnum IP_CGS_ORTH
      
      parameter (IP_MGS_ORTH               =  0)
      parameter (IP_CGS_ORTH               =  1)

      PetscEnum IP_ORTH_REFINE_NEVER
      PetscEnum IP_ORTH_REFINE_IFNEEDED
      PetscEnum IP_ORTH_REFINE_ALWAYS 

      parameter (IP_ORTH_REFINE_NEVER      =  0)  
      parameter (IP_ORTH_REFINE_IFNEEDED   =  1)  
      parameter (IP_ORTH_REFINE_ALWAYS     =  2)  

      PetscEnum IPINNER_HERMITIAN
      PetscEnum IPINNER_SYMMETRIC

      parameter (IPINNER_HERMITIAN         =  0)
      parameter (IPINNER_SYMMETRIC         =  1)
