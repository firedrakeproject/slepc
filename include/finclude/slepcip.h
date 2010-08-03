!
!  Include file for Fortran use of the IP object in SLEPc
!
!
!  - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
!  SLEPc - Scalable Library for Eigenvalue Problem Computations
!  Copyright (c) 2002-2010, Universidad Politecnica de Valencia, Spain
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
#include "finclude/slepcipdef.h"

      PetscEnum IP_ORTH_MGS
      PetscEnum IP_ORTH_CGS
      
      parameter (IP_ORTH_MGS               =  0)
      parameter (IP_ORTH_CGS               =  1)

      PetscEnum IP_ORTH_REFINE_NEVER
      PetscEnum IP_ORTH_REFINE_IFNEEDED
      PetscEnum IP_ORTH_REFINE_ALWAYS 

      parameter (IP_ORTH_REFINE_NEVER      =  0)  
      parameter (IP_ORTH_REFINE_IFNEEDED   =  1)  
      parameter (IP_ORTH_REFINE_ALWAYS     =  2)  

      PetscEnum IP_INNER_HERMITIAN
      PetscEnum IP_INNER_SYMMETRIC

      parameter (IP_INNER_HERMITIAN        =  0)
      parameter (IP_INNER_SYMMETRIC        =  1)
