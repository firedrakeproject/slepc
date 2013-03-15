!
!  Include file for Fortran use of the IP object in SLEPc
!
!
!  - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
!  SLEPc - Scalable Library for Eigenvalue Problem Computations
!  Copyright (c) 2002-2012, Universitat Politecnica de Valencia, Spain
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

      PetscEnum IP_ORTHOG_MGS
      PetscEnum IP_ORTHOG_CGS

      parameter (IP_ORTHOG_MGS               =  0)
      parameter (IP_ORTHOG_CGS               =  1)

      PetscEnum IP_ORTHOG_REFINE_NEVER
      PetscEnum IP_ORTHOG_REFINE_IFNEEDED
      PetscEnum IP_ORTHOG_REFINE_ALWAYS 

      parameter (IP_ORTHOG_REFINE_NEVER      =  0)  
      parameter (IP_ORTHOG_REFINE_IFNEEDED   =  1)  
      parameter (IP_ORTHOG_REFINE_ALWAYS     =  2)  

!
!  End of Fortran include file for the IP package in SLEPc
!
