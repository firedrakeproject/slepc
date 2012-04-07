!
!  Include file for Fortran use of the PS object in SLEPc
!
!
!  - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
!  SLEPc - Scalable Library for Eigenvalue Problem Computations
!  Copyright (c) 2002-2011, Universitat Politecnica de Valencia, Spain
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
#include "finclude/slepcpsdef.h"

      PetscEnum PS_STATE_RAW
      PetscEnum PS_STATE_INTERMEDIATE
      PetscEnum PS_STATE_CONDENSED
      PetscEnum PS_STATE_SORTED
      
      parameter (PS_STATE_RAW                =  0)
      parameter (PS_STATE_INTERMEDIATE       =  1)
      parameter (PS_STATE_CONDENSED          =  2)
      parameter (PS_STATE_SORTED             =  3)

      PetscEnum PS_MAT_A
      PetscEnum PS_MAT_B
      PetscEnum PS_MAT_C
      PetscEnum PS_MAT_Q
      PetscEnum PS_MAT_X
      PetscEnum PS_MAT_Y
      PetscEnum PS_MAT_U
      PetscEnum PS_MAT_VT

      parameter (PS_MAT_A                    =  0)  
      parameter (PS_MAT_B                    =  1)  
      parameter (PS_MAT_C                    =  2)  
      parameter (PS_MAT_Q                    =  3)  
      parameter (PS_MAT_X                    =  4)  
      parameter (PS_MAT_Y                    =  5)  
      parameter (PS_MAT_U                    =  6)  
      parameter (PS_MAT_VT                   =  7)  

!
!  End of Fortran include file for the PS package in SLEPc
!
