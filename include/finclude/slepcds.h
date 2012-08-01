!
!  Include file for Fortran use of the DS object in SLEPc
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
#include "finclude/slepcdsdef.h"

      PetscEnum DS_STATE_RAW
      PetscEnum DS_STATE_INTERMEDIATE
      PetscEnum DS_STATE_CONDENSED
      PetscEnum DS_STATE_SORTED
      
      parameter (DS_STATE_RAW                =  0)
      parameter (DS_STATE_INTERMEDIATE       =  1)
      parameter (DS_STATE_CONDENSED          =  2)
      parameter (DS_STATE_SORTED             =  3)

      PetscEnum DS_MAT_A
      PetscEnum DS_MAT_B
      PetscEnum DS_MAT_C
      PetscEnum DS_MAT_Q
      PetscEnum DS_MAT_X
      PetscEnum DS_MAT_Y
      PetscEnum DS_MAT_U
      PetscEnum DS_MAT_VT

      parameter (DS_MAT_A                    =  0)  
      parameter (DS_MAT_B                    =  1)  
      parameter (DS_MAT_C                    =  2)  
      parameter (DS_MAT_Q                    =  3)  
      parameter (DS_MAT_X                    =  4)  
      parameter (DS_MAT_Y                    =  5)  
      parameter (DS_MAT_U                    =  6)  
      parameter (DS_MAT_VT                   =  7)  

!
!  End of Fortran include file for the DS package in SLEPc
!
