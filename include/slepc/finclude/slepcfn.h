!
!  Include file for Fortran use of the FN object in SLEPc
!
!
!  - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
!  SLEPc - Scalable Library for Eigenvalue Problem Computations
!  Copyright (c) 2002-2016, Universitat Politecnica de Valencia, Spain
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
#include "slepc/finclude/slepcfndef.h"

      PetscEnum FN_COMBINE_ADD
      PetscEnum FN_COMBINE_MULTIPLY
      PetscEnum FN_COMBINE_DIVIDE
      PetscEnum FN_COMBINE_COMPOSE

      parameter (FN_COMBINE_ADD           =  0)
      parameter (FN_COMBINE_MULTIPLY      =  1)
      parameter (FN_COMBINE_DIVIDE        =  2)
      parameter (FN_COMBINE_COMPOSE       =  3)

!
!  End of Fortran include file for the FN package in SLEPc
!
