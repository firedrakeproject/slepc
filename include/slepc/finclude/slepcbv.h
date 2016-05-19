!
!  Include file for Fortran use of the BV object in SLEPc
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
#include "slepc/finclude/slepcbvdef.h"

      PetscEnum BV_ORTHOG_CGS
      PetscEnum BV_ORTHOG_MGS

      parameter (BV_ORTHOG_CGS             =  0)
      parameter (BV_ORTHOG_MGS             =  1)

      PetscEnum BV_ORTHOG_REFINE_IFNEEDED
      PetscEnum BV_ORTHOG_REFINE_NEVER
      PetscEnum BV_ORTHOG_REFINE_ALWAYS

      parameter (BV_ORTHOG_REFINE_IFNEEDED =  0)
      parameter (BV_ORTHOG_REFINE_NEVER    =  1)
      parameter (BV_ORTHOG_REFINE_ALWAYS   =  2)

      PetscEnum BV_ORTHOG_BLOCK_GS
      PetscEnum BV_ORTHOG_BLOCK_CHOL

      parameter (BV_ORTHOG_BLOCK_GS        =  0)
      parameter (BV_ORTHOG_BLOCK_CHOL      =  1)

      PetscEnum BV_MATMULT_VECS
      PetscEnum BV_MATMULT_MAT
      PetscEnum BV_MATMULT_MAT_SAVE

      parameter (BV_MATMULT_VECS           =  0)
      parameter (BV_MATMULT_MAT            =  1)
      parameter (BV_MATMULT_MAT_SAVE       =  2)

!
!  End of Fortran include file for the BV package in SLEPc
!
