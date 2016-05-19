!
!  Include file for Fortran use of the SVD object in SLEPc
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
#include "slepc/finclude/slepcsvddef.h"

      PetscEnum SVD_CONVERGED_TOL
      PetscEnum SVD_CONVERGED_USER
      PetscEnum SVD_DIVERGED_ITS
      PetscEnum SVD_DIVERGED_BREAKDOWN
      PetscEnum SVD_CONVERGED_ITERATING

      parameter (SVD_CONVERGED_TOL          =  1)
      parameter (SVD_CONVERGED_USER         =  2)
      parameter (SVD_DIVERGED_ITS           = -1)
      parameter (SVD_DIVERGED_BREAKDOWN     = -2)
      parameter (SVD_CONVERGED_ITERATING    =  0)

      integer SVD_LARGEST
      integer SVD_SMALLEST

      parameter (SVD_LARGEST                =  0)
      parameter (SVD_SMALLEST               =  1)

      PetscEnum SVD_ERROR_ABSOLUTE
      PetscEnum SVD_ERROR_RELATIVE

      parameter (SVD_ERROR_ABSOLUTE         =  0)
      parameter (SVD_ERROR_RELATIVE         =  1)

      PetscEnum SVD_CONV_ABS
      PetscEnum SVD_CONV_REL
      PetscEnum SVD_CONV_USER

      parameter (SVD_CONV_ABS               =  0)
      parameter (SVD_CONV_REL               =  1)
      parameter (SVD_CONV_USER              =  2)

      PetscEnum SVD_STOP_BASIC
      PetscEnum SVD_STOP_USER

      parameter (SVD_STOP_BASIC             =  0)
      parameter (SVD_STOP_USER              =  1)

!
!   Possible arguments to SVDMonitorSet()
!
      external SVDMONITORALL
      external SVDMONITORLG
      external SVDMONITORLGALL
      external SVDMONITORCONVERGED
      external SVDMONITORFIRST

!
!  End of Fortran include file for the SVD package in SLEPc
!
