!
!  Include file for Fortran use of the SVD object in SLEPc
!
!
!  - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
!  SLEPc - Scalable Library for Eigenvalue Problem Computations
!  Copyright (c) 2002-2015, Universitat Politecnica de Valencia, Spain
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

!  Convergence flags.
!  They sould match the flags in $SLEPC_DIR/include/slepcsvd.h

      PetscEnum SVD_CONVERGED_TOL
      PetscEnum SVD_DIVERGED_ITS
      PetscEnum SVD_DIVERGED_BREAKDOWN
      PetscEnum SVD_CONVERGED_ITERATING

      parameter (SVD_CONVERGED_TOL          =  2)
      parameter (SVD_DIVERGED_ITS           = -3)
      parameter (SVD_DIVERGED_BREAKDOWN     = -4)
      parameter (SVD_CONVERGED_ITERATING    =  0)

      integer SVD_LARGEST
      integer SVD_SMALLEST

      parameter (SVD_LARGEST                =  0)
      parameter (SVD_SMALLEST               =  1)

      PetscEnum SVD_ERROR_ABSOLUTE
      PetscEnum SVD_ERROR_RELATIVE

      parameter (SVD_ERROR_ABSOLUTE         =  0)
      parameter (SVD_ERROR_RELATIVE         =  1)

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
