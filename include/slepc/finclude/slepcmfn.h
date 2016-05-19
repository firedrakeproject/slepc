!
!  Include file for Fortran use of the MFN object in SLEPc
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
#include "slepc/finclude/slepcsysdef.h"
#include "slepc/finclude/slepcmfndef.h"

      PetscEnum MFN_CONVERGED_TOL
      PetscEnum MFN_CONVERGED_ITS
      PetscEnum MFN_DIVERGED_ITS
      PetscEnum MFN_DIVERGED_BREAKDOWN
      PetscEnum MFN_CONVERGED_ITERATING

      parameter (MFN_CONVERGED_TOL          =  2)
      parameter (MFN_CONVERGED_ITS          =  3)
      parameter (MFN_DIVERGED_ITS           = -3)
      parameter (MFN_DIVERGED_BREAKDOWN     = -4)
      parameter (MFN_CONVERGED_ITERATING    =  0)

!
!   Possible arguments to MFNMonitorSet()
!
      external MFNMONITORDEFAULT
      external MFNMONITORLG

!
!  End of Fortran include file for the MFN package in SLEPc
!
