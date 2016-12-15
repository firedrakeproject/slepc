!
!  Include file for Fortran use of the LME object in SLEPc
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
#include "slepc/finclude/slepcsys.h"
#include "slepc/finclude/slepclme.h"

      type tLME
        PetscFortranAddr:: v
      end type tLME

      PetscEnum LME_CONVERGED_TOL
      PetscEnum LME_DIVERGED_ITS
      PetscEnum LME_DIVERGED_BREAKDOWN
      PetscEnum LME_CONVERGED_ITERATING

      parameter (LME_CONVERGED_TOL          =  1)
      parameter (LME_DIVERGED_ITS           = -1)
      parameter (LME_DIVERGED_BREAKDOWN     = -2)
      parameter (LME_CONVERGED_ITERATING    =  0)

      PetscEnum LME_LYAPUNOV
      PetscEnum LME_SYLVESTER
      PetscEnum LME_GEN_LYAPUNOV
      PetscEnum LME_GEN_SYLVESTER
      PetscEnum LME_DT_LYAPUNOV
      PetscEnum LME_STEIN

      parameter (LME_LYAPUNOV               =  0)
      parameter (LME_SYLVESTER              =  1)
      parameter (LME_GEN_LYAPUNOV           =  2)
      parameter (LME_GEN_SYLVESTER          =  3)
      parameter (LME_DT_LYAPUNOV            =  4)
      parameter (LME_STEIN                  =  5)

!
!   Possible arguments to LMEMonitorSet()
!
      external LMEMONITORDEFAULT
      external LMEMONITORLG

!
!  End of Fortran include file for the LME package in SLEPc
!
