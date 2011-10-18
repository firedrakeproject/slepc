!
!  Include file for Fortran use of the QEP object in SLEPc
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
#include "finclude/slepcqepdef.h"

!  Convergence flags.
!  They should match the flags in $SLEPC_DIR/include/slepcqep.h

      PetscEnum QEP_CONVERGED_TOL        
      PetscEnum QEP_DIVERGED_ITS
      PetscEnum QEP_DIVERGED_BREAKDOWN
      PetscEnum QEP_CONVERGED_ITERATING

      parameter (QEP_CONVERGED_TOL          =  2)
      parameter (QEP_DIVERGED_ITS           = -3)
      parameter (QEP_DIVERGED_BREAKDOWN     = -4)
      parameter (QEP_CONVERGED_ITERATING    =  0)

      PetscEnum QEP_GENERAL
      PetscEnum QEP_HERMITIAN
      PetscEnum QEP_GYROSCOPIC

      parameter (QEP_GENERAL                =  1)
      parameter (QEP_HERMITIAN              =  2)
      parameter (QEP_GYROSCOPIC             =  3)
      
      PetscEnum QEP_LARGEST_MAGNITUDE
      PetscEnum QEP_SMALLEST_MAGNITUDE
      PetscEnum QEP_LARGEST_REAL
      PetscEnum QEP_SMALLEST_REAL
      PetscEnum QEP_LARGEST_IMAGINARY
      PetscEnum QEP_SMALLEST_IMAGINARY

      parameter (QEP_LARGEST_MAGNITUDE      =  1)
      parameter (QEP_SMALLEST_MAGNITUDE     =  2)
      parameter (QEP_LARGEST_REAL           =  3)
      parameter (QEP_SMALLEST_REAL          =  4)
      parameter (QEP_LARGEST_IMAGINARY      =  5)
      parameter (QEP_SMALLEST_IMAGINARY     =  6)
       
      external QEPMONITORALL
      external QEPMONITORLG
      external QEPMONITORLGALL
      external QEPMONITORCONVERGED
      external QEPMONITORFIRST
