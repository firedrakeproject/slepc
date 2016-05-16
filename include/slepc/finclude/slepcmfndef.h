!
!  Include file for Fortran use of the MFN object in SLEPc
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
#if !defined(__SLEPCMFNDEF_H)
#define __SLEPCMFNDEF_H

#include "petsc/finclude/petscmatdef.h"
#include "slepc/finclude/slepcfndef.h"
#include "slepc/finclude/slepcbvdef.h"

#if !defined(PETSC_USE_FORTRAN_DATATYPES)
#define MFN PetscFortranAddr
#endif

#define MFNType            character*(80)
#define MFNConvergedReason PetscEnum

#define MFNKRYLOV      'krylov'
#define MFNEXPOKIT     'expokit'

#endif
