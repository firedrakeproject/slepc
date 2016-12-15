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
#if !defined(__SLEPCLMEDEF_H)
#define __SLEPCLMEDEF_H

#include "petsc/finclude/petscmat.h"
#include "slepc/finclude/slepcbv.h"

#define LME type(tLME)

#define LMEType            character*(80)
#define LMEConvergedReason PetscEnum
#define LMEProblemType     PetscEnum

#define LMEKRYLOV      'krylov'
#define LMEEKRYLOV     'ekrylov'

#endif
