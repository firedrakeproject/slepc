!
!  Include file for Fortran use of the NEP object in SLEPc
!
!  - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
!  SLEPc - Scalable Library for Eigenvalue Problem Computations
!  Copyright (c) 2002-2014, Universitat Politecnica de Valencia, Spain
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
#if !defined(__SLEPCNEP_H)
#define __SLEPCNEP_H

#include "finclude/slepcbvdef.h"
#include "finclude/slepcdsdef.h"
#include "finclude/slepcrgdef.h"
#include "finclude/slepcfndef.h"
#include "finclude/slepcepsdef.h"
#include "finclude/slepcpepdef.h"

#if !defined(PETSC_USE_FORTRAN_DATATYPES)
#define NEP                PetscFortranAddr
#endif

#define NEPType            character*(80)
#define NEPWhich           PetscEnum
#define NEPRefine          PetscEnum
#define NEPConvergedReason PetscEnum

#define NEPRII       'rii'
#define NEPSLP       'slp'
#define NEPNARNOLDI  'narnoldi'
#define NEPINTERPOL  'interpol'

#endif
