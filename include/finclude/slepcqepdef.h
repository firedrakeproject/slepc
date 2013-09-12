!
!  Include file for Fortran use of the QEP object in SLEPc
!
!  - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
!  SLEPc - Scalable Library for Eigenvalue Problem Computations
!  Copyright (c) 2002-2013, Universitat Politecnica de Valencia, Spain
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
#if !defined(__SLEPCQEP_H)
#define __SLEPCQEP_H

#include "finclude/slepcipdef.h"
#include "finclude/slepcstdef.h"
#include "finclude/slepcdsdef.h"
#include "finclude/slepcepsdef.h"

#if !defined(PETSC_USE_FORTRAN_DATATYPES)
#define QEP                PetscFortranAddr
#endif

#define QEPType            character*(80)
#define QEPProblemType     PetscEnum
#define QEPWhich           PetscEnum
#define QEPConvergedReason PetscEnum

#define QEPLINEAR    'linear'
#define QEPQARNOLDI  'qarnoldi'
#define QEPQLANCZOS  'qlanczos'
#define QEPSTOAR     'stoar'

#endif
