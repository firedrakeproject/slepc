!
!  Include file for Fortran use of the PEP object in SLEPc
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
#if !defined(__SLEPCPEPDEF_H)
#define __SLEPCPEPDEF_H

#include "slepc/finclude/slepcbvdef.h"
#include "slepc/finclude/slepcstdef.h"
#include "slepc/finclude/slepcdsdef.h"
#include "slepc/finclude/slepcrgdef.h"
#include "slepc/finclude/slepcepsdef.h"

#if !defined(PETSC_USE_FORTRAN_DATATYPES)
#define PEP PetscFortranAddr
#endif

#define PEPType            character*(80)
#define PEPProblemType     PetscEnum
#define PEPWhich           PetscEnum
#define PEPBasis           PetscEnum
#define PEPScale           PetscEnum
#define PEPRefine          PetscEnum
#define PEPRefineScheme    PetscEnum
#define PEPExtract         PetscEnum
#define PEPConv            PetscEnum
#define PEPStop            PetscEnum
#define PEPErrorType       PetscEnum
#define PEPConvergedReason PetscEnum

#define PEPLINEAR    'linear'
#define PEPQARNOLDI  'qarnoldi'
#define PEPTOAR      'toar'
#define PEPSTOAR     'stoar'
#define PEPJD        'jd'

#endif
