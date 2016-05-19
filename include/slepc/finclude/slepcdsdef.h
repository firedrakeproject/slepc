!
!  Include file for Fortran use of the DS object in SLEPc
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
#if !defined(__SLEPCDSDEF_H)
#define __SLEPCDSDEF_H

#if !defined(PETSC_USE_FORTRAN_DATATYPES)
#define DS PetscFortranAddr
#endif

#define DSType      character*(80)
#define DSStateType PetscEnum
#define DSMatType   PetscEnum

#define DSHEP       'hep'
#define DSNHEP      'nhep'
#define DSGHEP      'ghep'
#define DSGHIEP     'ghiep'
#define DSGNHEP     'gnhep'
#define DSSVD       'svd'
#define DSPEP       'pep'
#define DSNEP       'nep'

#endif
