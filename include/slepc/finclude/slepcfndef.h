!
!  Include file for Fortran use of the FN object in SLEPc
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
#if !defined(__SLEPCFNDEF_H)
#define __SLEPCFNDEF_H

#if !defined(PETSC_USE_FORTRAN_DATATYPES)
#define FN PetscFortranAddr
#endif

#define FNType        character*(80)
#define FNCombineType PetscEnum

#define FNCOMBINE  'combine'
#define FNRATIONAL 'rational'
#define FNEXP      'exp'
#define FNLOG      'log'
#define FNPHI      'phi'
#define FNSQRT     'sqrt'
#define FNINVSQRT  'invsqrt'

#endif
