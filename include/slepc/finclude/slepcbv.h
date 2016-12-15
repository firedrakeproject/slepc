!
!  Include file for Fortran use of the BV object in SLEPc
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
#if !defined(__SLEPCBVDEF_H)
#define __SLEPCBVDEF_H

#include "petsc/finclude/petscmat.h"

#define BV type(tBV)

#define BVType             character*(80)
#define BVOrthogType       PetscEnum
#define BVOrthogRefineType PetscEnum
#define BVOrthogBlockType  PetscEnum
#define BVMatMultType      PetscEnum

#define BVMAT        'mat'
#define BVSVEC       'svec'
#define BVVECS       'vecs'
#define BVCONTIGUOUS 'contiguous'

#endif
