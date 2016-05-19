!
!  Basic include file for Fortran use of the SLEPc package
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

#include "petscconf.h"
#include "petsc/finclude/petscdef.h"
#include "slepcversion.h"
#include "slepc/finclude/slepcsysdef.h"

#if !defined(PETSC_USE_FORTRAN_DATATYPES)
      external SlepcConvMonitorDestroy
#endif

! Default tolerance for the different solvers, depending on the precision

      PetscReal SLEPC_DEFAULT_TOL
#if defined(PETSC_USE_REAL_SINGLE)
      parameter(SLEPC_DEFAULT_TOL     =  1e-6)
#elif defined(PETSC_USE_REAL_DOUBLE)
      parameter(SLEPC_DEFAULT_TOL     =  1e-8)
#elif defined(PETSC_USE_REAL___FLOAT128)
      parameter(SLEPC_DEFAULT_TOL     = 1e-16)
#else
      parameter(SLEPC_DEFAULT_TOL     =  1e-7)
#endif

