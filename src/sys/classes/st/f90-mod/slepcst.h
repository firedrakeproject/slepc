!
!  - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
!  SLEPc - Scalable Library for Eigenvalue Problem Computations
!  Copyright (c) 2002-, Universitat Politecnica de Valencia, Spain
!
!  This file is part of SLEPc.
!  SLEPc is distributed under a 2-clause BSD license (see LICENSE).
!  - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
!
!  Used by slepcstmod.F90 to create Fortran module file
!
#include "slepc/finclude/slepcst.h"

      type, extends(tPetscObject) :: tST
      end type tST

      ST, parameter :: SLEPC_NULL_ST = tST(0)

      PetscEnum, parameter :: ST_MATMODE_COPY          =  0
      PetscEnum, parameter :: ST_MATMODE_INPLACE       =  1
      PetscEnum, parameter :: ST_MATMODE_SHELL         =  2

#if defined(_WIN32) && defined(PETSC_USE_SHARED_LIBRARIES)
!DEC$ ATTRIBUTES DLLEXPORT::SLEPC_NULL_ST
#endif
