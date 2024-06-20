!
!  - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
!  SLEPc - Scalable Library for Eigenvalue Problem Computations
!  Copyright (c) 2002-, Universitat Politecnica de Valencia, Spain
!
!  This file is part of SLEPc.
!  SLEPc is distributed under a 2-clause BSD license (see LICENSE).
!  - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
!
!  Used by slepclmemod.F90 to create Fortran module file
!
#include "slepc/finclude/slepcsys.h"
#include "slepc/finclude/slepclme.h"

      type, extends(tPetscObject) :: tLME
      end type tLME

      LME, parameter :: SLEPC_NULL_LME = tLME(0)

      PetscEnum, parameter :: LME_CONVERGED_TOL          =  1
      PetscEnum, parameter :: LME_DIVERGED_ITS           = -1
      PetscEnum, parameter :: LME_DIVERGED_BREAKDOWN     = -2
      PetscEnum, parameter :: LME_CONVERGED_ITERATING    =  0

      PetscEnum, parameter :: LME_LYAPUNOV               =  0
      PetscEnum, parameter :: LME_SYLVESTER              =  1
      PetscEnum, parameter :: LME_GEN_LYAPUNOV           =  2
      PetscEnum, parameter :: LME_GEN_SYLVESTER          =  3
      PetscEnum, parameter :: LME_DT_LYAPUNOV            =  4
      PetscEnum, parameter :: LME_STEIN                  =  5

!
!   Possible arguments to LMEMonitorSet()
!
      external LMEMONITORDEFAULT

#if defined(_WIN32) && defined(PETSC_USE_SHARED_LIBRARIES)
!DEC$ ATTRIBUTES DLLEXPORT::SLEPC_NULL_LME
#endif
