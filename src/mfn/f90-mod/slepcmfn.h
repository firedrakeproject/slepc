!
!  - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
!  SLEPc - Scalable Library for Eigenvalue Problem Computations
!  Copyright (c) 2002-, Universitat Politecnica de Valencia, Spain
!
!  This file is part of SLEPc.
!  SLEPc is distributed under a 2-clause BSD license (see LICENSE).
!  - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
!
!  Used by slepcmfnmod.F90 to create Fortran module file
!
#include "slepc/finclude/slepcsys.h"
#include "slepc/finclude/slepcmfn.h"

      type, extends(tPetscObject) :: tMFN
      end type tMFN

      MFN, parameter :: SLEPC_NULL_MFN = tMFN(0)

      PetscEnum, parameter :: MFN_CONVERGED_TOL          =  1
      PetscEnum, parameter :: MFN_CONVERGED_ITS          =  2
      PetscEnum, parameter :: MFN_DIVERGED_ITS           = -1
      PetscEnum, parameter :: MFN_DIVERGED_BREAKDOWN     = -2
      PetscEnum, parameter :: MFN_CONVERGED_ITERATING    =  0

!
!   Possible arguments to MFNMonitorSet()
!
      external MFNMONITORDEFAULT

#if defined(_WIN32) && defined(PETSC_USE_SHARED_LIBRARIES)
!DEC$ ATTRIBUTES DLLEXPORT::SLEPC_NULL_MFN
#endif
