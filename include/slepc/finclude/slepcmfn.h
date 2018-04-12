!
!  - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
!  SLEPc - Scalable Library for Eigenvalue Problem Computations
!  Copyright (c) 2002-2018, Universitat Politecnica de Valencia, Spain
!
!  This file is part of SLEPc.
!  SLEPc is distributed under a 2-clause BSD license (see LICENSE).
!  - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
!
!  Include file for Fortran use of the MFN object in SLEPc
!
#if !defined(__SLEPCMFNDEF_H)
#define __SLEPCMFNDEF_H

#include "petsc/finclude/petscmat.h"
#include "slepc/finclude/slepcfn.h"
#include "slepc/finclude/slepcbv.h"

#define MFN type(tMFN)

#define MFNType            character*(80)
#define MFNConvergedReason PetscEnum

#define MFNKRYLOV      'krylov'
#define MFNEXPOKIT     'expokit'

#endif

