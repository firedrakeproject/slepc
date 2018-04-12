!
!  - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
!  SLEPc - Scalable Library for Eigenvalue Problem Computations
!  Copyright (c) 2002-2018, Universitat Politecnica de Valencia, Spain
!
!  This file is part of SLEPc.
!  SLEPc is distributed under a 2-clause BSD license (see LICENSE).
!  - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
!
!  Basic include file for Fortran use of the SLEPc package
!
#if !defined(__SLEPCSYSDEF_H)
#define __SLEPCSYSDEF_H
#include "petscconf.h"
#include "petsc/finclude/petsc.h"
#include "slepcversion.h"

#define SlepcSC type(tSlepcSC)

#define SlepcConvMonitor PetscFortranAddr

#endif

