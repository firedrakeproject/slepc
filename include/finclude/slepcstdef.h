!
!  Include file for Fortran use of the ST object in SLEPc
!
!  - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
!     SLEPc - Scalable Library for Eigenvalue Problem Computations
!     Copyright (c) 2002-2007, Universidad Politecnica de Valencia, Spain
!
!     This file is part of SLEPc. See the README file for conditions of use
!     and additional information.
!  - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
!
#if !defined(__SLEPCST_H)
#define __SLEPCST_H

#include "finclude/petsckspdef.h"

#if !defined(PETSC_USE_FORTRAN_DATATYPES)
#define ST        PetscFortranAddr
#endif

#define STType    character*(80)
#define STMatMode PetscEnum

#define STSHELL     'shell'
#define STSHIFT     'shift'
#define STSINV      'sinvert'
#define STCAYLEY    'cayley'
#define STFOLD      'fold'

#endif
