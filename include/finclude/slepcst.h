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
#include "finclude/slepcstdef.h"

      PetscEnum STMATMODE_COPY
      PetscEnum STMATMODE_INPLACE
      PetscEnum STMATMODE_SHELL

      parameter (STMATMODE_COPY          =  0)
      parameter (STMATMODE_INPLACE       =  1)
      parameter (STMATMODE_SHELL         =  2)

