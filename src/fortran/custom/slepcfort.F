!
!  Utility routine used to set some values into a Fortran common block

      subroutine SlepcSetCommonBlock()
      implicit none

#include "include/finclude/petsc.h"
#include "include/finclude/slepc.h"

      SLEPC_NULL_CHARACTER = PETSC_NULL_CHARACTER
      SLEPC_NULL_INTEGER   = PETSC_NULL_INTEGER
      SLEPC_NULL_SCALAR    = PETSC_NULL_SCALAR
      SLEPC_NULL_DOUBLE    = PETSC_NULL_DOUBLE

      return
      end

