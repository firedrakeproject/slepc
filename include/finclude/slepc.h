!
!  Include file for Fortran use of the SLEPc package

#include "petscconf.h"
#include "finclude/petscdef.h"

#if !defined (PETSC_AVOID_DECLARATIONS)
! ------------------------------------------------------------------------
!     BEGIN COMMON-BLOCK VARIABLES

!     Fortran Null
!
      character*(80)     SLEPC_NULL_CHARACTER
      PetscFortranInt    SLEPC_NULL_INTEGER
      PetscFortranDouble SLEPC_NULL_DOUBLE
      PetscScalar        SLEPC_NULL_SCALAR
!
!     A SLEPC_NULL_FUNCTION pointer
!
!      external SLEPC_NULL_FUNCTION
!
!     Common block to store some of the SLEPc constants,
!     which can be set only at runtime.
!     (A string should be in a different common block.)
!  
      common /slepcfortran1/ SLEPC_NULL_CHARACTER
      common /slepcfortran2/ SLEPC_NULL_INTEGER
      common /slepcfortran3/ SLEPC_NULL_SCALAR
      common /slepcfortran4/ SLEPC_NULL_DOUBLE

!     END COMMON-BLOCK VARIABLES
! ----------------------------------------------------------------------------
!
!  End of Fortran include file for the SLEPc package

#endif
