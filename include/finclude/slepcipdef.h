!
!  Include file for Fortran use of the IP object in SLEPc
!
!
!  - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
!     SLEPc - Scalable Library for Eigenvalue Problem Computations
!     Copyright (c) 2002-2007, Universidad Politecnica de Valencia, Spain
!
!     This file is part of SLEPc. See the README file for conditions of use
!     and additional information.
!  - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
!
#if !defined(__SLEPCIP_H)
#define __SLEPCIP_H

#if !defined(PETSC_USE_FORTRAN_DATATYPES)
#define IP PetscFortranAddr
#endif

#define IPOrthogonalizationType PetscEnum
#define IPOrthogonalizationRefinementType PetscEnum
#define IPBilinearForm PetscEnum

#endif
