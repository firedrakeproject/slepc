!
!  - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
!  SLEPc - Scalable Library for Eigenvalue Problem Computations
!  Copyright (c) 2002-2019, Universitat Politecnica de Valencia, Spain
!
!  This file is part of SLEPc.
!  SLEPc is distributed under a 2-clause BSD license (see LICENSE).
!  - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
!
!  Include file for Fortran use of the BV object in SLEPc
!
#include "slepc/finclude/slepcbv.h"

      type tBV
        PetscFortranAddr:: v PETSC_FORTRAN_TYPE_INITIALIZE
      end type tBV

      BV, parameter :: SLEPC_NULL_BV = tBV(0)

      PetscEnum BV_ORTHOG_CGS
      PetscEnum BV_ORTHOG_MGS

      parameter (BV_ORTHOG_CGS             =  0)
      parameter (BV_ORTHOG_MGS             =  1)

      PetscEnum BV_ORTHOG_REFINE_IFNEEDED
      PetscEnum BV_ORTHOG_REFINE_NEVER
      PetscEnum BV_ORTHOG_REFINE_ALWAYS

      parameter (BV_ORTHOG_REFINE_IFNEEDED =  0)
      parameter (BV_ORTHOG_REFINE_NEVER    =  1)
      parameter (BV_ORTHOG_REFINE_ALWAYS   =  2)

      PetscEnum BV_ORTHOG_BLOCK_GS
      PetscEnum BV_ORTHOG_BLOCK_CHOL
      PetscEnum BV_ORTHOG_BLOCK_TSQR
      PetscEnum BV_ORTHOG_BLOCK_TSQRCHOL
      PetscEnum BV_ORTHOG_BLOCK_SVQB

      parameter (BV_ORTHOG_BLOCK_GS        =  0)
      parameter (BV_ORTHOG_BLOCK_CHOL      =  1)
      parameter (BV_ORTHOG_BLOCK_TSQR      =  2)
      parameter (BV_ORTHOG_BLOCK_TSQRCHOL  =  3)
      parameter (BV_ORTHOG_BLOCK_SVQB      =  4)

      PetscEnum BV_MATMULT_VECS
      PetscEnum BV_MATMULT_MAT
      PetscEnum BV_MATMULT_MAT_SAVE

      parameter (BV_MATMULT_VECS           =  0)
      parameter (BV_MATMULT_MAT            =  1)
      parameter (BV_MATMULT_MAT_SAVE       =  2)

!
!  End of Fortran include file for the BV package in SLEPc
!
