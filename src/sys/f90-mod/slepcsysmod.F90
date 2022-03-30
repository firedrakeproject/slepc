!
!  - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
!  SLEPc - Scalable Library for Eigenvalue Problem Computations
!  Copyright (c) 2002-, Universitat Politecnica de Valencia, Spain
!
!  This file is part of SLEPc.
!  SLEPc is distributed under a 2-clause BSD license (see LICENSE).
!  - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
!
        module slepcsysdef
        use petscmatdef
        use petscsys
#include <../src/sys/f90-mod/slepcsys.h>
        end module

        module slepcsys
        use,intrinsic :: iso_c_binding
        use slepcsysdef
#include <../src/sys/f90-mod/slepcsys.h90>
        interface
#include <../src/sys/f90-mod/ftn-auto-interfaces/slepcsys.h90>
        end interface
        interface SlepcInitialize
          module procedure SlepcInitializeWithHelp, SlepcInitializeNoHelp, SlepcInitializeNoArguments
        end interface
      contains
      subroutine SlepcInitializeWithHelp(filename,help,ierr)
          character(len=*)           :: filename
          character(len=*)           :: help
          PetscErrorCode             :: ierr

          if (filename .ne. PETSC_NULL_CHARACTER) then
             filename = trim(filename)
          endif
          call SlepcInitializeF(filename,help,PETSC_TRUE,ierr)
        end subroutine SlepcInitializeWithHelp

        subroutine SlepcInitializeNoHelp(filename,ierr)
          character(len=*)           :: filename
          PetscErrorCode             :: ierr

          if (filename .ne. PETSC_NULL_CHARACTER) then
             filename = trim(filename)
          endif
          call SlepcInitializeF(filename,PETSC_NULL_CHARACTER,PETSC_TRUE,ierr)
        end subroutine SlepcInitializeNoHelp

        subroutine SlepcInitializeNoArguments(ierr)
          PetscErrorCode             :: ierr

          call SlepcInitializeF(PETSC_NULL_CHARACTER,PETSC_NULL_CHARACTER,PETSC_FALSE,ierr)
        end subroutine SlepcInitializeNoArguments
        end module

