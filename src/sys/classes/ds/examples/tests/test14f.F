!  - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
!  SLEPc - Scalable Library for Eigenvalue Problem Computations
!  Copyright (c) 2002-2016, Universitat Politecnica de Valencia, Spain
!
!  This file is part of SLEPc.
!     
!  SLEPc is free software: you can redistribute it and/or modify it under  the
!  terms of version 3 of the GNU Lesser General Public License as published by
!  the Free Software Foundation.
!
!  SLEPc  is  distributed in the hope that it will be useful, but WITHOUT  ANY 
!  WARRANTY;  without even the implied warranty of MERCHANTABILITY or  FITNESS 
!  FOR  A  PARTICULAR PURPOSE. See the GNU Lesser General Public  License  for 
!  more details.
!
!  You  should have received a copy of the GNU Lesser General  Public  License
!  along with SLEPc. If not, see <http://www.gnu.org/licenses/>.
!  - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
!
!  Program usage: mpiexec -n <np> ./test14f [-help] [-n <n>] [all SLEPc options] 
!
!  Description: Simple example that tests solving a DSNHEP problem.
!
!  The command line options are:
!    -n <n>, where <n> = matrix size
!
! ---------------------------------------------------------------------- 
!
      program main
      implicit none

#include <petsc/finclude/petscsys.h>
#include <petsc/finclude/petscmat.h>
#include <slepc/finclude/slepcds.h>

! - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
!     Declarations
! - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
!
!  Variables:
!     A     problem matrix
!     ds    dense solver context

      Mat            A
      DS             ds
      PetscInt       n, i, ld, zero
      PetscMPIInt    rank
      PetscErrorCode ierr
      PetscBool      flg
      PetscScalar    aa(1), wr(100), wi(100)
      PetscReal      re, im
      PetscOffset    ia

! - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
!     Beginning of program
! - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 

      zero = 0
      call SlepcInitialize(PETSC_NULL_CHARACTER,ierr)
      call MPI_Comm_rank(PETSC_COMM_WORLD,rank,ierr)
      n = 10
      call PetscOptionsGetInt(PETSC_NULL_OBJECT,PETSC_NULL_CHARACTER,   &
     &                        '-n',n,flg,ierr)
      if (n .gt. 100) then
        if (rank .eq. 0) then
          write(*,100) n
        endif
        SETERRQ(PETSC_COMM_SELF,1,' ',ierr)
      endif
 100  format (/'Program currently limited to n=100, you set n=',I3)

      if (rank .eq. 0) then
        write(*,110) n
      endif
 110  format (/'Solve a Dense System of type NHEP, n =',I3,' (Fortran)')

! - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
!     Create DS object
! - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 

      call DSCreate(PETSC_COMM_WORLD,ds,ierr)
      call DSSetType(ds,DSNHEP,ierr)
      call DSSetFromOptions(ds,ierr)
      ld = n
      call DSAllocate(ds,ld,ierr)
      call DSSetDimensions(ds,n,zero,zero,zero,ierr)

! - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
!     Fill with Grcar matrix
! - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 

      call DSGetMat(ds,DS_MAT_A,A,ierr)
      call MatDenseGetArray(A,aa,ia,ierr)
      call FillUpMatrix(n,aa(ia+1))
      call MatDenseRestoreArray(A,aa,ia,ierr)
      call DSRestoreMat(ds,DS_MAT_A,A,ierr)
      call DSSetState(ds,DS_STATE_INTERMEDIATE,ierr)

! - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
!     Solve the problem and show eigenvalues
! - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 

      call DSSolve(ds,wr,wi,ierr) 
!     call DSSort(ds,wr,wi,PETSC_NULL_SCALAR,PETSC_NULL_SCALAR,         &
!    &            PETSC_NULL_INTEGER,ierr) 

      if (rank .eq. 0) then
        write(*,*) 'Computed eigenvalues ='
        do i=1,n
#if defined(PETSC_USE_COMPLEX)
          re = PetscRealPart(wr(i))
          im = PetscImaginaryPart(wr(i))
#else
          re = wr(i)
          im = wi(i)
#endif
          if (abs(im).lt.1.d-10) then
            write(*,120) re
          else
            write(*,130) re, im
          endif
        end do
      endif
 120  format ('  ',F8.5)
 130  format ('  ',F8.5,SP,F8.5,'i')

!     *** Clean up
      call DSDestroy(ds,ierr)
      call SlepcFinalize(ierr)
      end

! -----------------------------------------------------------------

      subroutine FillUpMatrix(n,X)
      PetscInt    n,i,j
      PetscScalar X(n,n)

      do i=2,n
        X(i,i-1) = -1.d0
      end do
      do j=0,3
        do i=1,n-j
          X(i,i+j) = 1.d0
        end do
      end do
      return
      end

