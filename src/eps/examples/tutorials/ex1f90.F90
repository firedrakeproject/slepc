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
!  Program usage: mpiexec -n <np> ./ex1f90 [-help] [-n <n>] [all SLEPc options] 
!
!  Description: Simple example that solves an eigensystem with the EPS object.
!  The standard symmetric eigenvalue problem to be solved corresponds to the 
!  Laplacian operator in 1 dimension. 
!
!  The command line options are:
!    -n <n>, where <n> = number of grid points = matrix size
!
! ---------------------------------------------------------------------- 
!
      program main

#include <slepc/finclude/slepcepsdef.h>
      use slepceps

      implicit none

! For usage without modules, uncomment the following lines and remove 
! the previous lines between 'program main' and 'implicit none'
!
!#include <petsc/finclude/petsc.h>
!#include <slepc/finclude/slepc.h>

! - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
!     Declarations
! - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
!
!  Variables:
!     A      operator matrix
!     eps    eigenproblem solver context

#if defined(PETSC_USE_FORTRAN_DATATYPES)
      type(Mat)      A
      type(EPS)      eps
#else
      Mat            A
      EPS            eps
#endif
      EPSType        tname
      PetscInt       n, i, Istart, Iend, one, two, three
      PetscInt       nev
      PetscInt       row(1), col(3)
      PetscMPIInt    rank
      PetscErrorCode ierr
      PetscBool      flg, terse
      PetscScalar    value(3)

! - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
!     Beginning of program
! - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 

      one = 1
      two = 2
      three = 3
      call SlepcInitialize(PETSC_NULL_CHARACTER,ierr)
      call MPI_Comm_rank(PETSC_COMM_WORLD,rank,ierr)
      n = 30
      call PetscOptionsGetInt(PETSC_NULL_OBJECT,PETSC_NULL_CHARACTER,   &
     &                        '-n',n,flg,ierr)

      if (rank .eq. 0) then
        write(*,100) n
      endif
 100  format (/'1-D Laplacian Eigenproblem, n =',I4,' (Fortran)')

! - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
!     Compute the operator matrix that defines the eigensystem, Ax=kx
! - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 

      call MatCreate(PETSC_COMM_WORLD,A,ierr)
      call MatSetSizes(A,PETSC_DECIDE,PETSC_DECIDE,n,n,ierr)
      call MatSetFromOptions(A,ierr)
      call MatSetUp(A,ierr)

      call MatGetOwnershipRange(A,Istart,Iend,ierr)
      if (Istart .eq. 0) then 
        row(1) = 0
        col(1) = 0
        col(2) = 1
        value(1) =  2.0
        value(2) = -1.0
        call MatSetValues(A,one,row,two,col,value,INSERT_VALUES,ierr)
        Istart = Istart+1
      endif
      if (Iend .eq. n) then 
        row(1) = n-1
        col(1) = n-2
        col(2) = n-1
        value(1) = -1.0
        value(2) =  2.0
        call MatSetValues(A,one,row,two,col,value,INSERT_VALUES,ierr)
        Iend = Iend-1
      endif
      value(1) = -1.0
      value(2) =  2.0
      value(3) = -1.0
      do i=Istart,Iend-1
        row(1) = i
        col(1) = i-1
        col(2) = i
        col(3) = i+1
        call MatSetValues(A,one,row,three,col,value,INSERT_VALUES,ierr)
      enddo

      call MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY,ierr)
      call MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY,ierr)

! - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
!     Create the eigensolver and display info
! - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 

!     ** Create eigensolver context
      call EPSCreate(PETSC_COMM_WORLD,eps,ierr)

!     ** Set operators. In this case, it is a standard eigenvalue problem
      call EPSSetOperators(eps,A,PETSC_NULL_OBJECT,ierr)
      call EPSSetProblemType(eps,EPS_HEP,ierr)

!     ** Set solver parameters at runtime
      call EPSSetFromOptions(eps,ierr)

! - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
!     Solve the eigensystem
! - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 

      call EPSSolve(eps,ierr) 

!     ** Optional: Get some information from the solver and display it
      call EPSGetType(eps,tname,ierr)
      if (rank .eq. 0) then
        write(*,120) tname
      endif
 120  format (' Solution method: ',A)
      call EPSGetDimensions(eps,nev,PETSC_NULL_INTEGER,                 &
     &                      PETSC_NULL_INTEGER,ierr)
      if (rank .eq. 0) then
        write(*,130) nev
      endif
 130  format (' Number of requested eigenvalues:',I4)

! - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
!     Display solution and clean up
! - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 

!     ** show detailed info unless -terse option is given by user
      call PetscOptionsHasName(PETSC_NULL_OBJECT,PETSC_NULL_CHARACTER,  &
     &                        '-terse',terse,ierr)
      if (terse) then
        call EPSErrorView(eps,EPS_ERROR_RELATIVE,PETSC_NULL_OBJECT,ierr)
      else
        call PetscViewerPushFormat(PETSC_VIEWER_STDOUT_WORLD,           &
     &                   PETSC_VIEWER_ASCII_INFO_DETAIL,ierr)
        call EPSReasonView(eps,PETSC_VIEWER_STDOUT_WORLD,ierr)
        call EPSErrorView(eps,EPS_ERROR_RELATIVE,                       &
     &                   PETSC_VIEWER_STDOUT_WORLD,ierr)
        call PetscViewerPopFormat(PETSC_VIEWER_STDOUT_WORLD,ierr)
      endif
      call EPSDestroy(eps,ierr)
      call MatDestroy(A,ierr)

      call SlepcFinalize(ierr)
      end

