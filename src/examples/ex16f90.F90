!  - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
!  SLEPc - Scalable Library for Eigenvalue Problem Computations
!  Copyright (c) 2002-2010, Universidad Politecnica de Valencia, Spain
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
!  Program usage: mpirun -np n ex16f90 [-help] [-n <n>] [-m <m>] [SLEPc opts] 
!
!  Description: Simple example that solves a quadratic eigensystem with the
!  QEP object. This is the Fortran90 equivalent to ex16.c
!
!  The command line options are:
!    -n <n>, where <n> = number of grid subdivisions in x dimension
!    -m <m>, where <m> = number of grid subdivisions in y dimension
!
! ---------------------------------------------------------------------- 
!
      program main

#include "finclude/slepcqepdef.h"
      use slepcqep

      implicit none

! For usage without modules, uncomment the following lines and remove 
! the previous lines between 'program main' and 'implicit none'
!
!#include "finclude/petsc.h"
!#include "finclude/slepc.h"

! - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
!     Declarations
! - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
!
!  Variables:
!     M,C,K  problem matrices
!     solver quadratic eigenproblem solver context

#if defined(PETSC_USE_FORTRAN_DATATYPES)
      type(Mat)      M, C, K
      type(QEP)      solver
#else
      Mat            M, C, K
      QEP            solver
#endif
      QEPType        tname
      PetscReal      tol, error, re, im
      PetscScalar    kr, ki
      PetscInt       N, nx, ny, i, j, Istart, Iend, II
      PetscInt       nev, maxit, its, nconv
      PetscMPIInt    rank
      PetscErrorCode ierr
      PetscBool      flg

! - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
!     Beginning of program
! - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 

      call SlepcInitialize(PETSC_NULL_CHARACTER,ierr)
      call MPI_Comm_rank(PETSC_COMM_WORLD,rank,ierr)
      nx = 10
      call PetscOptionsGetInt(PETSC_NULL_CHARACTER,'-n',nx,flg,ierr)
      call PetscOptionsGetInt(PETSC_NULL_CHARACTER,'-m',ny,flg,ierr)
      if (.not. flg) then
        ny = nx
      endif
      N = nx*ny
      if (rank .eq. 0) then
        write(*,100) N, nx, ny
      endif
 100  format (/'Quadratic Eigenproblem, N=',I6,' (',I4,'x',I4,' grid)')

! - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
!     Compute the matrices that define the eigensystem, (k^2*K+k*X+M)x=0
! - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 

!     ** K is the 2-D Laplacian
      call MatCreate(PETSC_COMM_WORLD,K,ierr)
      call MatSetSizes(K,PETSC_DECIDE,PETSC_DECIDE,N,N,ierr)
      call MatSetFromOptions(K,ierr)
      call MatGetOwnershipRange(K,Istart,Iend,ierr)
      do II=Istart,Iend-1
        i = II/nx
        j = II-i*nx
        if (i .gt. 0) then 
          call MatSetValue(K,II,II-nx,-1.D0,INSERT_VALUES,ierr)
        endif
        if (i .lt. ny-1) then 
          call MatSetValue(K,II,II+nx,-1.D0,INSERT_VALUES,ierr)
        endif
        if (j .gt. 0) then 
          call MatSetValue(K,II,II-1,-1.D0,INSERT_VALUES,ierr)
        endif
        if (j .lt. nx-1) then 
          call MatSetValue(K,II,II+1,-1.D0,INSERT_VALUES,ierr)
        endif
        call MatSetValue(K,II,II,4.D0,INSERT_VALUES,ierr)
      end do
      call MatAssemblyBegin(K,MAT_FINAL_ASSEMBLY,ierr)
      call MatAssemblyEnd(K,MAT_FINAL_ASSEMBLY,ierr)

!     ** C is the zero matrix
      call MatCreate(PETSC_COMM_WORLD,C,ierr)
      call MatSetSizes(C,PETSC_DECIDE,PETSC_DECIDE,N,N,ierr)
      call MatSetFromOptions(C,ierr)
      call MatAssemblyBegin(C,MAT_FINAL_ASSEMBLY,ierr)
      call MatAssemblyEnd(C,MAT_FINAL_ASSEMBLY,ierr)

!     ** M is the identity matrix
      call MatCreate(PETSC_COMM_WORLD,M,ierr)
      call MatSetSizes(M,PETSC_DECIDE,PETSC_DECIDE,N,N,ierr)
      call MatSetFromOptions(M,ierr)
      call MatAssemblyBegin(M,MAT_FINAL_ASSEMBLY,ierr)
      call MatAssemblyEnd(M,MAT_FINAL_ASSEMBLY,ierr)
      call MatShift(M,1.D0,ierr)

! - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
!     Create the eigensolver and set various options
! - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 

!     ** Create eigensolver context
      call QEPCreate(PETSC_COMM_WORLD,solver,ierr)

!     ** Set matrices and problem type
      call QEPSetOperators(solver,M,C,K,ierr)
      call QEPSetProblemType(solver,QEP_GENERAL,ierr)

!     ** Set solver parameters at runtime
      call QEPSetFromOptions(solver,ierr)

! - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
!     Solve the eigensystem
! - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 

      call QEPSolve(solver,ierr) 
      call QEPGetIterationNumber(solver,its,ierr)
      if (rank .eq. 0) then
        write(*,110) its
      endif
 110  format (/' Number of iterations of the method:',I4)
  
!     ** Optional: Get some information from the solver and display it
      call QEPGetType(solver,tname,ierr)
      if (rank .eq. 0) then
        write(*,120) tname
      endif
 120  format (' Solution method: ',A)
      call QEPGetDimensions(solver,nev,PETSC_NULL_INTEGER,              &
     &                      PETSC_NULL_INTEGER,ierr)
      if (rank .eq. 0) then
        write(*,130) nev
      endif
 130  format (' Number of requested eigenvalues:',I4)
      call QEPGetTolerances(solver,tol,maxit,ierr)
      if (rank .eq. 0) then
        write(*,140) tol, maxit
      endif
 140  format (' Stopping condition: tol=',1P,E10.4,', maxit=',I4)

! - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
!     Display solution and clean up
! - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 

!     ** Get number of converged eigenpairs
      call QEPGetConverged(solver,nconv,ierr)
      if (rank .eq. 0) then
        write(*,150) nconv
      endif
 150  format (' Number of converged eigenpairs:',I4/)

!     ** Display eigenvalues and relative errors
      if (nconv.gt.0) then
        if (rank .eq. 0) then
          write(*,*) '         k          ||(k^2M+Ck+K)x||/||kx||'
          write(*,*) ' ----------------- -------------------------'
        endif
        do i=0,nconv-1
!         ** Get converged eigenpairs: i-th eigenvalue is stored in kr 
!         ** (real part) and ki (imaginary part)
          call QEPGetEigenpair(solver,i,kr,ki,PETSC_NULL_OBJECT,        &
     &                         PETSC_NULL_OBJECT,ierr)

!         ** Compute the relative error associated to each eigenpair
          call QEPComputeRelativeError(solver,i,error,ierr)
          if (rank .eq. 0) then
            if (ki.ne.0.D0) then
              write(*,'(1P,E11.4,E11.4,A,E12.4)') kr, ki, ' j ', error
            else
              write(*,'(1P,A,E12.4,A,E12.4)') '   ', kr, '      ', error
            endif
          endif

        enddo
        if (rank .eq. 0) then
          write(*,*)
        endif
      endif

!     ** Free work space
      call QEPDestroy(solver,ierr)
      call MatDestroy(K,ierr)
      call MatDestroy(C,ierr)
      call MatDestroy(M,ierr)
      call SlepcFinalize(ierr)
      end

