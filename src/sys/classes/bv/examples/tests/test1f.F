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
!  Program usage: mpiexec -n <np> ./test1f [-help]
!
!  Description: Simple example that tests BV interface functions.
!
! ----------------------------------------------------------------------
!
      program main
#include <slepc/finclude/slepcbv.h>
      use slepcbv
      implicit none

! - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
!     Declarations
! - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

#define KMAX 35

      Vec            t,v
      Mat            Q,M
      BV             X,Y;
      PetscMPIInt    rank
      PetscInt       i,j,n,k,l,izero,ione
      PetscScalar    qq(1),z(KMAX),val
      PetscScalar    one,mone,two,zero
      PetscOffset    iq
      PetscReal      nrm
      PetscBool      flg
      PetscErrorCode ierr

! - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
!     Beginning of program
! - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

      n = 10
      k = 5
      l = 3
      one = 1.0
      mone = -1.0
      two = 2.0
      zero = 0.0
      izero = 0
      ione = 1
      call SlepcInitialize(PETSC_NULL_CHARACTER,ierr)
      call MPI_Comm_rank(PETSC_COMM_WORLD,rank,ierr)
      call PetscOptionsGetInt(PETSC_NULL_OPTIONS,PETSC_NULL_CHARACTER,  &
     &                        '-n',n,flg,ierr)
      call PetscOptionsGetInt(PETSC_NULL_OPTIONS,PETSC_NULL_CHARACTER,  &
     &                        '-k',k,flg,ierr)
      call PetscOptionsGetInt(PETSC_NULL_OPTIONS,PETSC_NULL_CHARACTER,  &
     &                        '-l',l,flg,ierr)
      if (k .gt. KMAX) then
        if (rank .eq. 0) then
          write(*,100) KMAX,k
        endif
        SETERRQ(PETSC_COMM_SELF,1,' ')
      endif
 100  format (/'Program currently limited to k=',I2,', you set k=',I3)
      if (rank .eq. 0) then
        write(*,110) k,n
      endif
 110  format (/'Test BV with',I3,' columns of length',I3,' (Fortran)')

! - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
!     Initialize data
! - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

!     ** Create template vector
      call VecCreate(PETSC_COMM_WORLD,t,ierr)
      call VecSetSizes(t,PETSC_DECIDE,n,ierr)
      call VecSetFromOptions(t,ierr)

!     ** Create BV object X
      call BVCreate(PETSC_COMM_WORLD,X,ierr)
      call PetscObjectSetName(X,'X',ierr)
      call BVSetSizesFromVec(X,t,k,ierr)
      call BVSetFromOptions(X,ierr)

!     ** Fill X entries
      do j=0,k-1
        call BVGetColumn(X,j,v,ierr)
        call VecSet(v,zero,ierr)
        do i=0,3
          if (i+j<n) then
            val = 3*i+j-2
            call VecSetValue(v,i+j,val,INSERT_VALUES,ierr)
          end if
        end do
        call VecAssemblyBegin(v,ierr)
        call VecAssemblyEnd(v,ierr)
        call BVRestoreColumn(X,j,v,ierr)
      end do

!     ** Create BV object Y
      call BVCreate(PETSC_COMM_WORLD,Y,ierr)
      call PetscObjectSetName(Y,'Y',ierr)
      call BVSetSizesFromVec(Y,t,l,ierr)
      call BVSetFromOptions(Y,ierr)

!     ** Fill Y entries
      do j=0,l-1
        call BVGetColumn(Y,j,v,ierr)
        val = (j+1)/4.0
        call VecSet(v,val,ierr)
        call BVRestoreColumn(Y,j,v,ierr)
      end do

!     ** Create Mat
      call MatCreateSeqDense(PETSC_COMM_SELF,k,l,PETSC_NULL_SCALAR,     &
     &      Q,ierr)
      call PetscObjectSetName(Q,'Q',ierr)
      call MatDenseGetArray(Q,qq,iq,ierr)
      do i=0,k-1
        do j=0,l-1
          if (i<j) then
            qq(iq+1+i+j*k) = 2.0
          else
            qq(iq+1+i+j*k) = -0.5
          end if
        end do
      end do
      call MatDenseRestoreArray(Q,qq,iq,ierr)

! - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
!     Test several operations
! - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

!     ** Test BVMult
      call BVMult(Y,two,one,X,Q,ierr)

!     ** Test BVMultVec
      call BVGetColumn(Y,izero,v,ierr)
      z(1) = 2.0
      do i=2,k
        z(i) = -0.5*z(i-1)
      end do
      call BVMultVec(X,mone,one,v,z,ierr)
      call BVRestoreColumn(Y,izero,v,ierr)

!     ** Test BVDot
      call MatCreateSeqDense(PETSC_COMM_SELF,l,k,PETSC_NULL_SCALAR,     &
     &      M,ierr)
      call PetscObjectSetName(M,'M',ierr)
      call BVDot(X,Y,M,ierr)

!     ** Test BVDotVec
      call BVGetColumn(Y,izero,v,ierr)
      call BVDotVec(X,v,z,ierr)
      call BVRestoreColumn(Y,izero,v,ierr)

!     ** Test BVMultInPlace and BVScale
      call BVMultInPlace(X,Q,ione,l,ierr)
      call BVScale(X,two,ierr)

!     ** Test BVNorm
      call BVNormColumn(X,izero,NORM_2,nrm,ierr)
      if (rank .eq. 0) then
        write(*,120) nrm
      endif
 120  format ('2-Norm of X[0] = ',f8.4)
      call BVNorm(X,NORM_FROBENIUS,nrm,ierr)
      if (rank .eq. 0) then
        write(*,130) nrm
      endif
 130  format ('Frobenius Norm of X = ',f8.4)

!     *** Clean up
      call BVDestroy(X,ierr)
      call BVDestroy(Y,ierr)
      call VecDestroy(t,ierr)
      call MatDestroy(Q,ierr)
      call MatDestroy(M,ierr)
      call SlepcFinalize(ierr)
      end

