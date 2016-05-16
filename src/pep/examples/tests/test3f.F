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
!  Description: Simple example to test the PEP Fortran interface.
!
! ---------------------------------------------------------------------- 
!
      program main
      implicit none

#include <petsc/finclude/petscsys.h>
#include <petsc/finclude/petscvec.h>
#include <petsc/finclude/petscmat.h>
#include <petsc/finclude/petscviewer.h>
#include <slepc/finclude/slepcsys.h>
#include <slepc/finclude/slepcpep.h>

! - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
!     Declarations
! - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
      Mat                A(3),B
      PEP                pep
      ST                 st
      KSP                ksp
      DS                 ds
      PetscReal          tol,tolabs,alpha,lambda
      PetscScalar        tget,val
      PetscInt           n,i,its,Istart,Iend
      PetscInt           nev,ncv,mpd,nmat,np
      PEPWhich           which
      PEPConvergedReason reason
      PEPType            tname
      PEPExtract         extr
      PEPBasis           basis
      PEPScale           scal
      PEPRefine          refine
      PEPRefineScheme    rscheme
      PEPConv            conv
      PEPStop            stp
      PEPProblemType     ptype
      PetscMPIInt        rank
      PetscErrorCode     ierr
      SlepcConvMonitor   ctx
      PetscViewerAndFormat vf

! - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
!     Beginning of program
! - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 

      call SlepcInitialize(PETSC_NULL_CHARACTER,ierr)
      call MPI_Comm_rank(PETSC_COMM_WORLD,rank,ierr)
      n = 20
      if (rank .eq. 0) then
        write(*,100) n
      endif
 100  format (/'Diagonal Quadratic Eigenproblem, n =',I3,' (Fortran)')

      call MatCreate(PETSC_COMM_WORLD,A(1),ierr)
      call MatSetSizes(A(1),PETSC_DECIDE,PETSC_DECIDE,n,n,ierr)
      call MatSetFromOptions(A(1),ierr)
      call MatSetUp(A(1),ierr)
      call MatGetOwnershipRange(A(1),Istart,Iend,ierr)
      do i=Istart,Iend-1
        val = i+1.
        call MatSetValue(A(1),i,i,val,INSERT_VALUES,ierr)
      enddo
      call MatAssemblyBegin(A(1),MAT_FINAL_ASSEMBLY,ierr)
      call MatAssemblyEnd(A(1),MAT_FINAL_ASSEMBLY,ierr)

      call MatCreate(PETSC_COMM_WORLD,A(2),ierr)
      call MatSetSizes(A(2),PETSC_DECIDE,PETSC_DECIDE,n,n,ierr)
      call MatSetFromOptions(A(2),ierr)
      call MatSetUp(A(2),ierr)
      call MatGetOwnershipRange(A(2),Istart,Iend,ierr)
      do i=Istart,Iend-1
        val = 1
        call MatSetValue(A(2),i,i,val,INSERT_VALUES,ierr)
      enddo
      call MatAssemblyBegin(A(2),MAT_FINAL_ASSEMBLY,ierr)
      call MatAssemblyEnd(A(2),MAT_FINAL_ASSEMBLY,ierr)

      call MatCreate(PETSC_COMM_WORLD,A(3),ierr)
      call MatSetSizes(A(3),PETSC_DECIDE,PETSC_DECIDE,n,n,ierr)
      call MatSetFromOptions(A(3),ierr)
      call MatSetUp(A(3),ierr)
      call MatGetOwnershipRange(A(3),Istart,Iend,ierr)
      do i=Istart,Iend-1
        val = n/(i+1.)
        call MatSetValue(A(3),i,i,val,INSERT_VALUES,ierr)
      enddo
      call MatAssemblyBegin(A(3),MAT_FINAL_ASSEMBLY,ierr)
      call MatAssemblyEnd(A(3),MAT_FINAL_ASSEMBLY,ierr)

! - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
!     Create eigensolver and test interface functions
! - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 

      call PEPCreate(PETSC_COMM_WORLD,pep,ierr)
      nmat = 3
      call PEPSetOperators(pep,nmat,A,ierr)
      call PEPGetNumMatrices(pep,nmat,ierr)
      if (rank .eq. 0) then
        write(*,110) nmat-1
      endif
 110  format (' Polynomial of degree ',I2)
      i = 0
      call PEPGetOperators(pep,i,B,ierr)
      call MatView(B,PETSC_NULL_OBJECT,ierr)

      call PEPSetType(pep,PEPTOAR,ierr)
      call PEPGetType(pep,tname,ierr)
      if (rank .eq. 0) then
        write(*,120) tname
      endif
 120  format (' Type set to ',A)

      call PEPGetProblemType(pep,ptype,ierr)
      if (rank .eq. 0) then
        write(*,130) ptype
      endif
 130  format (' Problem type before changing = ',I2)
      call PEPSetProblemType(pep,PEP_HERMITIAN,ierr)
      call PEPGetProblemType(pep,ptype,ierr)
      if (rank .eq. 0) then
        write(*,140) ptype
      endif
 140  format (' ... changed to ',I2)

      call PEPGetExtract(pep,extr,ierr)
      if (rank .eq. 0) then
        write(*,150) extr
      endif
 150  format (' Extraction before changing = ',I2)
      call PEPSetExtract(pep,PEP_EXTRACT_STRUCTURED,ierr)
      call PEPGetExtract(pep,extr,ierr)
      if (rank .eq. 0) then
        write(*,160) extr
      endif
 160  format (' ... changed to ',I2)

      alpha = .1
      its = 5
      lambda = 1.
      scal = PEP_SCALE_SCALAR
      call PEPSetScale(pep,scal,alpha,PETSC_NULL_OBJECT,                &
     &                 PETSC_NULL_OBJECT,its,lambda,ierr)
      call PEPGetScale(pep,scal,alpha,PETSC_NULL_OBJECT,                &
     &                 PETSC_NULL_OBJECT,its,lambda,ierr)
      if (rank .eq. 0) then
        write(*,170) scal,alpha,its
      endif
 170  format (' Scaling: ',I2,', alpha=',F6.4,', its=',I2)

      basis = PEP_BASIS_CHEBYSHEV1
      call PEPSetBasis(pep,basis,ierr)
      call PEPGetBasis(pep,basis,ierr)
      if (rank .eq. 0) then
        write(*,180) basis
      endif
 180  format (' Polynomial basis: ',I2)
 
      np = 1
      tol = 1e-9
      its = 2
      refine = PEP_REFINE_SIMPLE
      rscheme = PEP_REFINE_SCHEME_SCHUR
      call PEPSetRefine(pep,refine,np,tol,its,rscheme,ierr)
      call PEPGetRefine(pep,refine,np,tol,its,rscheme,ierr)
      if (rank .eq. 0) then
        write(*,190) refine,tol,its,rscheme
      endif
 190  format (' Refinement: ',I2,', tol=',F6.4,', its=',I2', schem=',I2)

      tget = 4.8
      call PEPSetTarget(pep,tget,ierr)
      call PEPGetTarget(pep,tget,ierr)
      call PEPSetWhichEigenpairs(pep,PEP_TARGET_MAGNITUDE,ierr)
      call PEPGetWhichEigenpairs(pep,which,ierr)
      if (rank .eq. 0) then
        write(*,200) which,PetscRealPart(tget)
      endif
 200  format (' Which = ',I2,', target = ',F3.1)

      nev = 4
      call PEPSetDimensions(pep,nev,PETSC_DEFAULT_INTEGER,              &
     &                      PETSC_DEFAULT_INTEGER,ierr)
      call PEPGetDimensions(pep,nev,ncv,mpd,ierr)
      if (rank .eq. 0) then
        write(*,210) nev,ncv,mpd
      endif
 210  format (' Dimensions: nev=',I2,', ncv=',I2,', mpd=',I2)

      tol = 2.2e-4
      its = 200
      call PEPSetTolerances(pep,tol,its,ierr)
      call PEPGetTolerances(pep,tol,its,ierr)
      if (rank .eq. 0) then
        write(*,220) tol,its
      endif
 220  format (' Tolerance =',F7.5,', max_its =',I4)

      call PEPSetConvergenceTest(pep,PEP_CONV_ABS,ierr)
      call PEPGetConvergenceTest(pep,conv,ierr)
      call PEPSetStoppingTest(pep,PEP_STOP_BASIC,ierr)
      call PEPGetStoppingTest(pep,stp,ierr)
      if (rank .eq. 0) then
        write(*,230) conv,stp
      endif
 230  format (' Convergence test =',I2,', stopping test =',I2)

      call PetscViewerAndFormatCreate(PETSC_VIEWER_STDOUT_WORLD,        &
     &                   PETSC_VIEWER_DEFAULT,vf,ierr)
      call PEPMonitorSet(pep,PEPMONITORFIRST,vf,                        &
     &                   PetscViewerAndFormatDestroy,ierr)
      call SlepcConvMonitorCreate(PETSC_VIEWER_STDOUT_WORLD,            &
     &                   PETSC_VIEWER_DEFAULT,ctx,ierr)
      call PEPMonitorSet(pep,PEPMONITORCONVERGED,ctx,                   &
     &                   SlepcConvMonitorDestroy,ierr)
      call PEPMonitorCancel(pep,ierr)

      call PEPGetST(pep,st,ierr) 
      call STGetKSP(st,ksp,ierr) 
      tol = 1.e-8
      tolabs = 1.e-35
      call KSPSetTolerances(ksp,tol,tolabs,PETSC_DEFAULT_REAL,          &
     &                      PETSC_DEFAULT_INTEGER,ierr)
      call STView(st,PETSC_NULL_OBJECT,ierr) 
      call PEPGetDS(pep,ds,ierr) 
      call DSView(ds,PETSC_NULL_OBJECT,ierr) 

      call PEPSetFromOptions(pep,ierr)
      call PEPSolve(pep,ierr) 
      call PEPGetConvergedReason(pep,reason,ierr)
      call PEPGetIterationNumber(pep,its,ierr)
      if (rank .eq. 0) then
        write(*,240) reason,its
      endif
 240  format (' Finished - converged reason =',I2,', its=',I4)

! - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
!     Display solution and clean up
! - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
      call PEPErrorView(pep,PEP_ERROR_RELATIVE,PETSC_NULL_OBJECT,ierr)
      call PEPDestroy(pep,ierr)
      call MatDestroy(A(1),ierr)
      call MatDestroy(A(2),ierr)
      call MatDestroy(A(3),ierr)

      call SlepcFinalize(ierr)
      end
