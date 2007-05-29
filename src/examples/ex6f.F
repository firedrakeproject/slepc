!
!  Program usage: mpirun -np n ex6f [-help] [-m <m>] [all SLEPc options] 
!
!  Description: This example solves the eigensystem arising in the Ising
!  model for ferromagnetic materials. The file mvmisg.f must be linked
!  together. Information about the model can be found at the following 
!  site http://math.nist.gov/MatrixMarket/data/NEP
!
!  The command line options are:
!    -m <m>, where <m> is the number of 2x2 blocks, i.e. matrix size N=2*m
!
! ---------------------------------------------------------------------- 
!
      program main
      implicit none

#include "finclude/petsc.h"
#include "finclude/petscvec.h"
#include "finclude/petscmat.h"
#include "finclude/slepc.h"
#include "finclude/slepceps.h"

! - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
!     Declarations
! - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
!
!  Variables:
!     A     operator matrix
!     eps   eigenproblem solver context

      Mat          A
      EPS          eps
      EPSType      type
      PetscReal    tol, error
      PetscScalar  kr, ki
      integer      size, rank, N, m, nev, ierr, maxit, i, its, nconv
      PetscTruth   flg

!     This is the routine to use for matrix-free approach
!
      external MatIsing_Mult

! - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
!     Beginning of program
! - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 

      call SlepcInitialize(PETSC_NULL_CHARACTER,ierr)
#if defined(PETSC_USE_COMPLEX)
      write(*,*) 'This example requires real numbers.'
      goto 999
#endif
      call MPI_Comm_size(PETSC_COMM_WORLD,size,ierr)
      call MPI_Comm_rank(PETSC_COMM_WORLD,rank,ierr)
      if (size .ne. 1) then
         if (rank .eq. 0) then
            write(*,*) 'This is a uniprocessor example only!'
         endif
         SETERRQ(1,' ',ierr)
      endif
      m = 30
      call PetscOptionsGetInt(PETSC_NULL_CHARACTER,'-m',m,flg,ierr)
      N = 2*m

      if (rank .eq. 0) then
        write(*,*)
        write(*,'(A,I6,A)') 'Ising Model Eigenproblem, m=',m,', (N=2*m)'
        write(*,*)
      endif

! - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
!     Register the matrix-vector subroutine for the operator that defines
!     the eigensystem, Ax=kx
! - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 

      call MatCreateShell(PETSC_COMM_WORLD,N,N,N,N,PETSC_NULL_OBJECT,A,
     &                    ierr)
      call MatShellSetOperation(A,MATOP_MULT,MatIsing_Mult,ierr)

! - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
!     Create the eigensolver and display info
! - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 

!     ** Create eigensolver context
      call EPSCreate(PETSC_COMM_WORLD,eps,ierr)

!     ** Set operators. In this case, it is a standard eigenvalue problem
      call EPSSetOperators(eps,A,PETSC_NULL_OBJECT,ierr)
      call EPSSetProblemType(eps,EPS_NHEP,ierr)

!     ** Set solver parameters at runtime
      call EPSSetFromOptions(eps,ierr)

! - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
!     Solve the eigensystem
! - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 

      call EPSSolve(eps,ierr) 
      call EPSGetIterationNumber(eps,its,ierr)
      if (rank .eq. 0) then
        write(*,'(A,I4)') ' Number of iterations of the method: ', its
      endif

!     ** Optional: Get some information from the solver and display it
      call EPSGetType(eps,type,ierr)
      if (rank .eq. 0) then
        write(*,'(A,A)') ' Solution method: ', type
      endif
      call EPSGetDimensions(eps,nev,PETSC_NULL_INTEGER,ierr)
      if (rank .eq. 0) then
        write(*,'(A,I2)') ' Number of requested eigenvalues:', nev
      endif
      call EPSGetTolerances(eps,tol,maxit,ierr)
      if (rank .eq. 0) then
        write(*,'(A,1PE10.4,A,I6)') ' Stopping condition: tol=', tol, 
     &                              ', maxit=', maxit
      endif

! - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
!     Display solution and clean up
! - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 

!     ** Get number of converged eigenpairs 
      call EPSGetConverged(eps,nconv,ierr)
      if (rank .eq. 0) then
        write(*,'(A,I2)') ' Number of converged eigenpairs:', nconv
      endif

!     ** Display eigenvalues and relative errors
      if (nconv.gt.0 .and. rank.eq.0) then
        write(*,*)
        write(*,*) '           k          ||Ax-kx||/||kx||'
        write(*,*) '   ----------------- ------------------'
        do i=0,nconv-1
!         ** Get converged eigenpairs: i-th eigenvalue is stored in kr 
!         ** (real part) and ki (imaginary part)
          call EPSGetEigenpair(eps,i,kr,ki,PETSC_NULL,PETSC_NULL,ierr)

!         ** Compute the relative error associated to each eigenpair
          call EPSComputeRelativeError(eps,i,error,ierr)

          if (ki.ne.0.D0) then
            write(*,'(1P,E11.4,E11.4,A,E12.4)') kr, ki, ' j ', error
          else
            write(*,'(1P,A,E12.4,A,E12.4)') '   ', kr, '       ', error
          endif
        enddo
      endif
      write(*,*)

!     ** Free work space
      call EPSDestroy(eps,ierr)
      call MatDestroy(A,ierr)

#if defined(PETSC_USE_COMPLEX)
 999  continue
#endif
      call SlepcFinalize(ierr)
      end

! ------------------------------------------------------------------- 
! 
!   MatIsing_Mult - user provided matrix-vector multiply 
!
!   Input Parameters:
!   A - matrix
!   x - input vector
!
!   Output Parameter:
!   y - output vector
! 
      subroutine MatIsing_Mult(A,x,y,ierr)
      implicit none

#include "finclude/petsc.h"

      Mat         A
      Vec         x,y
      integer     trans,one,ierr,N
      PetscScalar x_array(1),y_array(1)
      PetscOffset i_x,i_y

!     The actual routine for the matrix-vector product
      external mvmisg

      call MatGetSize(A,N,PETSC_NULL_INTEGER,ierr)
      call VecGetArray(x,x_array,i_x,ierr)
      call VecGetArray(y,y_array,i_y,ierr)

      trans = 0
      one = 1
      call mvmisg(trans,N,one,x_array(i_x+1),N,y_array(i_y+1),N)

      call VecRestoreArray(x,x_array,i_x,ierr)
      call VecRestoreArray(y,y_array,i_y,ierr)

      return
      end

