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
!/*T
!  Concepts: SLEPc - Basic functionality
!  Routines: SlepcInitialize(); SlepcFinalize();
!  Routines: EPSCreate(); EPSSetFromOptions();
!  Routines: EPSSolve(); EPSDestroy();
!T*/ 
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
!     x     basis vectors
!     eps   eigenproblem solver context

#define MAXNEV 30
      Mat          A
      Vec          x(MAXNEV)
      EPS          eps
      EPSType      type
      PetscReal    tol, error(MAXNEV)
      PetscScalar  kr(MAXNEV), ki(MAXNEV)
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
      write(6,*) "This example requires real numbers."
      goto 999
#endif
      call MPI_Comm_size(PETSC_COMM_WORLD,size,ierr)
      call MPI_Comm_rank(PETSC_COMM_WORLD,rank,ierr)
      if (size .ne. 1) then
         if (rank .eq. 0) then
            write(6,*) 'This is a uniprocessor example only!'
         endif
         SETERRQ(1,' ',ierr)
      endif
      m = 30
      call PetscOptionsGetInt(PETSC_NULL_CHARACTER,'-m',m,flg,ierr)
      N = 2*m

      if (rank .eq. 0) then
        write(6,*)
        write(6,100) m
        write(6,*)
      endif
 100  format ('Ising Model Eigenproblem, m =',i6,', (N=2*m)' )

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

!     ** Set solver parameters at runtime
      call EPSSetFromOptions(eps,ierr)

! - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
!     Solve the eigensystem
! - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 

      call EPSSolve(eps,ierr) 
      call EPSGetIterationNumber(eps,its,ierr)
      if (rank .eq. 0) then
        write(6,*)
        write(6,140) its
      endif
 140  format (' Number of iterations of the method: ',i4)

!     ** Optional: Get some information from the solver and display it
      call EPSGetType(eps,type,ierr)
      if (rank .eq. 0) then
        write(6,110) type
      endif
 110  format (' Solution method: ',a)
      call EPSGetDimensions(eps,nev,PETSC_NULL_INTEGER,ierr)
      if (rank .eq. 0) then
        write(6,120) nev
      endif
 120  format (' Number of requested eigenvalues:',i2)
      call EPSGetTolerances(eps,tol,maxit,ierr)
      if (rank .eq. 0) then
        write(6,130) tol, maxit
      endif
 130  format (' Stopping condition: tol=',1pe10.4,', maxit=',i6)

! - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
!     Display solution and clean up
! - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 

!     ** Get number of converged eigenpairs (supposed to be < MAXNEV)
      call EPSGetConverged(eps,nconv,ierr)
      if (rank .eq. 0) then
        write(6,150) nconv
      endif
 150  format (' Number of converged approximate eigenpairs:',i2)

!     ** Get converged eigenpairs: i-th eigenvalue is stored in kr(i) (real part) and
!     ** ki(i) (imaginary part), and the corresponding eigenvector is stored in x(i)
      call EPSGetSolution(eps,kr,ki,x,ierr)

!     ** Compute the relative error associated to each eigenpair
      call EPSComputeError(eps,error,ierr)

!     ** Display eigenvalues and relative errors
      if (nconv.gt.0 .and. rank.eq.0) then
        write(6,*)
        write(6,160)
        write(6,170)
        do i=1,nconv
          if (ki(i).ne.0.D0) then
            write(6,180) kr(i), ki(i), error(i)
          else
            write(6,190) kr(i), error(i)
          endif
        enddo
        write(6,*)
      endif
 160  format ('           k           ||Ax-kx||/|kx|')
 170  format ('   ----------------- -----------------')
 180  format (1pe11.4,1pe11.4,' j ',1pe12.4)
 190  format ('   ',1pe12.4,'       ',1pe12.4)

!     ** Free work space
      call EPSDestroy(eps,ierr)
      call MatDestroy(A,ierr)

 999  continue
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
      integer     trans,one,ierr,i,N
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

