!
!  Program usage: mpirun -np n ex1f [-help] [-n <n>] [all SLEPc options] 
!
!  Description: Simple example that solves an eigensystem with the EPS object.
!  The standard symmetric eigenvalue problem to be solved corresponds to the 
!  Laplacian operator in 1 dimension. 
!
!  The command line options are:
!    -n <n>, where <n> = number of grid points = matrix size
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

      Mat          A
      Vec          x
      EPS          eps
      EPSType      type
      PetscReal    tol, error
      PetscScalar  kr, ki
      integer      rank, n, nev, ierr, maxit, i, its, nconv
      integer      col(3), Istart, Iend
      PetscTruth   flg
      PetscScalar  value(3)

! - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
!     Beginning of program
! - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 

      call SlepcInitialize(PETSC_NULL_CHARACTER,ierr)
      call MPI_Comm_rank(PETSC_COMM_WORLD,rank,ierr)
      n = 30
      call PetscOptionsGetInt(PETSC_NULL_CHARACTER,'-n',n,flg,ierr)

      if (rank .eq. 0) then
        write(6,*)
        write(6,100) n
        write(6,*)
      endif
 100  format ('1-D Laplacian Eigenproblem, n =',i6)

! - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
!     Compute the operator matrix that defines the eigensystem, Ax=kx
! - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 

      call MatCreate(PETSC_COMM_WORLD,PETSC_DECIDE,PETSC_DECIDE,n,n,A,
     &               ierr)
      call MatSetFromOptions(A,ierr)

      call MatGetOwnershipRange(A,Istart,Iend,ierr)
      if (Istart .eq. 0) then 
        i = 0
        col(1) = 0
        col(2) = 1
        value(1) =  2.0
        value(2) = -1.0
        call MatSetValues(A,1,i,2,col,value,INSERT_VALUES,ierr)
        Istart = Istart+1
      endif
      if (Iend .eq. n) then 
        i = n-1
        col(1) = n-2
        col(2) = n-1
        value(1) = -1.0
        value(2) =  2.0
        call MatSetValues(A,1,i,2,col,value,INSERT_VALUES,ierr)
        Iend = Iend-1
      endif
      value(1) = -1.0
      value(2) =  2.0
      value(3) = -1.0
      do i=Istart,Iend-1
        col(1) = i-1
        col(2) = i
        col(3) = i+1
        call MatSetValues(A,1,i,3,col,value,INSERT_VALUES,ierr)
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

!     ** Display eigenvalues and relative errors
      if (nconv.gt.0 .and. rank.eq.0) then
        write(6,*)
        write(6,160)
        write(6,170)
        do i=0,nconv-1
!     ** Get converged eigenpairs: i-th eigenvalue is stored in kr (real part) and
!     ** ki (imaginary part), and the corresponding eigenvector is stored in x
          call EPSGetEigenpair(eps,i,kr,ki,x,PETSC_NULL,ierr)

!     ** Compute the relative error associated to each eigenpair
          call EPSComputeRelativeError(eps,i,error,ierr)

          if (ki.ne.0.D0) then
            write(6,180) kr, ki, error
          else
            write(6,190) kr, error
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

      call SlepcFinalize(ierr)
      end

