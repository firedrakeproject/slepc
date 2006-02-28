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
      PetscReal    tol, error, re, im
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
        write(*,100) n
      endif
 100  format (/'1-D Laplacian Eigenproblem, n =',I3)

! - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
!     Compute the operator matrix that defines the eigensystem, Ax=kx
! - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 

      call MatCreate(PETSC_COMM_WORLD,A,ierr)
      call MatSetSizes(A,PETSC_DECIDE,PETSC_DECIDE,n,n,ierr)
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
      call EPSSetProblemType(eps,EPS_HEP,ierr)

!     ** Set solver parameters at runtime
      call EPSSetFromOptions(eps,ierr)

! - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
!     Solve the eigensystem
! - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 

      call EPSSolve(eps,ierr) 
      call EPSGetIterationNumber(eps,its,ierr)
      if (rank .eq. 0) then
        write(*,110) its
      endif
 110  format (/' Number of iterations of the method:',I4)
  
!     ** Optional: Get some information from the solver and display it
      call EPSGetType(eps,type,ierr)
      if (rank .eq. 0) then
        write(*,120) type
      endif
 120  format (' Solution method: ',A)
      call EPSGetDimensions(eps,nev,PETSC_NULL_INTEGER,ierr)
      if (rank .eq. 0) then
        write(*,130) nev
      endif
 130  format (/' Number of requested eigenvalues:',I2)
      call EPSGetTolerances(eps,tol,maxit,ierr)
      if (rank .eq. 0) then
        write(*,140) tol, maxit
      endif
 140  format (' Stopping condition: tol=',1P,E10.4,', maxit=',I4)

! - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
!     Display solution and clean up
! - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 

!     ** Get number of converged eigenpairs
      call EPSGetConverged(eps,nconv,ierr)
      if (rank .eq. 0) then
        write(*,150) nconv
      endif
 150  format (' Number of converged eigenpairs:',I2/)

!     ** Display eigenvalues and relative errors
      if (nconv.gt.0 .and. rank.eq.0) then
        write(*,*) '         k          ||Ax-kx||/||kx||'
        write(*,*) ' ----------------- ------------------'
        do i=0,nconv-1
!         ** Get converged eigenpairs: i-th eigenvalue is stored in kr 
!         ** (real part) and ki (imaginary part)
          call EPSGetEigenpair(eps,i,kr,ki,PETSC_NULL,PETSC_NULL,ierr)

!         ** Compute the relative error associated to each eigenpair
          call EPSComputeRelativeError(eps,i,error,ierr)

#if defined(PETSC_USE_COMPLEX)
          re = PetscRealPart(kr);
          im = PetscImaginaryPart(kr);
#else
          re = kr;
          im = ki;
#endif 
          if (im.ne.0.D0) then
            write(*,160) re, im, error
 160        format (1P,E11.4,E11.4,' j ',E12.4)
          else
            write(*,170) re, error
 170        format (1P,'   ',E12.4,'       ',E12.4)
          endif

        enddo
        write(*,*)
      endif

!     ** Free work space
      call EPSDestroy(eps,ierr)
      call MatDestroy(A,ierr)

      call SlepcFinalize(ierr)
      end

