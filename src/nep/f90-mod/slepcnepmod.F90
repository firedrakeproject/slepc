!
!  - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
!  SLEPc - Scalable Library for Eigenvalue Problem Computations
!  Copyright (c) 2002-, Universitat Politecnica de Valencia, Spain
!
!  This file is part of SLEPc.
!  SLEPc is distributed under a 2-clause BSD license (see LICENSE).
!  - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
!
        module slepcnepdefdummy
        use slepcpepdef
        use slepcfndef
#include <../src/nep/f90-mod/slepcnep.h>
        end module

        module slepcnepdef
        use slepcnepdefdummy
        interface operator (.ne.)
          function nepnotequal(A,B)
            import tNEP
            logical nepnotequal
            type(tNEP), intent(in) :: A,B
          end function
        end interface operator (.ne.)
        interface operator (.eq.)
          function nepequals(A,B)
            import tNEP
            logical nepequals
            type(tNEP), intent(in) :: A,B
          end function
        end interface operator (.eq.)
        end module

        function nepnotequal(A,B)
          use slepcnepdefdummy, only: tNEP
          logical nepnotequal
          type(tNEP), intent(in) :: A,B
          nepnotequal = (A%v .ne. B%v)
        end function

        function nepequals(A,B)
          use slepcnepdefdummy, only: tNEP
          logical nepequals
          type(tNEP), intent(in) :: A,B
          nepequals = (A%v .eq. B%v)
        end function

        module slepcnep
        use slepcnepdef
        use slepcpep
        use slepcfn
#include <../src/nep/f90-mod/slepcnep.h90>
        interface
#include <../src/nep/f90-mod/ftn-auto-interfaces/slepcnep.h90>
        end interface
        end module

! The following module imports all the functionality of SLEPc and PETSc
        module slepc
        use slepcnep
        use slepcmfn
        use petsc
        end module
