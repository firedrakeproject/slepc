!
!  - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
!  SLEPc - Scalable Library for Eigenvalue Problem Computations
!  Copyright (c) 2002-, Universitat Politecnica de Valencia, Spain
!
!  This file is part of SLEPc.
!  SLEPc is distributed under a 2-clause BSD license (see LICENSE).
!  - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
!
        module slepcpepdefdummy
        use slepcepsdef
#include <../src/pep/f90-mod/slepcpep.h>
        end module

        module slepcpepdef
        use slepcpepdefdummy
        interface operator (.ne.)
          function pepnotequal(A,B)
            import tPEP
            logical pepnotequal
            type(tPEP), intent(in) :: A,B
          end function
        end interface operator (.ne.)
        interface operator (.eq.)
          function pepequals(A,B)
            import tPEP
            logical pepequals
            type(tPEP), intent(in) :: A,B
          end function
        end interface operator (.eq.)
        end module

        function pepnotequal(A,B)
          use slepcpepdefdummy, only: tPEP
          logical pepnotequal
          type(tPEP), intent(in) :: A,B
          pepnotequal = (A%v .ne. B%v)
        end function

        function pepequals(A,B)
          use slepcpepdefdummy, only: tPEP
          logical pepequals
          type(tPEP), intent(in) :: A,B
          pepequals = (A%v .eq. B%v)
        end function

        module slepcpep
        use slepcpepdef
        use slepceps
#include <../src/pep/f90-mod/slepcpep.h90>
        interface
#include <../src/pep/f90-mod/ftn-auto-interfaces/slepcpep.h90>
        end interface
        end module
