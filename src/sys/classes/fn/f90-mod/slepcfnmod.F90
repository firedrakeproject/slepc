!
!  - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
!  SLEPc - Scalable Library for Eigenvalue Problem Computations
!  Copyright (c) 2002-, Universitat Politecnica de Valencia, Spain
!
!  This file is part of SLEPc.
!  SLEPc is distributed under a 2-clause BSD license (see LICENSE).
!  - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
!
        module slepcfndefdummy
        use slepcsysdef
#include <../src/sys/classes/fn/f90-mod/slepcfn.h>
        end module

        module slepcfndef
        use slepcfndefdummy
        interface operator (.ne.)
          function fnnotequal(A,B)
            import tFN
            logical fnnotequal
            type(tFN), intent(in) :: A,B
          end function
        end interface operator (.ne.)
        interface operator (.eq.)
          function fnequals(A,B)
            import tFN
            logical fnequals
            type(tFN), intent(in) :: A,B
          end function
        end interface operator (.eq.)
        end module

        function fnnotequal(A,B)
          use slepcfndefdummy, only: tFN
          logical fnnotequal
          type(tFN), intent(in) :: A,B
          fnnotequal = (A%v .ne. B%v)
        end function

        function fnequals(A,B)
          use slepcfndefdummy, only: tFN
          logical fnequals
          type(tFN), intent(in) :: A,B
          fnequals = (A%v .eq. B%v)
        end function

        module slepcfn
        use slepcfndef
        use slepcsys
#include <../src/sys/classes/fn/f90-mod/slepcfn.h90>
        interface
#include <../src/sys/classes/fn/f90-mod/ftn-auto-interfaces/slepcfn.h90>
        end interface
        end module
