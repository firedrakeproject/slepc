!
!  - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
!  SLEPc - Scalable Library for Eigenvalue Problem Computations
!  Copyright (c) 2002-, Universitat Politecnica de Valencia, Spain
!
!  This file is part of SLEPc.
!  SLEPc is distributed under a 2-clause BSD license (see LICENSE).
!  - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
!
        module slepcbvdefdummy
        use slepcsysdef
#include <../src/sys/classes/bv/f90-mod/slepcbv.h>
        end module

        module slepcbvdef
        use slepcbvdefdummy
        interface operator (.ne.)
          function bvnotequal(A,B)
            import tBV
            logical bvnotequal
            type(tBV), intent(in) :: A,B
          end function
        end interface operator (.ne.)
        interface operator (.eq.)
          function bvequals(A,B)
            import tBV
            logical bvequals
            type(tBV), intent(in) :: A,B
          end function
        end interface operator (.eq.)
        end module

        function bvnotequal(A,B)
          use slepcbvdefdummy, only: tBV
          logical bvnotequal
          type(tBV), intent(in) :: A,B
          bvnotequal = (A%v .ne. B%v)
        end function

        function bvequals(A,B)
          use slepcbvdefdummy, only: tBV
          logical bvequals
          type(tBV), intent(in) :: A,B
          bvequals = (A%v .eq. B%v)
        end function

        module slepcbv
        use slepcbvdef
        use slepcsys
#include <../src/sys/classes/bv/f90-mod/slepcbv.h90>
        interface
#include <../src/sys/classes/bv/f90-mod/ftn-auto-interfaces/slepcbv.h90>
        end interface
        end module
