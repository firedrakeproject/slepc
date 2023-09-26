!
!  - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
!  SLEPc - Scalable Library for Eigenvalue Problem Computations
!  Copyright (c) 2002-, Universitat Politecnica de Valencia, Spain
!
!  This file is part of SLEPc.
!  SLEPc is distributed under a 2-clause BSD license (see LICENSE).
!  - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
!
        module slepcstdefdummy
        use petsckspdef
        use slepcbvdef
#include <../src/sys/classes/st/f90-mod/slepcst.h>
        end module

        module slepcstdef
        use slepcstdefdummy
        interface operator (.ne.)
          function stnotequal(A,B)
            import tST
            logical stnotequal
            type(tST), intent(in) :: A,B
          end function
        end interface operator (.ne.)
        interface operator (.eq.)
          function stequals(A,B)
            import tST
            logical stequals
            type(tST), intent(in) :: A,B
          end function
        end interface operator (.eq.)
        end module

        function stnotequal(A,B)
          use slepcstdefdummy, only: tST
          logical stnotequal
          type(tST), intent(in) :: A,B
          stnotequal = (A%v .ne. B%v)
        end function

        function stequals(A,B)
          use slepcstdefdummy, only: tST
          logical stequals
          type(tST), intent(in) :: A,B
          stequals = (A%v .eq. B%v)
        end function

        module slepcst
        use slepcstdef
        use petscksp
        use slepcbv
#include <../src/sys/classes/st/f90-mod/slepcst.h90>
        interface
#include <../src/sys/classes/st/f90-mod/ftn-auto-interfaces/slepcst.h90>
        end interface
        end module
