!
!  - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
!  SLEPc - Scalable Library for Eigenvalue Problem Computations
!  Copyright (c) 2002-, Universitat Politecnica de Valencia, Spain
!
!  This file is part of SLEPc.
!  SLEPc is distributed under a 2-clause BSD license (see LICENSE).
!  - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
!
        module slepcrgdefdummy
        use slepcsysdef
#include <../src/sys/classes/rg/f90-mod/slepcrg.h>
        end module

        module slepcrgdef
        use slepcrgdefdummy
        interface operator (.ne.)
          function rgnotequal(A,B)
            import tRG
            logical rgnotequal
            type(tRG), intent(in) :: A,B
          end function
        end interface operator (.ne.)
        interface operator (.eq.)
          function rgequals(A,B)
            import tRG
            logical rgequals
            type(tRG), intent(in) :: A,B
          end function
        end interface operator (.eq.)
        end module

        function rgnotequal(A,B)
          use slepcrgdefdummy, only: tRG
          logical rgnotequal
          type(tRG), intent(in) :: A,B
          rgnotequal = (A%v .ne. B%v)
        end function

        function rgequals(A,B)
          use slepcrgdefdummy, only: tRG
          logical rgequals
          type(tRG), intent(in) :: A,B
          rgequals = (A%v .eq. B%v)
        end function

        module slepcrg
        use slepcrgdef
        use slepcsys
#include <../src/sys/classes/rg/f90-mod/slepcrg.h90>
        interface
#include <../src/sys/classes/rg/f90-mod/ftn-auto-interfaces/slepcrg.h90>
        end interface
        end module
