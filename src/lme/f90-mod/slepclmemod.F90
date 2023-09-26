!
!  - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
!  SLEPc - Scalable Library for Eigenvalue Problem Computations
!  Copyright (c) 2002-, Universitat Politecnica de Valencia, Spain
!
!  This file is part of SLEPc.
!  SLEPc is distributed under a 2-clause BSD license (see LICENSE).
!  - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
!
        module slepclmedefdummy
        use slepcbvdef
#include <../src/lme/f90-mod/slepclme.h>
        end module

        module slepclmedef
        use slepclmedefdummy
        interface operator (.ne.)
          function lmenotequal(A,B)
            import tLME
            logical lmenotequal
            type(tLME), intent(in) :: A,B
          end function
        end interface operator (.ne.)
        interface operator (.eq.)
          function lmeequals(A,B)
            import tLME
            logical lmeequals
            type(tLME), intent(in) :: A,B
          end function
        end interface operator (.eq.)
        end module

        function lmenotequal(A,B)
          use slepclmedefdummy, only: tLME
          logical lmenotequal
          type(tLME), intent(in) :: A,B
          lmenotequal = (A%v .ne. B%v)
        end function

        function lmeequals(A,B)
          use slepclmedefdummy, only: tLME
          logical lmeequals
          type(tLME), intent(in) :: A,B
          lmeequals = (A%v .eq. B%v)
        end function

        module slepclme
        use slepclmedef
        use slepcbv
#include <../src/lme/f90-mod/slepclme.h90>
        interface
#include <../src/lme/f90-mod/ftn-auto-interfaces/slepclme.h90>
        end interface
        end module
