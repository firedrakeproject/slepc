!
!  - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
!  SLEPc - Scalable Library for Eigenvalue Problem Computations
!  Copyright (c) 2002-, Universitat Politecnica de Valencia, Spain
!
!  This file is part of SLEPc.
!  SLEPc is distributed under a 2-clause BSD license (see LICENSE).
!  - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
!
        module slepcepsdefdummy
        use slepcstdef
        use slepcbvdef
        use slepcrgdef
        use slepcdsdef
        use slepclmedef
        use petscsnesdef
#include <../src/eps/f90-mod/slepceps.h>
        end module

        module slepcepsdef
        use slepcepsdefdummy
        interface operator (.ne.)
          function epsnotequal(A,B)
            import tEPS
            logical epsnotequal
            type(tEPS), intent(in) :: A,B
          end function
        end interface operator (.ne.)
        interface operator (.eq.)
          function epsequals(A,B)
            import tEPS
            logical epsequals
            type(tEPS), intent(in) :: A,B
          end function
        end interface operator (.eq.)
        end module

        function epsnotequal(A,B)
          use slepcepsdefdummy, only: tEPS
          logical epsnotequal
          type(tEPS), intent(in) :: A,B
          epsnotequal = (A%v .ne. B%v)
        end function

        function epsequals(A,B)
          use slepcepsdefdummy, only: tEPS
          logical epsequals
          type(tEPS), intent(in) :: A,B
          epsequals = (A%v .eq. B%v)
        end function

        module slepceps
        use slepcepsdef
        use slepcst
        use slepcbv
        use slepcrg
        use slepcds
        use slepclme
        use petscsnes
#include <../src/eps/f90-mod/slepceps.h90>
        interface
#include <../src/eps/f90-mod/ftn-auto-interfaces/slepceps.h90>
        end interface
        end module
