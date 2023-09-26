!
!  - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
!  SLEPc - Scalable Library for Eigenvalue Problem Computations
!  Copyright (c) 2002-, Universitat Politecnica de Valencia, Spain
!
!  This file is part of SLEPc.
!  SLEPc is distributed under a 2-clause BSD license (see LICENSE).
!  - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
!
        module slepcdsdefdummy
        use slepcfndef
        use slepcrgdef
#include <../src/sys/classes/ds/f90-mod/slepcds.h>
        end module

        module slepcdsdef
        use slepcdsdefdummy
        interface operator (.ne.)
          function dsnotequal(A,B)
            import tDS
            logical dsnotequal
            type(tDS), intent(in) :: A,B
          end function
        end interface operator (.ne.)
        interface operator (.eq.)
          function dsequals(A,B)
            import tDS
            logical dsequals
            type(tDS), intent(in) :: A,B
          end function
        end interface operator (.eq.)
        end module

        function dsnotequal(A,B)
          use slepcdsdefdummy, only: tDS
          logical dsnotequal
          type(tDS), intent(in) :: A,B
          dsnotequal = (A%v .ne. B%v)
        end function

        function dsequals(A,B)
          use slepcdsdefdummy, only: tDS
          logical dsequals
          type(tDS), intent(in) :: A,B
          dsequals = (A%v .eq. B%v)
        end function

        module slepcds
        use slepcdsdef
        use slepcfn
        use slepcrg
#include <../src/sys/classes/ds/f90-mod/slepcds.h90>
        interface
#include <../src/sys/classes/ds/f90-mod/ftn-auto-interfaces/slepcds.h90>
        end interface
        end module
