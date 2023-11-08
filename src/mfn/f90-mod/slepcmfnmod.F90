!
!  - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
!  SLEPc - Scalable Library for Eigenvalue Problem Computations
!  Copyright (c) 2002-, Universitat Politecnica de Valencia, Spain
!
!  This file is part of SLEPc.
!  SLEPc is distributed under a 2-clause BSD license (see LICENSE).
!  - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
!
        module slepcmfndefdummy
        use slepcbvdef
        use slepcfndef
#include <../src/mfn/f90-mod/slepcmfn.h>
        end module

        module slepcmfndef
        use slepcmfndefdummy
        interface operator (.ne.)
          function mfnnotequal(A,B)
            import tMFN
            logical mfnnotequal
            type(tMFN), intent(in) :: A,B
          end function
        end interface operator (.ne.)
        interface operator (.eq.)
          function mfnequals(A,B)
            import tMFN
            logical mfnequals
            type(tMFN), intent(in) :: A,B
          end function
        end interface operator (.eq.)
        end module

        function mfnnotequal(A,B)
          use slepcmfndefdummy, only: tMFN
          logical mfnnotequal
          type(tMFN), intent(in) :: A,B
          mfnnotequal = (A%v .ne. B%v)
        end function

        function mfnequals(A,B)
          use slepcmfndefdummy, only: tMFN
          logical mfnequals
          type(tMFN), intent(in) :: A,B
          mfnequals = (A%v .eq. B%v)
        end function

        module slepcmfn
        use slepcmfndef
        use slepcbv
        use slepcfn
#include <../src/mfn/f90-mod/slepcmfn.h90>
        interface
#include <../src/mfn/f90-mod/ftn-auto-interfaces/slepcmfn.h90>
        end interface
        end module
