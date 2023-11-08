!
!  - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
!  SLEPc - Scalable Library for Eigenvalue Problem Computations
!  Copyright (c) 2002-, Universitat Politecnica de Valencia, Spain
!
!  This file is part of SLEPc.
!  SLEPc is distributed under a 2-clause BSD license (see LICENSE).
!  - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
!
        module slepcsvddefdummy
        use slepcepsdef
#include <../src/svd/f90-mod/slepcsvd.h>
        end module

        module slepcsvddef
        use slepcsvddefdummy
        interface operator (.ne.)
          function svdnotequal(A,B)
            import tSVD
            logical svdnotequal
            type(tSVD), intent(in) :: A,B
          end function
        end interface operator (.ne.)
        interface operator (.eq.)
          function svdequals(A,B)
            import tSVD
            logical svdequals
            type(tSVD), intent(in) :: A,B
          end function
        end interface operator (.eq.)
        end module

        function svdnotequal(A,B)
          use slepcsvddefdummy, only: tSVD
          logical svdnotequal
          type(tSVD), intent(in) :: A,B
          svdnotequal = (A%v .ne. B%v)
        end function

        function svdequals(A,B)
          use slepcsvddefdummy, only: tSVD
          logical svdequals
          type(tSVD), intent(in) :: A,B
          svdequals = (A%v .eq. B%v)
        end function

        module slepcsvd
        use slepcsvddef
        use slepceps
#include <../src/svd/f90-mod/slepcsvd.h90>
        interface
#include <../src/svd/f90-mod/ftn-auto-interfaces/slepcsvd.h90>
        end interface
        end module
