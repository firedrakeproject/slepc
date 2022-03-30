!
!  - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
!  SLEPc - Scalable Library for Eigenvalue Problem Computations
!  Copyright (c) 2002-, Universitat Politecnica de Valencia, Spain
!
!  This file is part of SLEPc.
!  SLEPc is distributed under a 2-clause BSD license (see LICENSE).
!  - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
!
        module slepcmfndef
        use slepcsys
        use slepcbvdef
        use slepcfndef
#include <../src/mfn/f90-mod/slepcmfn.h>
        end module

        module slepcmfn
        use slepcmfndef
        use slepcbv
        use slepcfn
#include <../src/mfn/f90-mod/slepcmfn.h90>
        interface
#include <../src/mfn/f90-mod/ftn-auto-interfaces/slepcmfn.h90>
        end interface
        end module

