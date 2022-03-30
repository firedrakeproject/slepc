!
!  - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
!  SLEPc - Scalable Library for Eigenvalue Problem Computations
!  Copyright (c) 2002-, Universitat Politecnica de Valencia, Spain
!
!  This file is part of SLEPc.
!  SLEPc is distributed under a 2-clause BSD license (see LICENSE).
!  - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
!
        module slepcdsdef
        use slepcfndef
        use slepcrgdef
        use slepcsys
#include <../src/sys/classes/ds/f90-mod/slepcds.h>
        end module

        module slepcds
        use slepcdsdef
        use slepcfn
        use slepcrg
#include <../src/sys/classes/ds/f90-mod/slepcds.h90>
        interface
#include <../src/sys/classes/ds/f90-mod/ftn-auto-interfaces/slepcds.h90>
        end interface
        end module

