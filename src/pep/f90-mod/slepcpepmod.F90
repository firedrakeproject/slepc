!
!  - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
!  SLEPc - Scalable Library for Eigenvalue Problem Computations
!  Copyright (c) 2002-, Universitat Politecnica de Valencia, Spain
!
!  This file is part of SLEPc.
!  SLEPc is distributed under a 2-clause BSD license (see LICENSE).
!  - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
!
        module slepcpepdef
        use slepcsys
        use slepcstdef
        use slepcbvdef
        use slepcrgdef
        use slepcdsdef
        use slepcepsdef
#include <../src/pep/f90-mod/slepcpep.h>
        end module

        module slepcpep
        use slepcpepdef
        use slepceps
#include <../src/pep/f90-mod/slepcpep.h90>
        interface
#include <../src/pep/f90-mod/ftn-auto-interfaces/slepcpep.h90>
        end interface
        end module

