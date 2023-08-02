!
!  - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
!  SLEPc - Scalable Library for Eigenvalue Problem Computations
!  Copyright (c) 2002-, Universitat Politecnica de Valencia, Spain
!
!  This file is part of SLEPc.
!  SLEPc is distributed under a 2-clause BSD license (see LICENSE).
!  - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
!
        module slepcnepdef
        use slepcpepdef
        use slepcfndef
#include <../src/nep/f90-mod/slepcnep.h>
        end module

        module slepcnep
        use slepcnepdef
        use slepcpep
        use slepcfn
#include <../src/nep/f90-mod/slepcnep.h90>
        interface
#include <../src/nep/f90-mod/ftn-auto-interfaces/slepcnep.h90>
        end interface
        end module

! The following module imports all the functionality of SLEPc and PETSc
        module slepc
        use slepcnep
        use slepcmfn
        use petsc
        end module

