!  - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
!  SLEPc - Scalable Library for Eigenvalue Problem Computations
!  Copyright (c) 2002-2016, Universitat Politecnica de Valencia, Spain
!
!  This file is part of SLEPc.
!
!  SLEPc is free software: you can redistribute it and/or modify it under  the
!  terms of version 3 of the GNU Lesser General Public License as published by
!  the Free Software Foundation.
!
!  SLEPc  is  distributed in the hope that it will be useful, but WITHOUT  ANY
!  WARRANTY;  without even the implied warranty of MERCHANTABILITY or  FITNESS
!  FOR  A  PARTICULAR PURPOSE. See the GNU Lesser General Public  License  for
!  more details.
!
!  You  should have received a copy of the GNU Lesser General  Public  License
!  along with SLEPc. If not, see <http://www.gnu.org/licenses/>.
!  - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

        module slepcnepdef
        use slepcbvdef
        use slepcrgdef
        use slepcdsdef
        use slepcfndef
        use slepcepsdef
        use slepcpepdef
#include <../src/nep/f90-mod/slepcnep.h>
        end module

        module slepcnep
        use slepcnepdef
#include <../src/nep/f90-mod/slepcnep.h90>
        interface
#include <../src/nep/f90-mod/ftn-auto-interfaces/slepcnep.h90>
        end interface
        end module

