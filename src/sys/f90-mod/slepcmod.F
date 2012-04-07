!  - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
!  SLEPc - Scalable Library for Eigenvalue Problem Computations
!  Copyright (c) 2002-2011, Universitat Politecnica de Valencia, Spain
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

#define PETSC_USE_FORTRAN_MODULES

#include "finclude/petscdef.h"
#include "finclude/slepcdef.h"

        module slepcstdef
        use petscksp
#include "finclude/slepcst.h"
#include "finclude/ftn-custom/slepcstdef.h90"
        end module

        module slepcst
        use slepcstdef
#include "finclude/slepcst.h90"
        end module

        module slepcipdef
        use petscsys
#include "finclude/slepcip.h"
#include "finclude/ftn-custom/slepcipdef.h90"
        end module

        module slepcip
        use slepcipdef
#include "finclude/slepcip.h90"
        end module

        module slepcpsdef
        use petscsys
#include "finclude/slepcps.h"
#include "finclude/ftn-custom/slepcpsdef.h90"
        end module

        module slepcps
        use slepcpsdef
#include "finclude/slepcps.h90"
        end module

        module slepcepsdef
        use slepcst
        use slepcip
        use slepcps
#include "finclude/slepceps.h"
#include "finclude/ftn-custom/slepcepsdef.h90"
        end module

        module slepceps
        use slepcepsdef
#include "finclude/slepceps.h90"
        end module

        module slepcsvddef
        use slepcip
        use slepceps
#include "finclude/slepcsvd.h"
#include "finclude/ftn-custom/slepcsvddef.h90"
        end module

        module slepcsvd
        use slepcsvddef
#include "finclude/slepcsvd.h90"
        end module

        module slepcqepdef
        use slepcip
        use slepceps
#include "finclude/slepcqep.h"
#include "finclude/ftn-custom/slepcqepdef.h90"
        end module

        module slepcqep
        use slepcqepdef
#include "finclude/slepcqep.h90"
        end module

