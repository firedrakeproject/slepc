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

#define PETSC_USE_FORTRAN_MODULES

        module slepcsysdef
#include <slepc/finclude/slepcsys.h>
        end module

        module slepcsys
        use slepcsysdef
        end module

        module slepcrgdef
#include <slepc/finclude/slepcrg.h>
#include <slepc/finclude/ftn-custom/slepcrgdef.h90>
        end module

        module slepcrg
        use slepcrgdef
#include <slepc/finclude/slepcrg.h90>
        end module

        module slepcfndef
#include <slepc/finclude/slepcfn.h>
#include <slepc/finclude/ftn-custom/slepcfndef.h90>
        end module

        module slepcfn
        use slepcfndef
#include <slepc/finclude/slepcfn.h90>
        end module

        module slepcdsdef
        use slepcsys
        use slepcfn
#include <slepc/finclude/slepcds.h>
#include <slepc/finclude/ftn-custom/slepcdsdef.h90>
        end module

        module slepcds
        use slepcdsdef
#include <slepc/finclude/slepcds.h90>
        end module

        module slepcbvdef
#include <slepc/finclude/slepcbv.h>
#include <slepc/finclude/ftn-custom/slepcbvdef.h90>
        end module

        module slepcbv
        use slepcbvdef
#include <slepc/finclude/slepcbv.h90>
        end module

        module slepcstdef
        use petscksp
#include <slepc/finclude/slepcst.h>
#include <slepc/finclude/ftn-custom/slepcstdef.h90>
        end module

        module slepcst
        use slepcstdef
#include <slepc/finclude/slepcst.h90>
        end module

