#define PETSC_USE_FORTRAN_MODULES

#include "finclude/slepcalldef.h"

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
        use petsc
#include "finclude/slepcip.h"
#include "finclude/ftn-custom/slepcipdef.h90"
        end module

        module slepcip
        use slepcipdef
#include "finclude/slepcip.h90"
        end module

        module slepcepsdef
        use slepcst
        use slepcip
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
