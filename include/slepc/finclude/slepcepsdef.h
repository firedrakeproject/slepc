!
!  Include file for Fortran use of the EPS object in SLEPc
!
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
!
#if !defined(__SLEPCEPSDEF_H)
#define __SLEPCEPSDEF_H

#include "slepc/finclude/slepcstdef.h"
#include "slepc/finclude/slepcbvdef.h"
#include "slepc/finclude/slepcdsdef.h"
#include "slepc/finclude/slepcrgdef.h"

#if !defined(PETSC_USE_FORTRAN_DATATYPES)
#define EPS PetscFortranAddr
#endif

#define EPSType                character*(80)
#define EPSConvergedReason     PetscEnum
#define EPSErrorType           PetscEnum
#define EPSProblemType         PetscEnum
#define EPSWhich               PetscEnum
#define EPSExtraction          PetscEnum
#define EPSBalance             PetscEnum
#define EPSConv                PetscEnum
#define EPSStop                PetscEnum
#define EPSPowerShiftType      PetscEnum
#define EPSLanczosReorthogType PetscEnum
#define EPSPRIMMEMethod        PetscEnum
#define EPSCISSQuadRule        PetscEnum
#define EPSCISSExtraction      PetscEnum

#define EPSPOWER       'power'
#define EPSSUBSPACE    'subspace'
#define EPSARNOLDI     'arnoldi'
#define EPSLANCZOS     'lanczos'
#define EPSKRYLOVSCHUR 'krylovschur'
#define EPSGD          'gd'
#define EPSJD          'jd'
#define EPSRQCG        'rqcg'
#define EPSLOBPCG      'lobpcg'
#define EPSCISS        'ciss'
#define EPSLAPACK      'lapack'
#define EPSARPACK      'arpack'
#define EPSBLZPACK     'blzpack'
#define EPSTRLAN       'trlan'
#define EPSBLOPEX      'blopex'
#define EPSPRIMME      'primme'
#define EPSFEAST       'feast'

#endif
