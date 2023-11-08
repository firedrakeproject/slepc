!
!  - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
!  SLEPc - Scalable Library for Eigenvalue Problem Computations
!  Copyright (c) 2002-, Universitat Politecnica de Valencia, Spain
!
!  This file is part of SLEPc.
!  SLEPc is distributed under a 2-clause BSD license (see LICENSE).
!  - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
!
!  Include file for Fortran use of the SVD object in SLEPc
!
#if !defined(SLEPCSVDDEF_H)
#define SLEPCSVDDEF_H

#include "slepc/finclude/slepcbv.h"
#include "slepc/finclude/slepcds.h"
#include "slepc/finclude/slepceps.h"

#define SVD type(tSVD)

#define SVDType             character*(80)
#define SVDProblemType      PetscEnum
#define SVDConvergedReason  PetscEnum
#define SVDErrorType        PetscEnum
#define SVDWhich            PetscEnum
#define SVDConv             PetscEnum
#define SVDStop             PetscEnum
#define SVDPRIMMEMethod     PetscEnum
#define SVDTRLanczosGBidiag PetscEnum
#define SVDKSVDEigenMethod  PetscEnum
#define SVDKSVDPolarMethod  PetscEnum

#define SVDCROSS      'cross'
#define SVDCYCLIC     'cyclic'
#define SVDLAPACK     'lapack'
#define SVDLANCZOS    'lanczos'
#define SVDTRLANCZOS  'trlanczos'
#define SVDRANDOMIZED 'randomized'
#define SVDSCALAPACK  'scalapack'
#define SVDKSVD       'ksvd'
#define SVDELEMENTAL  'elemental'
#define SVDPRIMME     'primme'

#endif
