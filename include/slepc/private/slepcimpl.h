/*
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
   SLEPc - Scalable Library for Eigenvalue Problem Computations
   Copyright (c) 2002-2016, Universitat Politecnica de Valencia, Spain

   This file is part of SLEPc.

   SLEPc is free software: you can redistribute it and/or modify it under  the
   terms of version 3 of the GNU Lesser General Public License as published by
   the Free Software Foundation.

   SLEPc  is  distributed in the hope that it will be useful, but WITHOUT  ANY
   WARRANTY;  without even the implied warranty of MERCHANTABILITY or  FITNESS
   FOR  A  PARTICULAR PURPOSE. See the GNU Lesser General Public  License  for
   more details.

   You  should have received a copy of the GNU Lesser General  Public  License
   along with SLEPc. If not, see <http://www.gnu.org/licenses/>.
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
*/

#if !defined(_SLEPCIMPL)
#define _SLEPCIMPL

#include <slepcsys.h>
#include <petsc/private/petscimpl.h>

PETSC_INTERN PetscBool SlepcBeganPetsc;

/*@C
    SlepcHeaderCreate - Creates a SLEPc object

    Input Parameters:
+   classid - the classid associated with this object
.   class_name - string name of class; should be static
.   descr - string containing short description; should be static
.   mansec - string indicating section in manual pages; should be static
.   comm - the MPI Communicator
.   destroy - the destroy routine for this object
-   view - the view routine for this object

    Output Parameter:
.   h - the newly created object

    Note:
    This is equivalent to PetscHeaderCreate but makes sure that SlepcInitialize
    has been called.

    Level: developer
@*/
#define SlepcHeaderCreate(h,classid,class_name,descr,mansec,comm,destroy,view) \
    ((!SlepcInitializeCalled && \
    PetscError(comm,__LINE__,PETSC_FUNCTION_NAME,__FILE__,1,PETSC_ERROR_INITIAL, \
    "Must call SlepcInitialize instead of PetscInitialize to use SLEPc classes")) ||  \
    PetscHeaderCreate(h,classid,class_name,descr,mansec,comm,destroy,view))

/* context for monitors of type XXXMonitorConverged */
struct _n_SlepcConvMonitor {
  PetscViewer       viewer;
  PetscViewerFormat format;
  PetscInt          oldnconv;
};

/* Private functions that are shared by several classes */
PETSC_EXTERN PetscErrorCode SlepcBasisReference_Private(PetscInt,Vec*,PetscInt*,Vec**);
PETSC_EXTERN PetscErrorCode SlepcBasisDestroy_Private(PetscInt*,Vec**);

PETSC_INTERN PetscErrorCode SlepcCitationsInitialize(void);
PETSC_INTERN PetscErrorCode SlepcInitialize_DynamicLibraries(void);
PETSC_INTERN PetscErrorCode SlepcInitialize_Packages(void);

#endif
