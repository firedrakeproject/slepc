/*
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
   SLEPc - Scalable Library for Eigenvalue Problem Computations
   Copyright (c) 2002-2013, Universitat Politecnica de Valencia, Spain

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

#include <slepc-private/qepimpl.h>

static PetscBool QEPPackageInitialized = PETSC_FALSE;

#undef __FUNCT__
#define __FUNCT__ "QEPFinalizePackage"
/*@C
   QEPFinalizePackage - This function destroys everything in the Slepc interface
   to the QEP package. It is called from SlepcFinalize().

   Level: developer

.seealso: SlepcFinalize()
@*/
PetscErrorCode QEPFinalizePackage(void)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscFunctionListDestroy(&QEPList);CHKERRQ(ierr);
  QEPPackageInitialized = PETSC_FALSE;
  QEPRegisterAllCalled  = PETSC_FALSE;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "QEPInitializePackage"
/*@C
   QEPInitializePackage - This function initializes everything in the QEP package.
   It is called from PetscDLLibraryRegister() when using dynamic libraries, and
   on the first call to QEPCreate() when using static libraries.

   Level: developer

.seealso: SlepcInitialize()
@*/
PetscErrorCode QEPInitializePackage(void)
{
  char           logList[256];
  char           *className;
  PetscBool      opt;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (QEPPackageInitialized) PetscFunctionReturn(0);
  QEPPackageInitialized = PETSC_TRUE;
  /* Register Classes */
  ierr = PetscClassIdRegister("Quadratic Eigenvalue Problem solver",&QEP_CLASSID);CHKERRQ(ierr);
  /* Register Constructors */
  ierr = QEPRegisterAll();CHKERRQ(ierr);
  /* Register Events */
  ierr = PetscLogEventRegister("QEPSetUp",QEP_CLASSID,&QEP_SetUp);CHKERRQ(ierr);
  ierr = PetscLogEventRegister("QEPSolve",QEP_CLASSID,&QEP_Solve);CHKERRQ(ierr);
  /* Process info exclusions */
  ierr = PetscOptionsGetString(NULL,"-info_exclude",logList,256,&opt);CHKERRQ(ierr);
  if (opt) {
    ierr = PetscStrstr(logList,"qep",&className);CHKERRQ(ierr);
    if (className) {
      ierr = PetscInfoDeactivateClass(QEP_CLASSID);CHKERRQ(ierr);
    }
  }
  /* Process summary exclusions */
  ierr = PetscOptionsGetString(NULL,"-log_summary_exclude",logList,256,&opt);CHKERRQ(ierr);
  if (opt) {
    ierr = PetscStrstr(logList,"qep",&className);CHKERRQ(ierr);
    if (className) {
      ierr = PetscLogEventDeactivateClass(QEP_CLASSID);CHKERRQ(ierr);
    }
  }
  ierr = PetscRegisterFinalize(QEPFinalizePackage);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#if defined(PETSC_HAVE_DYNAMIC_LIBRARIES)

#undef __FUNCT__
#define __FUNCT__ "PetscDLLibraryRegister_slepcqep"
/*
  PetscDLLibraryRegister - This function is called when the dynamic library
  it is in is opened.

  This one registers all the QEP methods that are in the basic SLEPc libslepcqep
  library.
 */
PETSC_EXTERN PetscErrorCode PetscDLLibraryRegister_slepcqep()
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = QEPInitializePackage();CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
#endif /* PETSC_HAVE_DYNAMIC_LIBRARIES */

