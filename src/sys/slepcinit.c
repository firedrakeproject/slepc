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

#include <slepc/private/slepcimpl.h>           /*I "slepcsys.h" I*/
#include <slepc/private/vecimplslepc.h>

#undef __FUNCT__
#define __FUNCT__ "SlepcGetVersion"
/*@C
    SlepcGetVersion - Gets the SLEPc version information in a string.

    Input Parameter:
.   len - length of the string

    Output Parameter:
.   version - version string

    Fortran Note:
    This routine is not supported in Fortran.

    Level: intermediate
@*/
PetscErrorCode SlepcGetVersion(char version[],size_t len)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
#if (SLEPC_VERSION_RELEASE == 1)
  ierr = PetscSNPrintf(version,len,"SLEPc Release Version %d.%d.%d, %s",SLEPC_VERSION_MAJOR,SLEPC_VERSION_MINOR,SLEPC_VERSION_SUBMINOR,SLEPC_VERSION_DATE);CHKERRQ(ierr);
#else
  ierr = PetscSNPrintf(version,len,"SLEPc Development GIT revision: %d  GIT Date: %s",SLEPC_VERSION_GIT,SLEPC_VERSION_DATE_GIT);CHKERRQ(ierr);
#endif
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "SlepcPrintVersion"
/*
   SlepcPrintVersion - Prints SLEPc version info.

   Collective on MPI_Comm
*/
static PetscErrorCode SlepcPrintVersion(MPI_Comm comm)
{
  PetscErrorCode ierr;
  char           version[256];

  PetscFunctionBegin;
  ierr = SlepcGetVersion(version,256);CHKERRQ(ierr);
  ierr = (*PetscHelpPrintf)(comm,"--------------------------------------------------------------------------\n");CHKERRQ(ierr);
  ierr = (*PetscHelpPrintf)(comm,"%s\n",version);CHKERRQ(ierr);
  ierr = (*PetscHelpPrintf)(comm,SLEPC_AUTHOR_INFO);CHKERRQ(ierr);
  ierr = (*PetscHelpPrintf)(comm,"See docs/manual.html for help.\n");CHKERRQ(ierr);
  ierr = (*PetscHelpPrintf)(comm,"SLEPc libraries linked from %s\n",SLEPC_LIB_DIR);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "SlepcPrintHelpIntro"
/*
   SlepcPrintHelpIntro - Prints introductory SLEPc help info.

   Collective on MPI_Comm
*/
static PetscErrorCode SlepcPrintHelpIntro(MPI_Comm comm)
{
  PetscErrorCode  ierr;

  PetscFunctionBegin;
  ierr = (*PetscHelpPrintf)(comm,"SLEPc help information includes that for the PETSc libraries, which provide\n");CHKERRQ(ierr);
  ierr = (*PetscHelpPrintf)(comm,"low-level system infrastructure and linear algebra tools.\n");CHKERRQ(ierr);
  ierr = (*PetscHelpPrintf)(comm,"--------------------------------------------------------------------------\n");CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/* ------------------------Nasty global variables -------------------------------*/
/*
   Indicates whether SLEPc started PETSc, or whether it was
   already started before SLEPc was initialized.
*/
PetscBool SlepcBeganPetsc = PETSC_FALSE;
PetscBool SlepcInitializeCalled = PETSC_FALSE;

#if defined(PETSC_HAVE_DYNAMIC_LIBRARIES)

#undef __FUNCT__
#define __FUNCT__ "SlepcLoadDynamicLibrary"
static PetscErrorCode SlepcLoadDynamicLibrary(const char *name,PetscBool *found)
{
  char           libs[PETSC_MAX_PATH_LEN],dlib[PETSC_MAX_PATH_LEN];
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscStrcpy(libs,SLEPC_LIB_DIR);CHKERRQ(ierr);
  ierr = PetscStrcat(libs,"/libslepc");CHKERRQ(ierr);
  ierr = PetscStrcat(libs,name);CHKERRQ(ierr);
  ierr = PetscDLLibraryRetrieve(PETSC_COMM_WORLD,libs,dlib,1024,found);CHKERRQ(ierr);
  if (*found) {
    ierr = PetscDLLibraryAppend(PETSC_COMM_WORLD,&PetscDLLibrariesLoaded,dlib);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "SlepcInitialize_DynamicLibraries"
/*
    SlepcInitialize_DynamicLibraries - Adds the default dynamic link libraries to the
    search path.
*/
PetscErrorCode SlepcInitialize_DynamicLibraries(void)
{
  PetscErrorCode ierr;
  PetscBool      found;
  PetscBool      preload;

  PetscFunctionBegin;
  preload = PETSC_FALSE;
  ierr = PetscOptionsGetBool(NULL,NULL,"-dynamic_library_preload",&preload,NULL);CHKERRQ(ierr);
  if (preload) {
#if defined(PETSC_USE_SINGLE_LIBRARY)
    ierr = SlepcLoadDynamicLibrary("",&found);CHKERRQ(ierr);
    if (!found) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_FILE_OPEN,"Unable to locate SLEPc dynamic library\nYou cannot move the dynamic libraries!");
#else
    ierr = SlepcLoadDynamicLibrary("sys",&found);CHKERRQ(ierr);
    if (!found) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_FILE_OPEN,"Unable to locate SLEPc dynamic library\nYou cannot move the dynamic libraries!");
    ierr = SlepcLoadDynamicLibrary("eps",&found);CHKERRQ(ierr);
    if (!found) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_FILE_OPEN,"Unable to locate SLEPc dynamic library\nYou cannot move the dynamic libraries!");
    ierr = SlepcLoadDynamicLibrary("pep",&found);CHKERRQ(ierr);
    if (!found) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_FILE_OPEN,"Unable to locate SLEPc dynamic library\nYou cannot move the dynamic libraries!");
    ierr = SlepcLoadDynamicLibrary("nep",&found);CHKERRQ(ierr);
    if (!found) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_FILE_OPEN,"Unable to locate SLEPc dynamic library\nYou cannot move the dynamic libraries!");
    ierr = SlepcLoadDynamicLibrary("svd",&found);CHKERRQ(ierr);
    if (!found) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_FILE_OPEN,"Unable to locate SLEPc dynamic library\nYou cannot move the dynamic libraries!");
    ierr = SlepcLoadDynamicLibrary("mfn",&found);CHKERRQ(ierr);
    if (!found) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_FILE_OPEN,"Unable to locate SLEPc dynamic library\nYou cannot move the dynamic libraries!");
#endif
  }

#if defined(PETSC_HAVE_THREADSAFETY)
  ierr = STInitializePackage();CHKERRQ(ierr);
  ierr = DSInitializePackage();CHKERRQ(ierr);
  ierr = FNInitializePackage();CHKERRQ(ierr);
  ierr = BVInitializePackage();CHKERRQ(ierr);
  ierr = RGInitializePackage();CHKERRQ(ierr);
  ierr = EPSInitializePackage();CHKERRQ(ierr);
  ierr = SVDInitializePackage();CHKERRQ(ierr);
  ierr = PEPInitializePackage();CHKERRQ(ierr);
  ierr = NEPInitializePackage();CHKERRQ(ierr);
  ierr = MFNInitializePackage();CHKERRQ(ierr);
#endif
  PetscFunctionReturn(0);
}
#endif

#undef __FUNCT__
#define __FUNCT__ "SlepcCitationsInitialize"
PetscErrorCode SlepcCitationsInitialize()
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscCitationsRegister("@Article{slepc-toms,\n"
    "   author = \"Vicente Hernandez and Jose E. Roman and Vicente Vidal\",\n"
    "   title = \"{SLEPc}: A Scalable and Flexible Toolkit for the Solution of Eigenvalue Problems\",\n"
    "   journal = \"{ACM} Trans. Math. Software\",\n"
    "   volume = \"31\",\n"
    "   number = \"3\",\n"
    "   pages = \"351--362\",\n"
    "   year = \"2005,\"\n"
    "   doi = \"http://dx.doi.org/10.1145/1089014.1089019\"\n"
    "}\n",NULL);CHKERRQ(ierr);
  ierr = PetscCitationsRegister("@TechReport{slepc-manual,\n"
    "   author = \"J. E. Roman and C. Campos and E. Romero and A. Tomas\",\n"
    "   title = \"{SLEPc} Users Manual\",\n"
    "   number = \"DSIC-II/24/02 - Revision 3.7\",\n"
    "   institution = \"D. Sistemes Inform\\`atics i Computaci\\'o, Universitat Polit\\`ecnica de Val\\`encia\",\n"
    "   year = \"2016\"\n"
    "}\n",NULL);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "SlepcInitialize"
/*@C
   SlepcInitialize - Initializes the SLEPc library. SlepcInitialize() calls
   PetscInitialize() if that has not been called yet, so this routine should
   always be called near the beginning of your program.

   Collective on MPI_COMM_WORLD or PETSC_COMM_WORLD if it has been set

   Input Parameters:
+  argc - count of number of command line arguments
.  args - the command line arguments
.  file - [optional] PETSc database file, defaults to ~username/.petscrc
          (use NULL for default)
-  help - [optional] Help message to print, use NULL for no message

   Fortran Note:
   Fortran syntax is very similar to that of PetscInitialize()

   Level: beginner

.seealso: SlepcFinalize(), PetscInitialize()
@*/
PetscErrorCode SlepcInitialize(int *argc,char ***args,const char file[],const char help[])
{
  PetscErrorCode ierr;
  PetscBool      flg;

  PetscFunctionBegin;
  if (SlepcInitializeCalled) PetscFunctionReturn(0);
  ierr = PetscSetHelpVersionFunctions(SlepcPrintHelpIntro,SlepcPrintVersion);CHKERRQ(ierr);
  ierr = PetscInitialized(&flg);CHKERRQ(ierr);
  if (!flg) {
    ierr = PetscInitialize(argc,args,file,help);CHKERRQ(ierr);
    SlepcBeganPetsc = PETSC_TRUE;
  }

  ierr = SlepcCitationsInitialize();CHKERRQ(ierr);

#if defined(PETSC_HAVE_DYNAMIC_LIBRARIES)
  ierr = SlepcInitialize_DynamicLibraries();CHKERRQ(ierr);
#endif

#if defined(PETSC_HAVE_DRAND48)
  /* work-around for Cygwin drand48() initialization bug */
  srand48(0);
#endif

  SlepcInitializeCalled = PETSC_TRUE;
  ierr = PetscInfo(0,"SLEPc successfully started\n");CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "SlepcFinalize"
/*@
   SlepcFinalize - Checks for options to be called at the conclusion
   of the SLEPc program and calls PetscFinalize().

   Collective on PETSC_COMM_WORLD

   Level: beginner

.seealso: SlepcInitialize(), PetscFinalize()
@*/
PetscErrorCode SlepcFinalize(void)
{
  PetscErrorCode ierr = 0;

  PetscFunctionBegin;
  ierr = PetscInfo(0,"SlepcFinalize() called\n");CHKERRQ(ierr);
  if (SlepcBeganPetsc) {
    ierr = PetscFinalize();
  }
  SlepcInitializeCalled = PETSC_FALSE;
  PetscFunctionReturn(ierr);
}

#undef __FUNCT__
#define __FUNCT__ "SlepcInitializeNoArguments"
/*@C
   SlepcInitializeNoArguments - Calls SlepcInitialize() from C/C++ without
   the command line arguments.

   Collective

   Level: advanced

.seealso: SlepcInitialize(), SlepcInitializeFortran()
@*/
PetscErrorCode SlepcInitializeNoArguments(void)
{
  PetscErrorCode ierr;
  int            argc = 0;
  char           **args = 0;

  PetscFunctionBegin;
  ierr = SlepcInitialize(&argc,&args,NULL,NULL);
  PetscFunctionReturn(ierr);
}

#undef __FUNCT__
#define __FUNCT__ "SlepcInitialized"
/*@
   SlepcInitialized - Determine whether SLEPc is initialized.

   Level: beginner

.seealso: SlepcInitialize(), SlepcInitializeFortran()
@*/
PetscErrorCode SlepcInitialized(PetscBool *isInitialized)
{
  PetscFunctionBegin;
  PetscValidPointer(isInitialized,1);
  *isInitialized = SlepcInitializeCalled;
  PetscFunctionReturn(0);
}

PETSC_EXTERN PetscBool PetscBeganMPI;

#undef __FUNCT__
#define __FUNCT__ "SlepcInitializeNoPointers"
/*
   SlepcInitializeNoPointers - Calls SlepcInitialize() from C/C++ without the pointers
   to argc and args (analogue to PetscInitializeNoPointers).

   Collective

   Level: advanced

.seealso: SlepcInitialize()
*/
PetscErrorCode SlepcInitializeNoPointers(int argc,char **args,const char *filename,const char *help)
{
  PetscErrorCode ierr;
  int            myargc = argc;
  char           **myargs = args;

  PetscFunctionBegin;
  ierr = SlepcInitialize(&myargc,&myargs,filename,help);CHKERRQ(ierr);
  ierr = PetscPopSignalHandler();CHKERRQ(ierr);
  PetscBeganMPI = PETSC_FALSE;
  PetscFunctionReturn(ierr);
}

