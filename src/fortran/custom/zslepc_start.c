/*
  This file contains the Fortran version of SlepcInitialize().
*/

/*
    This is to prevent the Cray T3D version of MPI (University of Edinburgh)
  from redefining MPI_INIT(). They put this in to detect errors in C code,
  but here we do want to be calling the Fortran version from a C subroutine. 
*/
#define T3DMPI_FORTRAN
#define T3EMPI_FORTRAN

#include "src/fortran/custom/zpetsc.h" 
#include "slepc.h"

extern PetscTruth SlepcBeganPetsc;
extern PetscRandom rctx;
extern int ST_COOKIE;
extern int EPS_COOKIE;

static PetscTruth SlepcInitializeCalled=PETSC_FALSE;

#if defined(PETSC_HAVE_NAGF90)
#define iargc_  f90_unix_MP_iargc
#define getarg_ f90_unix_MP_getarg
#endif

#ifdef PETSC_HAVE_FORTRAN_CAPS
#define petscinitialize_              PETSCINITIALIZE
#define slepcinitialize_              SLEPCINITIALIZE
#define iargc_                        IARGC
#define getarg_                       GETARG
#if defined(PARCH_win32)
#define IARGC                         NARGS
#endif

#elif !defined(PETSC_HAVE_FORTRAN_UNDERSCORE)
#define petscinitialize_              petscinitialize
#define slepcinitialize_              slepcinitialize
/*
    HP-UX does not have Fortran underscore but iargc and getarg 
  do have underscores????
*/
#if !defined(PETSC_HAVE_FORTRAN_IARGC_UNDERSCORE)
#define iargc_                        iargc
#define getarg_                       getarg
#endif

#endif

EXTERN_C_BEGIN
extern void PETSC_STDCALL mpi_init_(int*);

#if defined(PETSC_USE_FORTRAN_MIXED_STR_ARG)
extern void petscinitialize_(CHAR,int,int*);
#else
extern void petscinitialize_(CHAR,int*,int);
#endif

/*
     Different Fortran compilers handle command lines in different ways
*/
#if defined(PARCH_win32)
/*
extern short  __declspec(dllimport) __stdcall iargc_();
extern void __declspec(dllimport) __stdcall  getarg_(short*,char*,int,short *);
*/
extern short __stdcall iargc_();
extern void __stdcall  getarg_(short*,char*,int,short *);

#else
extern int  iargc_();
extern void getarg_(int*,char*,int);
/*
      The Cray T3D/T3E use the PXFGETARG() function
*/
#if defined(PETSC_HAVE_PXFGETARG)
extern void PXFGETARG(int *,_fcd,int*,int*);
#endif
#endif
EXTERN_C_END

extern int SlepcRegisterEvents();

EXTERN_C_BEGIN
/*
    SlepcInitialize - Version called from Fortran.

    Notes:
    Since this routine is called from Fortran it does not return error codes.
*/
void PETSC_STDCALL slepcinitialize_(CHAR filename PETSC_MIXED_LEN(len),int *ierr PETSC_END_LEN(len))
{
  *ierr = 1;
  if (SlepcInitializeCalled) {*ierr = 0; return;}

  if (!PetscInitializeCalled) {
#if defined(PETSC_USE_FORTRAN_MIXED_STR_ARG)
    petscinitialize_(filename,len,ierr);
#else
    petscinitialize_(filename,ierr,len);
#endif
    if (*ierr) return;
    SlepcBeganPetsc = PETSC_TRUE;
  }

  PetscRandomCreate(PETSC_COMM_WORLD,RANDOM_DEFAULT,&rctx);

  EPS_COOKIE = 0;
  PetscLogClassRegister(&EPS_COOKIE,"Eigenproblem Solver");
  ST_COOKIE = 0;
  PetscLogClassRegister(&ST_COOKIE,"Spectral Transform");

  *ierr = SlepcRegisterEvents(); 

  SlepcInitializeCalled = PETSC_TRUE;
  PetscLogInfo(0,"SlepcInitialize: SLEPc successfully started from Fortran\n");

}  

EXTERN_C_END
