
#include "src/fortran/custom/zpetsc.h" 
/* #include "sys.h" */
#include "petscsys.h"

#ifdef PETSC_HAVE_FORTRAN_CAPS
#define slepcinitializefortran_     SLEPCINITIALIZEFORTRAN
#define slepcsetcommonblock_        SLEPCSETCOMMONBLOCK
#define slepc_null_function_        SLEPC_NULL_FUNCTION
#elif !defined(PETSC_HAVE_FORTRAN_UNDERSCORE)
#define slepcinitializefortran_     slepcinitializefortran
#define slepcsetcommonblock_        slepcsetcommonblock
#define slepc_null_function_        slepc_null_function
#endif

#if defined(PETSC_HAVE_FORTRAN_UNDERSCORE_UNDERSCORE)
#define slepc_null_function_  slepc_null_function__
#endif

EXTERN_C_BEGIN
extern void PETSC_STDCALL slepcsetcommonblock_(void);
EXTERN_C_END

/*@C
   SlepcInitializeFortran - Routine that should be called from C after
   the call to SlepcInitialize() if one is using a C main program
   that calls Fortran routines that in turn call SLEPc routines.

   Collective on MPI_COMM_WORLD

   Level: beginner

   Notes:
   SlepcInitializeFortran() initializes some of the default SLEPc variables
   for use in Fortran if a user's main program is written in C.  
   SlepcInitializeFortran() is NOT needed if a user's main
   program is written in Fortran; in this case, just calling
   SlepcInitialize() in the main (Fortran) program is sufficient.

.seealso:  SlepcInitialize()

@*/

int SlepcInitializeFortran(void)
{
  slepcsetcommonblock_();
  return 0;
}
  
EXTERN_C_BEGIN

void PETSC_STDCALL slepcinitializefortran_(int *info)
{
  *info = SlepcInitializeFortran();
}

/*
  A valid address for the Fortran variable SLEPC_NULL_FUNCTION
*/
void slepc_null_function_(void)
{
  return;
}

EXTERN_C_END

