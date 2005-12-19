
#include "zpetsc.h" 

#ifdef PETSC_HAVE_FORTRAN_CAPS
#define slepcinitializefortran_     SLEPCINITIALIZEFORTRAN
#elif !defined(PETSC_HAVE_FORTRAN_UNDERSCORE)
#define slepcinitializefortran_     slepcinitializefortran
#endif

/*@C
   SlepcInitializeFortran - Routine that should be called from C after
   the call to SlepcInitialize() if one is using a C main program
   that calls Fortran routines that in turn call SLEPc routines.

   Collective on PETSC_COMM_WORLD

   Level: beginner

   Notes:
   SlepcInitializeFortran() initializes some of the default SLEPc variables
   for use in Fortran if a user's main program is written in C.  
   SlepcInitializeFortran() is NOT needed if a user's main
   program is written in Fortran; in this case, just calling
   SlepcInitialize() in the main (Fortran) program is sufficient.

.seealso:  SlepcInitialize()

@*/

PetscErrorCode SlepcInitializeFortran(void)
{
  PetscInitializeFortran();
  return 0;
}
  
EXTERN_C_BEGIN

void PETSC_STDCALL slepcinitializefortran_(int *info)
{
  *info = SlepcInitializeFortran();
}

EXTERN_C_END

