#include "zpetsc.h"
#include "slepceps.h"

#ifdef PETSC_HAVE_FORTRAN_CAPS
#define epsprimmegetmethod_  EPSPRIMMEGETMETHOD
#define epsprimmegetrestart_ EPSPRIMMEGETRESTART
#define epsprimmegetprecond_ EPSPRIMMEGETPRECOND
#elif !defined(PETSC_HAVE_FORTRAN_UNDERSCORE)
#define epsprimmegetmethod_  epsprimmegetmethod
#define epsprimmegetrestart_ epsprimmegetrestart
#define epsprimmegetprecond_ epsprimmegetprecond
#endif

EXTERN_C_BEGIN

void PETSC_STDCALL  epsprimmegetmethod_(EPS *eps,EPSPRIMMEMethod *method, int *__ierr ){
  *__ierr = EPSPRIMMEGetMethod(*eps,method);
}

void PETSC_STDCALL  epsprimmegetrestart_(EPS *eps,EPSPRIMMERestart *scheme, int *__ierr ){
  *__ierr = EPSPRIMMEGetRestart(*eps,scheme);
}

void PETSC_STDCALL  epsprimmegetprecond_(EPS *eps,EPSPRIMMEPrecond *precond, int *__ierr ){
  *__ierr = EPSPRIMMEGetPrecond(*eps,precond);
}

EXTERN_C_END

