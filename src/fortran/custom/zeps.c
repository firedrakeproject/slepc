
#include "src/fortran/custom/zpetsc.h"
#include "slepceps.h"
#include "src/eps/epsimpl.h"

#ifdef PETSC_HAVE_FORTRAN_CAPS
#define epsview_                    EPSVIEW
#define epssetoptionsprefix_        EPSSETOPTIONSPREFIX
#define epsappendoptionsprefix_     EPSAPPENDOPTIONSPREFIX
#define epsgetoptionsprefix_        EPSGETOPTIONSPREFIX
#define epscreate_                  EPSCREATE
#define epssettype_                 EPSSETTYPE           
#define epsgettype_                 EPSGETTYPE
#define epsdefaultestimatesmonitor_ EPSDEFAULTESTIMATESMONITOR
#define epsdefaultvaluesmonitor_    EPSDEFAULTVALUESMONITOR
#define epssetmonitor_              EPSSETMONITOR
#define epssetvaluesmonitor_        EPSSETVALUESMONITOR
#elif !defined(PETSC_HAVE_FORTRAN_UNDERSCORE)
#define epsview_                    epsview
#define epssetoptionsprefix_        ekepsview
#define epssetoptionsprefix_        epssetoptionsprefix
#define epsappendoptionsprefix_     epsappendoptionsprefix
#define epsgetoptionsprefix_        epsgetoptionsprefix
#define epscreate_                  epscreate
#define epssettype_                 epssettype           
#define epsgettype_                 epsgettype
#define epsdefaultestimatesmonitor_ epsdefaultestimatesmonitor
#define epsdefaultvaluesmonitor_    epsdefaultvaluesmonitor
#define epssetmonitor_              epssetmonitor
#define epssetvaluesmonitor_        epssetvaluesmonitor
#endif

EXTERN_C_BEGIN

void PETSC_STDCALL epsview_(EPS *eps,PetscViewer *viewer, int *ierr)
{
  PetscViewer v;
  PetscPatchDefaultViewers_Fortran(viewer,v);
  *ierr = EPSView(*eps,v);
}

void PETSC_STDCALL epssettype_(EPS *eps,CHAR type PETSC_MIXED_LEN(len),int *ierr PETSC_END_LEN(len))
{
  char *t;

  FIXCHAR(type,len,t);
  *ierr = EPSSetType(*eps,t);
  FREECHAR(type,t);
}

void PETSC_STDCALL epsgettype_(EPS *eps,CHAR name PETSC_MIXED_LEN(len),int *ierr PETSC_END_LEN(len))
{
  char *tname;

  *ierr = EPSGetType(*eps,&tname);if (*ierr) return;
#if defined(PETSC_USES_CPTOFCD)
  {
    char *t = _fcdtocp(name); int len1 = _fcdlen(name);
    *ierr = PetscStrncpy(t,tname,len1); 
  }
#else
  *ierr = PetscStrncpy(name,tname,len);
#endif
}

void PETSC_STDCALL epssetoptionsprefix_(EPS *eps,CHAR prefix PETSC_MIXED_LEN(len),
                                        int *ierr PETSC_END_LEN(len))
{
  char *t;

  FIXCHAR(prefix,len,t);
  *ierr = EPSSetOptionsPrefix(*eps,t);
  FREECHAR(prefix,t);
}

void PETSC_STDCALL epsappendoptionsprefix_(EPS *eps,CHAR prefix PETSC_MIXED_LEN(len),
                                           int *ierr PETSC_END_LEN(len))
{
  char *t;

  FIXCHAR(prefix,len,t);
  *ierr = EPSAppendOptionsPrefix(*eps,t);
  FREECHAR(prefix,t);
}

void PETSC_STDCALL epscreate_(MPI_Comm *comm,EPS *eps,int *ierr){
  *ierr = EPSCreate((MPI_Comm)PetscToPointerComm(*comm),eps);
}

/*
        These are not usually called from Fortran but allow Fortran users 
   to transparently set these monitors from .F code
   
   functions, hence no STDCALL
*/
void  epsdefaultestimatesmonitor_(EPS *eps,int *it,int *nconv,PetscReal *errest,int *nest,void *ctx,int *ierr)
{
  *ierr = EPSDefaultEstimatesMonitor(*eps,*it,*nconv,errest,*nest,ctx);
}
 
void  epsdefaultvaluesmonitor_(EPS *eps,int *it,int *nconv,PetscScalar *eigr,PetscScalar *eigi,int *neig,void *ctx,int *ierr)
{
  *ierr = EPSDefaultValuesMonitor(*eps,*it,*nconv,eigr,eigi,*neig,ctx);
}
 
static void (PETSC_STDCALL *f1)(EPS*,int*,int*,PetscReal*,int*,void*,int*);
static int ourmonitor(EPS eps,int i,int nc,PetscReal *d,int l,void* ctx)
{
  int ierr = 0;
  (*f1)(&eps,&i,&nc,d,&l,ctx,&ierr);CHKERRQ(ierr);
  return 0;
}

void PETSC_STDCALL epssetmonitor_(EPS *eps,void (PETSC_STDCALL *monitor)(EPS*,int*,int*,PetscReal*,int*,void*,int*),
                    void *mctx,void (PETSC_STDCALL *monitordestroy)(void *,int *),int *ierr)
{
  if ((void(*)())monitor == (void(*)())epsdefaultestimatesmonitor_) {
    *ierr = EPSSetMonitor(*eps,EPSDefaultEstimatesMonitor,0);
  } else {
    f1  = monitor;
    if (FORTRANNULLFUNCTION(monitordestroy)) {
      *ierr = EPSSetMonitor(*eps,ourmonitor,mctx);
    } else {
      *ierr = EPSSetMonitor(*eps,ourmonitor,mctx);
    }
  }
}

static void (PETSC_STDCALL *f3)(EPS*,int*,int*,PetscScalar*,PetscScalar*,int*,void*,int*);
static int ourmonitor2(EPS eps,int i,int nc,PetscScalar *d1,PetscScalar *d2,int l,void* ctx)
{
  int ierr = 0;
  (*f3)(&eps,&i,&nc,d1,d2,&l,ctx,&ierr);CHKERRQ(ierr);
  return 0;
}

void PETSC_STDCALL epssetvaluesmonitor_(EPS *eps,void (PETSC_STDCALL *monitor)(EPS*,int*,int*,PetscScalar*,PetscScalar*,int*,void*,int*),
                    void *mctx,void (PETSC_STDCALL *monitordestroy)(void *,int *),int *ierr)
{
  if ((void(*)())monitor == (void(*)())epsdefaultvaluesmonitor_) {
    *ierr = EPSSetValuesMonitor(*eps,EPSDefaultValuesMonitor,0);
  } else {
    f3  = monitor;
    if (FORTRANNULLFUNCTION(monitordestroy)) {
      *ierr = EPSSetValuesMonitor(*eps,ourmonitor2,mctx);
    } else {
      *ierr = EPSSetValuesMonitor(*eps,ourmonitor2,mctx);
    }
  }
}

void PETSC_STDCALL epsgetoptionsprefix_(EPS *eps,CHAR prefix PETSC_MIXED_LEN(len),int *ierr PETSC_END_LEN(len))
{
  char *tname;

  *ierr = EPSGetOptionsPrefix(*eps,&tname);
#if defined(PETSC_USES_CPTOFCD)
  {
    char *t = _fcdtocp(prefix); int len1 = _fcdlen(prefix);
    *ierr = PetscStrncpy(t,tname,len1); if (*ierr) return;
  }
#else
  *ierr = PetscStrncpy(prefix,tname,len); if (*ierr) return;
#endif
}

EXTERN_C_END

