
#include "src/fortran/custom/zpetsc.h"
#include "slepcst.h"

#ifdef PETSC_HAVE_FORTRAN_CAPS
#define stsettype_                STSETTYPE           
#define stregisterdestroy_        STREGISTERDESTROY
#define stgettype_                STGETTYPE
#define stdestroy_                STDESTROY
#define stcreate_                 STCREATE
#define stgetoperators_           STGETOPERATORS
#define stsetoptionsprefix_       STSETOPTIONSPREFIX
#define stappendoptionsprefix_    STAPPENDOPTIONSPREFIX
#define stgetoptionsprefix_       STGETOPTIONSPREFIX
#define stview_                   STVIEW
#elif !defined(PETSC_HAVE_FORTRAN_UNDERSCORE)
#define stsettype_                stsettype
#define stregisterdestroy_        stregisterdestroy
#define stgettype_                stgettype
#define stdestroy_                stdestroy
#define stcreate_                 stcreate
#define stgetoperators_           stgetoperators
#define stsetoptionsprefix_       stsetoptionsprefix
#define stappendoptionsprefix_    stappendoptionsprefix
#define stgetoptionsprefix_       stgetoptionsprefix
#define stview_                   stview
#endif

EXTERN_C_BEGIN

void PETSC_STDCALL stsettype_(ST *st,CHAR type PETSC_MIXED_LEN(len),int *ierr PETSC_END_LEN(len))
{
  char *t;

  FIXCHAR(type,len,t);
  *ierr = STSetType(*st,t);
  FREECHAR(type,t);
}

void PETSC_STDCALL stregisterdestroy_(int *ierr)
{
  *ierr = STRegisterDestroy();
}

void PETSC_STDCALL stgettype_(ST *st,CHAR name PETSC_MIXED_LEN(len),int *ierr PETSC_END_LEN(len))
{
  char *tname;

  *ierr = STGetType(*st,&tname);
#if defined(PETSC_USES_CPTOFCD)
  {
  char *t = _fcdtocp(name); int len1 = _fcdlen(name);
  *ierr = PetscStrncpy(t,tname,len1); if (*ierr) return;
  }
#else
  *ierr = PetscStrncpy(name,tname,len);if (*ierr) return;
#endif
}

void PETSC_STDCALL stdestroy_(ST *st,int *ierr)
{
  *ierr = STDestroy(*st);
}

void PETSC_STDCALL stcreate_(MPI_Comm *comm,ST *newst,int *ierr)
{
  *ierr = STCreate((MPI_Comm)PetscToPointerComm(*comm),newst);
}

void PETSC_STDCALL stgetoperators_(ST *st,Mat *mat,Mat *pmat,int *ierr)
{
  if (FORTRANNULLOBJECT(mat))   mat = PETSC_NULL;
  if (FORTRANNULLOBJECT(pmat))  pmat = PETSC_NULL;
  *ierr = STGetOperators(*st,mat,pmat);
}

void PETSC_STDCALL stsetoptionsprefix_(ST *st,CHAR prefix PETSC_MIXED_LEN(len),
                                       int *ierr PETSC_END_LEN(len))
{
  char *t;

  FIXCHAR(prefix,len,t);
  *ierr = STSetOptionsPrefix(*st,t);
  FREECHAR(prefix,t);
}

void PETSC_STDCALL stappendoptionsprefix_(ST *st,CHAR prefix PETSC_MIXED_LEN(len),
                                          int *ierr PETSC_END_LEN(len))
{
  char *t;

  FIXCHAR(prefix,len,t);
  *ierr = STAppendOptionsPrefix(*st,t);
  FREECHAR(prefix,t);
}

void PETSC_STDCALL stgetoptionsprefix_(ST *st,CHAR prefix PETSC_MIXED_LEN(len),
                                       int *ierr PETSC_END_LEN(len))
{
  char *tname;

  *ierr = STGetOptionsPrefix(*st,&tname);
#if defined(PETSC_USES_CPTOFCD)
  {
    char *t = _fcdtocp(prefix); int len1 = _fcdlen(prefix);
    *ierr = PetscStrncpy(t,tname,len1);if (*ierr) return;
  }
#else
  *ierr = PetscStrncpy(prefix,tname,len);if (*ierr) return;
#endif
}

void PETSC_STDCALL stview_(ST *st,PetscViewer *viewer, int *ierr)
{
  PetscViewer v;
  PetscPatchDefaultViewers_Fortran(viewer,v);
  *ierr = STView(*st,v);
}

EXTERN_C_END

