/*
   Private data structure used by the ARPACK interface
*/

#if !defined(__ARPACKP_H)
#define __ARPACKP_H

#include "src/eps/epsimpl.h"

typedef struct {
  PetscTruth  *select;
  PetscScalar *workev;
  PetscScalar *workd;
  PetscScalar *workl;
  int         lworkl;
#if defined(PETSC_USE_COMPLEX)
  PetscReal  *rwork;
#endif
} EPS_ARPACK;

/*
   Definition of routines from the ARPACK package
*/

#include "slepcblaslapack.h"

#if !defined(_petsc_mpi_uni)

#if !defined(PETSC_USE_COMPLEX)

/*
    These are real case 
*/

#if defined(PETSC_USES_FORTRAN_SINGLE) 
/*
   For these machines we must call the single precision Fortran version
*/
#define ARnaupd_ SLEPC_FORTRAN(psnaupd,PSNAUPD)
#define ARneupd_ SLEPC_FORTRAN(psneupd,PSNEUPD)
#define ARsaupd_ SLEPC_FORTRAN(pssaupd,PSSAUPD)
#define ARseupd_ SLEPC_FORTRAN(psseupd,PSSEUPD)

#else

#define ARnaupd_ SLEPC_FORTRAN(pdnaupd,PDNAUPD)
#define ARneupd_ SLEPC_FORTRAN(pdneupd,PDNEUPD)
#define ARsaupd_ SLEPC_FORTRAN(pdsaupd,PDSAUPD)
#define ARseupd_ SLEPC_FORTRAN(pdseupd,PDSEUPD)

#endif

#else
/*
   Complex 
*/
#if defined(PETSC_USES_FORTRAN_SINGLE) 

#define ARnaupd_ SLEPC_FORTRAN(pcnaupd,PCNAUPD)
#define ARneupd_ SLEPC_FORTRAN(pcneupd,PCNEUPD)

#else

#define ARnaupd_ SLEPC_FORTRAN(pznaupd,PZNAUPD)
#define ARneupd_ SLEPC_FORTRAN(pzneupd,PZNEUPD)

#endif

#endif

#else
/* _petsc_mpi_uni */

#if !defined(PETSC_USE_COMPLEX)

/*
    These are real case 
*/

#if defined(PETSC_USES_FORTRAN_SINGLE) 
/*
   For these machines we must call the single precision Fortran version
*/
#define ARnaupd__ SLEPC_FORTRAN(snaupd,SNAUPD)
#define ARneupd__ SLEPC_FORTRAN(sneupd,SNEUPD)
#define ARsaupd__ SLEPC_FORTRAN(ssaupd,SSAUPD)
#define ARseupd__ SLEPC_FORTRAN(sseupd,SSEUPD)

#else

#define ARnaupd__ SLEPC_FORTRAN(dnaupd,DNAUPD)
#define ARneupd__ SLEPC_FORTRAN(dneupd,DNEUPD)
#define ARsaupd__ SLEPC_FORTRAN(dsaupd,DSAUPD)
#define ARseupd__ SLEPC_FORTRAN(dseupd,DSEUPD)

#endif

#else
/*
   Complex 
*/
#if defined(PETSC_USES_FORTRAN_SINGLE) 

#define ARnaupd__ SLEPC_FORTRAN(cnaupd,CNAUPD)
#define ARneupd__ SLEPC_FORTRAN(cneupd,CNEUPD)

#else

#define ARnaupd__ SLEPC_FORTRAN(znaupd,ZNAUPD)
#define ARneupd__ SLEPC_FORTRAN(zneupd,ZNEUPD)

#endif

#endif

#endif

EXTERN_C_BEGIN

#if !defined(_petsc_mpi_uni)

extern void   ARsaupd_(MPI_Fint*,int*,char*,int*,char*,int*,PetscReal*,PetscScalar*,
                       int*,PetscScalar*,int*,int*,int*,PetscScalar*,PetscScalar*,int*,int*,int,int);
extern void   ARseupd_(MPI_Fint*,PetscTruth*,char*,PetscTruth*,PetscReal*,PetscReal*,
                       int*,PetscReal*,
                       char*,int*,char*,int*,PetscReal*,PetscScalar*,
                       int*,PetscScalar*,int*,int*,int*,PetscScalar*,PetscScalar*,int*,int*,int,int,int);

#if !defined(PETSC_USE_COMPLEX)
extern void   ARnaupd_(MPI_Fint*,int*,char*,int*,char*,int*,PetscReal*,PetscScalar*,
                       int*,PetscScalar*,int*,int*,int*,PetscScalar*,PetscScalar*,int*,int*,int,int);
extern void   ARneupd_(MPI_Fint*,PetscTruth*,char*,PetscTruth*,PetscReal*,PetscReal*,
                       PetscReal*,int*,PetscReal*,PetscReal*,PetscReal*,
                       char*,int*,char*,int*,PetscReal*,PetscScalar*,
                       int*,PetscScalar*,int*,int*,int*,PetscScalar*,PetscScalar*,int*,int*,int,int,int);
#else
extern void   ARnaupd_(MPI_Fint*,int*,char*,int*,char*,int*,PetscReal*,PetscScalar*,
                       int*,PetscScalar*,int*,int*,int*,PetscScalar*,PetscScalar*,int*,PetscReal*,int*,
                       int,int);
extern void   ARneupd_(MPI_Fint*,PetscTruth*,char*,PetscTruth*,PetscScalar*,PetscScalar*,
                       int*,PetscScalar*,PetscScalar*,
                       char*,int*,char*,int*,PetscReal*,PetscScalar*,
                       int*,PetscScalar*,int*,int*,int*,PetscScalar*,PetscScalar*,int*,PetscReal*,int*,
                       int,int,int);
#endif

#else
/* _petsc_mpi_uni */

extern void   ARsaupd__(int*,char*,int*,char*,int*,PetscReal*,PetscScalar*,
                       int*,PetscScalar*,int*,int*,int*,PetscScalar*,PetscScalar*,int*,int*,int,int);
#define ARsaupd_(a,b,c,d,e,f,g,h,i,j,k,l,m,n,o,p,q,r,s) ARsaupd__(b,c,d,e,f,g,h,i,j,k,l,m,n,o,p,q,r,s) 
extern void   ARseupd__(PetscTruth*,char*,PetscTruth*,PetscReal*,PetscReal*,
                       int*,PetscReal*,
                       char*,int*,char*,int*,PetscReal*,PetscScalar*,
                       int*,PetscScalar*,int*,int*,int*,PetscScalar*,PetscScalar*,int*,int*,int,int,int);
#define ARseupd_(a,b,c,d,e,f,g,h,i,j,k,l,m,n,o,p,q,r,s,t,u,v,w,x,y,z) ARseupd__(b,c,d,e,f,g,h,i,j,k,l,m,n,o,p,q,r,s,t,u,v,w,x,y,z) 

#if !defined(PETSC_USE_COMPLEX)
extern void   ARnaupd__(int*,char*,int*,char*,int*,PetscReal*,PetscScalar*,
                       int*,PetscScalar*,int*,int*,int*,PetscScalar*,PetscScalar*,int*,int*,int,int);
#define ARnaupd_(a,b,c,d,e,f,g,h,i,j,k,l,m,n,o,p,q,r,s) ARnaupd__(b,c,d,e,f,g,h,i,j,k,l,m,n,o,p,q,r,s) 
extern void   ARneupd__(PetscTruth*,char*,PetscTruth*,PetscReal*,PetscReal*,
                       PetscReal*,int*,PetscReal*,PetscReal*,PetscReal*,
                       char*,int*,char*,int*,PetscReal*,PetscScalar*,
                       int*,PetscScalar*,int*,int*,int*,PetscScalar*,PetscScalar*,int*,int*,int,int,int);
#define ARneupd_(a,b,c,d,e,f,g,h,i,j,k,l,m,n,o,p,q,r,s,t,u,v,w,x,y,z,a1,a2,a3) ARneupd__(b,c,d,e,f,g,h,i,j,k,l,m,n,o,p,q,r,s,t,u,v,w,x,y,z,a1,a2,a3) 
#else
extern void   ARnaupd__(int*,char*,int*,char*,int*,PetscReal*,PetscScalar*,
                       int*,PetscScalar*,int*,int*,int*,PetscScalar*,PetscScalar*,int*,PetscReal*,int*,
                       int,int);
#define ARnaupd_(a,b,c,d,e,f,g,h,i,j,k,l,m,n,o,p,q,r,s,t) ARnaupd__(b,c,d,e,f,g,h,i,j,k,l,m,n,o,p,q,r,s,t) 
extern void   ARneupd__(PetscTruth*,char*,PetscTruth*,PetscScalar*,PetscScalar*,
                       int*,PetscScalar*,PetscScalar*,
                       char*,int*,char*,int*,PetscReal*,PetscScalar*,
                       int*,PetscScalar*,int*,int*,int*,PetscScalar*,PetscScalar*,int*,PetscReal*,int*,
                       int,int,int);
#define ARneupd_(a,b,c,d,e,f,g,h,i,j,k,l,m,n,o,p,q,r,s,t,u,v,w,x,y,z,a1,a2) ARneupd__(b,c,d,e,f,g,h,i,j,k,l,m,n,o,p,q,r,s,t,u,v,w,x,y,z,a1,a2) 
#endif

#endif

EXTERN_C_END

#endif

