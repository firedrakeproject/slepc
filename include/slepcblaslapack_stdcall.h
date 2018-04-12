/*
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
   SLEPc - Scalable Library for Eigenvalue Problem Computations
   Copyright (c) 2002-2018, Universitat Politecnica de Valencia, Spain

   This file is part of SLEPc.
   SLEPc is distributed under a 2-clause BSD license (see LICENSE).
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
*/

#if !defined(__SLEPCBLASLAPACK_STDCALL_H)
#define __SLEPCBLASLAPACK_STDCALL_H

/* LAPACK functions without string parameters */
#define BLASrot_     PETSCBLAS(rot,ROT)
#define BLASMIXEDrot_ PETSCBLASMIXED(rot,ROT)
#define LAPACKlaev2_ PETSCBLAS(laev2,LAEV2)
#define LAPACKgehrd_ PETSCBLAS(gehrd,GEHRD)
#define LAPACKgelqf_ PETSCBLAS(gelqf,GELQF)
#define LAPACKlarfg_ PETSCBLAS(larfg,LARFG)
#define LAPACKlag2_  PETSCBLASREAL(lag2,LAG2)
#define LAPACKlasv2_ PETSCBLASREAL(lasv2,LASV2)
#define LAPACKlartg_ PETSCBLAS(lartg,LARTG)
#define LAPACKREALlartg_ PETSCBLASREAL(lartg,LARTG)
#define LAPACKlaln2_ PETSCBLASREAL(laln2,LALN2)
#define LAPACKlaed4_ PETSCBLASREAL(laed4,LAED4)
#define LAPACKlamrg_ PETSCBLASREAL(lamrg,LAMRG)
#define LAPACKlapy2_ PETSCBLASREAL(lapy2,LAPY2)
#if !defined(PETSC_USE_COMPLEX)
#define LAPACKorghr_ PETSCBLAS(orghr,ORGHR)
#else
#define LAPACKorghr_ PETSCBLAS(unghr,UNGHR)
#endif
#define LAPACKtgexc_ PETSCBLAS(tgexc,TGEXC)
#define LAPACKgeqp3_ PETSCBLAS(geqp3,GEQP3)

PETSC_EXTERN void     PETSC_STDCALL PETSCBLASR(rot,ROT) (PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscReal*,PetscReal*);
PETSC_EXTERN void     PETSC_STDCALL PETSCBLAS(laev2,LAEV2) (PetscScalar*,PetscScalar*,PetscScalar*,PetscReal*,PetscReal*,PetscReal*,PetscScalar*);
PETSC_EXTERN void     PETSC_STDCALL PETSCBLAS(gehrd,GEHRD) (PetscBLASInt*,PetscBLASInt*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscScalar*,PetscScalar*,PetscBLASInt*,PetscBLASInt*);
PETSC_EXTERN void     PETSC_STDCALL PETSCBLAS(gelqf,GELQF) (PetscBLASInt*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscScalar*,PetscScalar*,PetscBLASInt*,PetscBLASInt*);
PETSC_EXTERN void     PETSC_STDCALL PETSCBLAS(larfg,LARFG) (PetscBLASInt*,PetscScalar*,PetscScalar*,PetscBLASInt*,PetscScalar*);
PETSC_EXTERN void     PETSC_STDCALL PETSCBLASREAL(lag2,LAG2) (PetscReal*,PetscBLASInt*,PetscReal*,PetscBLASInt*,PetscReal*,PetscReal*,PetscReal*,PetscReal*,PetscReal*,PetscReal*);
PETSC_EXTERN void     PETSC_STDCALL PETSCBLASREAL(lasv2,LASV2) (PetscReal*,PetscReal*,PetscReal*,PetscReal*,PetscReal*,PetscReal*,PetscReal*,PetscReal*,PetscReal*);
PETSC_EXTERN void     PETSC_STDCALL PETSCBLASREAL(lartg,LARTG) (PetscReal*,PetscReal*,PetscReal*,PetscReal*,PetscReal*);
PETSC_EXTERN void     PETSC_STDCALL PETSCBLASREAL(laln2,LALN2) (PetscBLASInt*,PetscBLASInt*,PetscBLASInt*,PetscReal*,PetscReal*,PetscReal*,PetscBLASInt*,PetscReal*,PetscReal*,PetscReal*,PetscBLASInt*,PetscReal*,PetscReal*,PetscReal*,PetscBLASInt*,PetscReal*,PetscReal*,PetscBLASInt*);
PETSC_EXTERN void     PETSC_STDCALL PETSCBLASREAL(laed4,LAED4) (PetscBLASInt*,PetscBLASInt*,PetscReal*,PetscReal*,PetscReal*,PetscReal*,PetscReal*,PetscBLASInt*);
PETSC_EXTERN void     PETSC_STDCALL PETSCBLASREAL(lamrg,LAMRG) (PetscBLASInt*,PetscBLASInt*,PetscReal*,PetscBLASInt*,PetscBLASInt*,PetscBLASInt*);
PETSC_EXTERN SlepcLRT PETSC_STDCALL PETSCBLASREAL(lapy2,LAPY2) (PetscReal*,PetscReal*);
#if !defined(PETSC_USE_COMPLEX)
PETSC_EXTERN void     PETSC_STDCALL PETSCBLAS(orghr,ORGHR) (PetscBLASInt*,PetscBLASInt*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscScalar*,PetscScalar*,PetscBLASInt*,PetscBLASInt*);
PETSC_EXTERN void     PETSC_STDCALL PETSCBLAS(tgexc,TGEXC) (PetscBLASInt*,PetscBLASInt*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscBLASInt*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscBLASInt*);
PETSC_EXTERN void     PETSC_STDCALL PETSCBLAS(geqp3,GEQP3) (PetscBLASInt*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscBLASInt*,PetscScalar*,PetscScalar*,PetscBLASInt*,PetscBLASInt*);
#else
PETSC_EXTERN void     PETSC_STDCALL PETSCBLAS(unghr,UNGHR) (PetscBLASInt*,PetscBLASInt*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscScalar*,PetscScalar*,PetscBLASInt*,PetscBLASInt*);
PETSC_EXTERN void     PETSC_STDCALL PETSCBLAS(tgexc,TGEXC) (PetscBLASInt*,PetscBLASInt*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscBLASInt*,PetscBLASInt*,PetscBLASInt*);
PETSC_EXTERN void     PETSC_STDCALL PETSCBLAS(geqp3,GEQP3) (PetscBLASInt*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscBLASInt*,PetscScalar*,PetscScalar*,PetscBLASInt*,PetscReal*,PetscBLASInt*);
#endif

/* LAPACK functions with string parameters */

/* same name for real and complex */
#define BLAStrmm_(a,b,c,d,e,f,g,h,i,j,k) PETSCBLAS(trmm,TRMM) ((a),1,(b),1,(c),1,(d),1,(e),(f),(g),(h),(i),(j),(k))
#define LAPACKlanhs_(a,b,c,d,e) PETSCBLAS(lanhs,LANHS) ((a),1,(b),(c),(d),(e))
#define LAPACKlange_(a,b,c,d,e,f) PETSCBLAS(lange,LANGE) ((a),1,(b),(c),(d),(e),(f))
#define LAPACKpbtrf_(a,b,c,d,e,f) PETSCBLAS(pbtrf,PBTRF) ((a),1,(b),(c),(d),(e),(f))
#define LAPACKlarf_(a,b,c,d,e,f,g,h,i) PETSCBLAS(larf,LARF) ((a),1,(b),(c),(d),(e),(f),(g),(h),(i))
#define LAPACKlacpy_(a,b,c,d,e,f,g) PETSCBLAS(lacpy,LACPY) ((a),1,(b),(c),(d),(e),(f),(g))
#define LAPACKlansy_(a,b,c,d,e,f) PETSCBLAS(lansy,LANSY) ((a),1,(b),1,(c),(d),(e),(f))
#define LAPACKlaset_(a,b,c,d,e,f,g) PETSCBLAS(laset,LASET) ((a),1,(b),(c),(d),(e),(f),(g))
#define LAPACKtrsyl_(a,b,c,d,e,f,g,h,i,j,k,l,m) PETSCBLAS(trsyl,TRSYL) ((a),1,(b),1,(c),(d),(e),(f),(g),(h),(i),(j),(k),(l),(m))
#define LAPACKtrtri_(a,b,c,d,e,f) PETSCBLAS(trtri,TRTRI) ((a),1,(b),1,(c),(d),(e),(f))

PETSC_EXTERN void      PETSC_STDCALL PETSCBLAS(trmm,TRMM) (const char*,PetscBLASInt,const char*,PetscBLASInt,const char*,PetscBLASInt,const char*,PetscBLASInt,PetscBLASInt*,PetscBLASInt*,PetscScalar*,PetscScalar*,PetscBLASInt*,PetscScalar*,PetscBLASInt*);
PETSC_EXTERN SlepcLRT  PETSC_STDCALL PETSCBLAS(lanhs,LANHS) (const char*,PetscBLASInt,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscReal*);
PETSC_EXTERN SlepcLRT  PETSC_STDCALL PETSCBLAS(lange,LANGE) (const char*,PetscBLASInt,PetscBLASInt*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscReal*);
PETSC_EXTERN SlepcLRT  PETSC_STDCALL PETSCBLAS(pbtrf,PBTRF) (const char*,PetscBLASInt,PetscBLASInt*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscBLASInt*);
PETSC_EXTERN void      PETSC_STDCALL PETSCBLAS(larf,LARF) (const char*,PetscBLASInt,PetscBLASInt*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscScalar*,PetscScalar*,PetscBLASInt*,PetscScalar*);
PETSC_EXTERN void      PETSC_STDCALL PETSCBLAS(lacpy,LACPY) (const char*,PetscBLASInt,PetscBLASInt*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscScalar*,PetscBLASInt*);
PETSC_EXTERN SlepcLRT  PETSC_STDCALL PETSCBLAS(lansy,LANSY) (const char*,PetscBLASInt,const char*,PetscBLASInt,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscReal*);
PETSC_EXTERN void      PETSC_STDCALL PETSCBLAS(laset,LASET) (const char*,PetscBLASInt,PetscBLASInt*,PetscBLASInt*,PetscScalar*,PetscScalar*,PetscScalar*,PetscBLASInt*);
PETSC_EXTERN void      PETSC_STDCALL PETSCBLAS(trsyl,TRSYL) (const char*,PetscBLASInt,const char*,PetscBLASInt,PetscBLASInt*,PetscBLASInt*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscReal*,PetscBLASInt*);
PETSC_EXTERN void      PETSC_STDCALL PETSCBLAS(trtri,TRTRI) (const char*,PetscBLASInt,const char*,PetscBLASInt,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscBLASInt*);

/* subroutines in which we use only the real version, do not care whether they have different name */
#define LAPACKstevr_(a,b,c,d,e,f,g,h,i,j,k,l,m,n,o,p,q,r,s,t) PETSCBLASREAL(stevr,STEVR) ((a),1,(b),1,(c),(d),(e),(f),(g),(h),(i),(j),(k),(l),(m),(n),(o),(p),(q),(r),(s),(t))
#define LAPACKbdsdc_(a,b,c,d,e,f,g,h,i,j,k,l,m,n) PETSCBLASREAL(bdsdc,BDSDC) ((a),1,(b),1,(c),(d),(e),(f),(g),(h),(i),(j),(k),(l),(m),(n))
#define LAPACKlamch_(a) PETSCBLASREAL(lamch,LAMCH) ((a),1)
#define LAPACKlamc3_(a,b) PETSCBLASREAL(lamc3,LAMC3) ((a),(b))

PETSC_EXTERN void     PETSC_STDCALL PETSCBLASREAL(stevr,STEVR) (const char*,PetscBLASInt,const char*,PetscBLASInt,PetscBLASInt*,PetscReal*,PetscReal*,PetscReal*,PetscReal*,PetscBLASInt*,PetscBLASInt*,PetscReal*,PetscBLASInt*,PetscReal*,PetscReal*,PetscBLASInt*,PetscBLASInt*,PetscReal*,PetscBLASInt*,PetscBLASInt*,PetscBLASInt*,PetscBLASInt*);
PETSC_EXTERN void     PETSC_STDCALL PETSCBLASREAL(bdsdc,BDSDC) (const char*,PetscBLASInt,const char*,PetscBLASInt,PetscBLASInt*,PetscReal*,PetscReal*,PetscReal*,PetscBLASInt*,PetscReal*,PetscBLASInt*,PetscReal*,PetscBLASInt*,PetscReal*,PetscBLASInt*,PetscBLASInt*);
PETSC_EXTERN SlepcLRT PETSC_STDCALL PETSCBLASREAL(lamch,LAMCH) (const char*,PetscBLASInt);
PETSC_EXTERN SlepcLRT PETSC_STDCALL PETSCBLASREAL(lamc3,LAMC3) (PetscReal*,PetscReal*);

/* subroutines with different name in real/complex */
#if !defined(PETSC_USE_COMPLEX)
#define LAPACKormlq_(a,b,c,d,e,f,g,h,i,j,k,l,m) PETSCBLAS(ormlq,ORMLQ) ((a),1,(b),1,(c),(d),(e),(f),(g),(h),(i),(j),(k),(l),(m))
#define LAPACKorgtr_(a,b,c,d,e,f,g,h) PETSCBLAS(orgtr,ORGTR) ((a),1,(b),(c),(d),(e),(f),(g),(h))
#define LAPACKsytrd_(a,b,c,d,e,f,g,h,i,j) PETSCBLAS(sytrd,SYTRD) ((a),1,(b),(c),(d),(e),(f),(g),(h),(i),(j))
#define LAPACKsyevr_(a,b,c,d,e,f,g,h,i,j,k,l,m,n,o,p,q,r,s,t,u) PETSCBLAS(syevr,SYEVR) ((a),1,(b),1,(c),1,(d),(e),(f),(g),(h),(i),(j),(k),(l),(m),(n),(o),(p),(q),(r),(s),(t),(u))
#define LAPACKsyevd_(a,b,c,d,e,f,g,h,i,j,k) PETSCBLAS(syevd,SYEVD) ((a),1,(b),1,(c),(d),(e),(f),(g),(h),(i),(j),(k))
#define LAPACKsygvd_(a,b,c,d,e,f,g,h,i,j,k,l,m,n)  PETSCBLAS(sygvd,SYGVD) ((a),(b),1,(c),1,(d),(e),(f),(g),(h),(i),(j),(k),(l),(m),(n))
#else
#define LAPACKormlq_(a,b,c,d,e,f,g,h,i,j,k,l,m) PETSCBLAS(unmlq,UNMLQ) ((a),1,(b),1,(c),(d),(e),(f),(g),(h),(i),(j),(k),(l),(m))
#define LAPACKorgtr_(a,b,c,d,e,f,g,h) PETSCBLAS(ungtr,UNGTR) ((a),1,(b),(c),(d),(e),(f),(g),(h))
#define LAPACKsytrd_(a,b,c,d,e,f,g,h,i,j) PETSCBLAS(hetrd,HETRD) ((a),1,(b),(c),(d),(e),(f),(g),(h),(i),(j))
#define LAPACKsyevr_(a,b,c,d,e,f,g,h,i,j,k,l,m,n,o,p,q,r,s,t,u,v,w) PETSCBLAS(heevr,HEEVR) ((a),1,(b),1,(c),1,(d),(e),(f),(g),(h),(i),(j),(k),(l),(m),(n),(o),(p),(q),(r),(s),(t),(u),(v),(w))
#define LAPACKsyevd_(a,b,c,d,e,f,g,h,i,j,k,l,m) PETSCBLAS(heevd,HEEVD) ((a),1,(b),1,(c),(d),(e),(f),(g),(h),(i),(j),(k),(l),(m))
#define LAPACKsygvd_(a,b,c,d,e,f,g,h,i,j,k,l,m,n,o,p) PETSCBLAS(hegvd,HEGVD) ((a),(b),1,(c),1,(d),(e),(f),(g),(h),(i),(j),(k),(l),(m),(n),(o),(p))
#endif

#if !defined(PETSC_USE_COMPLEX)
PETSC_EXTERN void PETSC_STDCALL PETSCBLAS(ormlq,ORMLQ) (const char*,PetscBLASInt,const char*,PetscBLASInt,PetscBLASInt*,PetscBLASInt*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscScalar*,PetscScalar*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscBLASInt*);
PETSC_EXTERN void PETSC_STDCALL PETSCBLAS(orgtr,ORGTR) (const char*,PetscBLASInt,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscScalar*,PetscScalar*,PetscBLASInt*,PetscBLASInt*);
PETSC_EXTERN void PETSC_STDCALL PETSCBLAS(sytrd,SYTRD) (const char*,PetscBLASInt,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscReal*,PetscReal*,PetscScalar*,PetscScalar*,PetscBLASInt*,PetscBLASInt*);
PETSC_EXTERN void PETSC_STDCALL PETSCBLAS(syevr,SYEVR) (const char*,PetscBLASInt,const char*,PetscBLASInt,const char*,PetscBLASInt,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscScalar*,PetscScalar*,PetscBLASInt*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscScalar*,PetscScalar*,PetscBLASInt*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscBLASInt*,PetscBLASInt*,PetscBLASInt*);
PETSC_EXTERN void PETSC_STDCALL PETSCBLAS(syevd,SYEVD) (const char*,PetscBLASInt,const char*,PetscBLASInt,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscScalar*,PetscScalar*,PetscBLASInt*,PetscBLASInt*,PetscBLASInt*,PetscBLASInt*);
PETSC_EXTERN void PETSC_STDCALL PETSCBLAS(sygvd,SYGVD) (PetscBLASInt*,const char*,PetscBLASInt,const char*,PetscBLASInt,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscScalar*,PetscScalar*,PetscBLASInt*,PetscBLASInt*,PetscBLASInt*,PetscBLASInt*);
#else
PETSC_EXTERN void PETSC_STDCALL PETSCBLAS(unmlq,UNMLQ) (const char*,PetscBLASInt,const char*,PetscBLASInt,PetscBLASInt*,PetscBLASInt*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscScalar*,PetscScalar*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscBLASInt*);
PETSC_EXTERN void PETSC_STDCALL PETSCBLAS(ungtr,UNGTR) (const char*,PetscBLASInt,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscScalar*,PetscScalar*,PetscBLASInt*,PetscBLASInt*);
PETSC_EXTERN void PETSC_STDCALL PETSCBLAS(hetrd,HETRD) (const char*,PetscBLASInt,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscReal*,PetscReal*,PetscScalar*,PetscScalar*,PetscBLASInt*,PetscBLASInt*);
PETSC_EXTERN void PETSC_STDCALL PETSCBLAS(heevr,HEEVR) (const char*,PetscBLASInt,const char*,PetscBLASInt,const char*,PetscBLASInt,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscReal*,PetscReal*,PetscBLASInt*,PetscBLASInt*,PetscReal*,PetscBLASInt*,PetscReal*,PetscScalar*,PetscBLASInt*,PetscBLASInt*, PetscScalar*,PetscBLASInt*,PetscReal*,PetscBLASInt*,PetscBLASInt*,PetscBLASInt*,PetscBLASInt*);
PETSC_EXTERN void PETSC_STDCALL PETSCBLAS(heevd,HEEVD) (const char*,PetscBLASInt,const char*,PetscBLASInt,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscReal*,PetscScalar*,PetscBLASInt*,PetscReal*,PetscBLASInt*,PetscBLASInt*,PetscBLASInt*,PetscBLASInt*);
PETSC_EXTERN void PETSC_STDCALL PETSCBLAS(hegvd,HEGVD) (PetscBLASInt*,const char*,PetscBLASInt,const char*,PetscBLASInt,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscReal*,PetscScalar*,PetscBLASInt*,PetscReal*,PetscBLASInt*,PetscBLASInt*,PetscBLASInt*,PetscBLASInt*);
#endif

/* subroutines with different signature in real/complex */
#if !defined(PETSC_USE_COMPLEX)
#define LAPACKggevx_(a,b,c,d,e,f,g,h,i,j,k,l,m,n,o,p,q,r,s,t,u,v,w,x,y,z,aa,ab,ac) PETSCBLAS(ggevx,GGEVX) ((a),1,(b),1,(c),1,(d),1,(e),(f),(g),(h),(i),(j),(k),(l),(m),(n),(o),(p),(q),(r),(s),(t),(u),(v),(w),(x),(y),(z),(aa),(ab),(ac))
#define LAPACKggev_(a,b,c,d,e,f,g,h,i,j,k,l,m,n,o,p,q) PETSCBLAS(ggev,GGEV) ((a),1,(b),1,(c),(d),(e),(f),(g),(h),(i),(j),(k),(l),(m),(n),(o),(p),(q))
#define LAPACKtrevc_(a,b,c,d,e,f,g,h,i,j,k,l,m,n) PETSCBLAS(trevc,TREVC) ((a),1,(b),1,(c),(d),(e),(f),(g),(h),(i),(j),(k),(l),(m),(n))
#define LAPACKgeevx_(a,b,c,d,e,f,g,h,i,j,k,l,m,n,o,p,q,r,s,t,u,v,w) PETSCBLAS(geevx,GEEVX) ((a),1,(b),1,(c),1,(d),1,(e),(f),(g),(h),(i),(j),(k),(l),(m),(n),(o),(p),(q),(r),(s),(t),(u),(v),(w))
#define LAPACKgees_(a,b,c,d,e,f,g,h,i,j,k,l,m,n,o) PETSCBLAS(gees,GEES) ((a),1,(b),1,(c),(d),(e),(f),(g),(h),(i),(j),(k),(l),(m),(n),(o))
#define LAPACKtrexc_(a,b,c,d,e,f,g,h,i,j) PETSCBLAS(trexc,TREXC) ((a),1,(b),(c),(d),(e),(f),(g),(h),(i),(j))
#define LAPACKgesdd_(a,b,c,d,e,f,g,h,i,j,k,l,m,n) PETSCBLAS(gesdd,GESDD) ((a),1,(b),(c),(d),(e),(f),(g),(h),(i),(j),(k),(l),(m),(n))
#define LAPACKtgevc_(a,b,c,d,e,f,g,h,i,j,k,l,m,n,o,p) PETSCBLAS(tgevc,TGEVC) ((a),1,(b),1,(c),(d),(e),(f),(g),(h),(i),(j),(k),(l),(m),(n),(o),(p))
#define LAPACKhsein_(a,b,c,d,e,f,g,h,i,j,k,l,m,n,o,p,q,r,s) PETSCBLAS(hsein,HSEIN) ((a),1,(b),1,(c),1,(d),(e),(f),(g),(h),(i),(j),(k),(l),(m),(n),(o),(p),(q),(r),(s))
#define LAPACKstedc_(a,b,c,d,e,f,g,h,i,j,k) PETSCBLAS(stedc,STEDC) ((a),1,(b),(c),(d),(e),(f),(g),(h),(i),(j),(k))
#define LAPACKlascl_(a,b,c,d,e,f,g,h,i,j) PETSCBLAS(lascl,LASCL) ((a),1,(b),(c),(d),(e),(f),(g),(h),(i),(j))
#else
#define LAPACKggevx_(a,b,c,d,e,f,g,h,i,j,k,l,m,n,o,p,q,r,s,t,u,v,w,x,y,z,aa,ab,ac) PETSCBLAS(ggevx,GGEVX) ((a),1,(b),1,(c),1,(d),1,(e),(f),(g),(h),(i),(j),(k),(l),(m),(n),(o),(p),(q),(r),(s),(t),(u),(v),(w),(x),(y),(z),(aa),(ab),(ac))
#define LAPACKggev_(a,b,c,d,e,f,g,h,i,j,k,l,m,n,o,p,q) PETSCBLAS(ggev,GGEV) ((a),1,(b),1,(c),(d),(e),(f),(g),(h),(i),(j),(k),(l),(m),(n),(o),(p),(q))
#define LAPACKtrevc_(a,b,c,d,e,f,g,h,i,j,k,l,m,n,o) PETSCBLAS(trevc,TREVC) ((a),1,(b),1,(c),(d),(e),(f),(g),(h),(i),(j),(k),(l),(m),(n),(o))
#define LAPACKgeevx_(a,b,c,d,e,f,g,h,i,j,k,l,m,n,o,p,q,r,s,t,u,v) PETSCBLAS(geevx,GEEVX) ((a),1,(b),1,(c),1,(d),1,(e),(f),(g),(h),(i),(j),(k),(l),(m),(n),(o),(p),(q),(r),(s),(t),(u),(v))
#define LAPACKgees_(a,b,c,d,e,f,g,h,i,j,k,l,m,n,o) PETSCBLAS(gees,GEES) ((a),1,(b),1,(c),(d),(e),(f),(g),(h),(i),(j),(k),(l),(m),(n),(o))
#define LAPACKtrexc_(a,b,c,d,e,f,g,h,i) PETSCBLAS(trexc,TREXC) ((a),1,(b),(c),(d),(e),(f),(g),(h),(i))
#define LAPACKgesdd_(a,b,c,d,e,f,g,h,i,j,k,l,m,n,o) PETSCBLAS(gesdd,GESDD) ((a),1,(b),(c),(d),(e),(f),(g),(h),(i),(j),(k),(l),(m),(n),(o))
#define LAPACKtgevc_(a,b,c,d,e,f,g,h,i,j,k,l,m,n,o,p,q) PETSCBLAS(tgevc,TGEVC) ((a),1,(b),1,(c),(d),(e),(f),(g),(h),(i),(j),(k),(l),(m),(n),(o),(p),(q))
#define LAPACKhsein_(a,b,c,d,e,f,g,h,i,j,k,l,m,n,o,p,q,r,s) PETSCBLAS(hsein,HSEIN) ((a),1,(b),1,(c),1,(d),(e),(f),(g),(h),(i),(j),(k),(l),(m),(n),(o),(p),(q),(r),(s))
#define LAPACKstedc_(a,b,c,d,e,f,g,h,i,j,k,l,m) PETSCBLAS(stedc,STEDC) ((a),1,(b),(c),(d),(e),(f),(g),(h),(i),(j),(k),(l),(m))
#define LAPACKlascl_(a,b,c,d,e,f,g,h,i,j) PETSCBLAS(lascl,LASCL) ((a),1,(b),(c),(d),(e),(f),(g),(h),(i),(j))
#endif

#if !defined(PETSC_USE_COMPLEX)
PETSC_EXTERN void PETSC_STDCALL PETSCBLAS(ggevx,GGEVX) (const char*,PetscBLASInt,const char*,PetscBLASInt,const char*,PetscBLASInt,const char*,PetscBLASInt,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscScalar*,PetscScalar*,PetscScalar*,PetscScalar*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscBLASInt*,PetscBLASInt*,PetscScalar*,PetscScalar*,PetscScalar*,PetscScalar*,PetscScalar*,PetscScalar*,PetscScalar*,PetscBLASInt*,PetscBLASInt*,PetscBLASInt*,PetscBLASInt*);
PETSC_EXTERN void PETSC_STDCALL PETSCBLAS(ggev,GGEV) (const char*,PetscBLASInt,const char*,PetscBLASInt,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscScalar*,PetscScalar*,PetscScalar*,PetscScalar*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscBLASInt*);
PETSC_EXTERN void PETSC_STDCALL PETSCBLAS(trevc,TREVC) (const char*,PetscBLASInt,const char*,PetscBLASInt,PetscBLASInt*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscBLASInt*,PetscBLASInt*,PetscScalar*,PetscBLASInt*);
PETSC_EXTERN void PETSC_STDCALL PETSCBLAS(geevx,GEEVX) (const char*,PetscBLASInt,const char*,PetscBLASInt,const char*,PetscBLASInt,const char*,PetscBLASInt,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscScalar*,PetscScalar*,PetscScalar*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscBLASInt*,PetscBLASInt*,PetscScalar*,PetscScalar*,PetscScalar*,PetscScalar*,PetscScalar*,PetscBLASInt*,PetscBLASInt*,PetscBLASInt*);
PETSC_EXTERN void PETSC_STDCALL PETSCBLAS(gees,GEES) (const char*,PetscBLASInt,const char*,PetscBLASInt,PetscBLASInt(*)(PetscReal,PetscReal),PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscBLASInt*,PetscScalar*,PetscScalar*,PetscScalar*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscBLASInt*,PetscBLASInt*);
PETSC_EXTERN void PETSC_STDCALL PETSCBLAS(trexc,TREXC) (const char*,PetscBLASInt,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscBLASInt*,PetscBLASInt*,PetscScalar*,PetscBLASInt*);
PETSC_EXTERN void PETSC_STDCALL PETSCBLAS(gesdd,GESDD) (const char*,PetscBLASInt,PetscBLASInt*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscReal*,PetscScalar*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscBLASInt*,PetscBLASInt*);
PETSC_EXTERN void PETSC_STDCALL PETSCBLAS(tgevc,TGEVC) (const char*,PetscBLASInt,const char*,PetscBLASInt,PetscBLASInt*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscBLASInt*,PetscBLASInt*,PetscScalar*,PetscBLASInt*);
PETSC_EXTERN void PETSC_STDCALL PETSCBLAS(hsein,HSEIN) (const char*,PetscBLASInt,const char*,PetscBLASInt,const char*,PetscBLASInt,PetscBLASInt*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscScalar*,PetscScalar*,PetscScalar*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscBLASInt*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscBLASInt*,PetscBLASInt*);
PETSC_EXTERN void PETSC_STDCALL PETSCBLAS(stedc,STEDC) (const char*,PetscBLASInt,PetscBLASInt*,PetscReal*,PetscReal*,PetscScalar*,PetscBLASInt*,PetscReal*,PetscBLASInt*,PetscBLASInt*,PetscBLASInt*,PetscBLASInt*);
PETSC_EXTERN void PETSC_STDCALL PETSCBLAS(lascl,LASCL) (const char*,PetscBLASInt,PetscBLASInt*,PetscBLASInt*,PetscScalar*,PetscScalar*,PetscBLASInt*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscBLASInt*);
#else
PETSC_EXTERN void PETSC_STDCALL PETSCBLAS(ggevx,GGEVX) (const char*,PetscBLASInt,const char*,PetscBLASInt,const char*,PetscBLASInt,const char*,PetscBLASInt,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscScalar*,PetscScalar*,PetscScalar*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscBLASInt*,PetscBLASInt*,PetscReal*,PetscReal*,PetscReal*,PetscReal*,PetscReal*,PetscReal*,PetscScalar*,PetscBLASInt*,PetscReal*,PetscBLASInt*,PetscBLASInt*,PetscBLASInt*);
PETSC_EXTERN void PETSC_STDCALL PETSCBLAS(ggev,GGEV) (const char*,PetscBLASInt,const char*,PetscBLASInt,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscScalar*,PetscScalar*,PetscScalar*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscReal*,PetscBLASInt*);
PETSC_EXTERN void PETSC_STDCALL PETSCBLAS(trevc,TREVC) (const char*,PetscBLASInt,const char*,PetscBLASInt,PetscBLASInt*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscBLASInt*,PetscBLASInt*,PetscScalar*,PetscReal*,PetscBLASInt*);
PETSC_EXTERN void PETSC_STDCALL PETSCBLAS(geevx,GEEVX) (const char*,PetscBLASInt,const char*,PetscBLASInt,const char*,PetscBLASInt,const char*,PetscBLASInt,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscScalar*,PetscScalar*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscBLASInt*,PetscBLASInt*,PetscReal*,PetscReal*,PetscReal*,PetscReal*,PetscScalar*,PetscBLASInt*,PetscReal*,PetscBLASInt*);
PETSC_EXTERN void PETSC_STDCALL PETSCBLAS(gees,GEES) (const char*,PetscBLASInt,const char*,PetscBLASInt,PetscBLASInt(*)(PetscScalar),PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscBLASInt*,PetscScalar*,PetscScalar*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscReal*,PetscBLASInt*,PetscBLASInt*);
PETSC_EXTERN void PETSC_STDCALL PETSCBLAS(trexc,TREXC) (const char*,PetscBLASInt,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscBLASInt*,PetscBLASInt*,PetscBLASInt*);
PETSC_EXTERN void PETSC_STDCALL PETSCBLAS(gesdd,GESDD) (const char*,PetscBLASInt,PetscBLASInt*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscReal*,PetscScalar*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscReal*,PetscBLASInt*,PetscBLASInt*);
PETSC_EXTERN void PETSC_STDCALL PETSCBLAS(tgevc,TGEVC) (const char*,PetscBLASInt,const char*,PetscBLASInt,PetscBLASInt*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscBLASInt*,PetscBLASInt*,PetscScalar*,PetscReal*,PetscBLASInt*);
PETSC_EXTERN void PETSC_STDCALL PETSCBLAS(hsein,HSEIN) (const char*,PetscBLASInt,const char*,PetscBLASInt,const char*,PetscBLASInt,PetscBLASInt*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscScalar*,PetscScalar*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscBLASInt*,PetscBLASInt*,PetscScalar*,PetscReal*,PetscBLASInt*,PetscBLASInt*,PetscBLASInt*);
PETSC_EXTERN void PETSC_STDCALL PETSCBLAS(stedc,STEDC) (const char*,PetscBLASInt,PetscBLASInt*,PetscReal*,PetscReal*,PetscScalar*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscReal*,PetscBLASInt*,PetscBLASInt*,PetscBLASInt*,PetscBLASInt*);
PETSC_EXTERN void PETSC_STDCALL PETSCBLAS(lascl,LASCL) (const char*,PetscBLASInt,PetscBLASInt*,PetscBLASInt*,PetscReal*,PetscReal*,PetscBLASInt*,PetscBLASInt*,PetscScalar*,PetscBLASInt*,PetscBLASInt*);
#endif

#if defined(PETSC_HAVE_COMPLEX)
/* complex subroutines to be called with scalar-type=real */
#define BLASCOMPLEXgemm_(a,b,c,d,e,f,g,h,i,j,k,l,m) PETSCBLASCOMPLEX(gemm,GEMM)((a),1,(b),1,(c),(d),(e),(f),(g),(h),(i),(j),(k),(l),(m))
PETSC_EXTERN void PETSC_STDCALL PETSCBLASCOMPLEX(gemm,GEMM)(const char*,const char*,PetscBLASInt*,PetscBLASInt*,PetscBLASInt*,PetscComplex*,PetscComplex*,PetscBLASInt*,PetscComplex*,PetscBLASInt*,PetscComplex*,PetscComplex*,PetscBLASInt*);
#define BLASCOMPLEXscal_   PETSCBLASCOMPLEX(scal,SCAL)
PETSC_EXTERN void PETSC_STDCALL PETSCBLASCOMPLEX(scal,SCAL)(const PetscBLASInt*,const PetscComplex*,PetscComplex*,const PetscBLASInt*);
#define LAPACKCOMPLEXgesv_ PETSCBLASCOMPLEX(gesv,GESV)
PETSC_EXTERN void PETSC_STDCALL PETSCBLASCOMPLEX(gesv,GESV)(const PetscBLASInt*,const PetscBLASInt*,PetscComplex*,const PetscBLASInt*,PetscBLASInt*,PetscComplex*,const PetscBLASInt*,PetscBLASInt*);
#endif

#endif

