/*
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
   SLEPc - Scalable Library for Eigenvalue Problem Computations
   Copyright (c) 2002-2018, Universitat Politecnica de Valencia, Spain

   This file is part of SLEPc.
   SLEPc is distributed under a 2-clause BSD license (see LICENSE).
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
*/

#if !defined(__SLEPCBLASLAPACK_MANGLE_H)
#define __SLEPCBLASLAPACK_MANGLE_H

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

/* LAPACK functions with string parameters */

/* same name for real and complex */
#define BLAStrmm_    PETSCBLAS(trmm,TRMM)
#define LAPACKlanhs_ PETSCBLAS(lanhs,LANHS)
#define LAPACKlange_ PETSCBLAS(lange,LANGE)
#define LAPACKpbtrf_ PETSCBLAS(pbtrf,PBTRF)
#define LAPACKlarf_  PETSCBLAS(larf,LARF)
#define LAPACKlacpy_ PETSCBLAS(lacpy,LACPY)
#define LAPACKlansy_ PETSCBLAS(lansy,LANSY)
#define LAPACKlaset_ PETSCBLAS(laset,LASET)
#define LAPACKtrsyl_ PETSCBLAS(trsyl,TRSYL)
#define LAPACKtrtri_ PETSCBLAS(trtri,TRTRI)

/* subroutines in which we use only the real version, do not care whether they have different name */
#define LAPACKstevr_ PETSCBLASREAL(stevr,STEVR)
#define LAPACKbdsdc_ PETSCBLASREAL(bdsdc,BDSDC)
#define LAPACKlamch_ PETSCBLASREAL(lamch,LAMCH)
#define LAPACKlamc3_ PETSCBLASREAL(lamc3,LAMC3)

/* subroutines with different name in real/complex */
#if !defined(PETSC_USE_COMPLEX)
#define LAPACKormlq_ PETSCBLAS(ormlq,ORMLQ)
#define LAPACKorgtr_ PETSCBLAS(orgtr,ORGTR)
#define LAPACKsytrd_ PETSCBLAS(sytrd,SYTRD)
#define LAPACKsyevr_ PETSCBLAS(syevr,SYEVR)
#define LAPACKsyevd_ PETSCBLAS(syevd,SYEVD)
#define LAPACKsygvd_ PETSCBLAS(sygvd,SYGVD)
#else
#define LAPACKormlq_ PETSCBLAS(unmlq,UNMLQ)
#define LAPACKorgtr_ PETSCBLAS(ungtr,UNGTR)
#define LAPACKsytrd_ PETSCBLAS(hetrd,HETRD)
#define LAPACKsyevr_ PETSCBLAS(heevr,HEEVR)
#define LAPACKsyevd_ PETSCBLAS(heevd,HEEVD)
#define LAPACKsygvd_ PETSCBLAS(hegvd,HEGVD)
#endif

/* subroutines with different signature in real/complex */
#define LAPACKggevx_ PETSCBLAS(ggevx,GGEVX)
#define LAPACKggev_  PETSCBLAS(ggev,GGEV)
#define LAPACKtrevc_ PETSCBLAS(trevc,TREVC)
#define LAPACKgeevx_ PETSCBLAS(geevx,GEEVX)
#define LAPACKgees_  PETSCBLAS(gees,GEES)
#define LAPACKtrexc_ PETSCBLAS(trexc,TREXC)
#define LAPACKgesdd_ PETSCBLAS(gesdd,GESDD)
#define LAPACKtgevc_ PETSCBLAS(tgevc,TGEVC)
#define LAPACKhsein_ PETSCBLAS(hsein,HSEIN)
#define LAPACKstedc_ PETSCBLAS(stedc,STEDC)
#define LAPACKlascl_ PETSCBLAS(lascl,LASCL)

#if defined(PETSC_HAVE_COMPLEX)
/* complex subroutines to be called with scalar-type=real */
#define BLASCOMPLEXgemm_   PETSCBLASCOMPLEX(gemm,GEMM)
#define BLASCOMPLEXscal_   PETSCBLASCOMPLEX(scal,SCAL)
#define LAPACKCOMPLEXgesv_ PETSCBLASCOMPLEX(gesv,GESV)
#endif

#endif
