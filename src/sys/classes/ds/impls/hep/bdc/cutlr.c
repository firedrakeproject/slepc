/*
   BDC - Block-divide and conquer (see description in README file).

   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
   SLEPc - Scalable Library for Eigenvalue Problem Computations
   Copyright (c) 2002-2013, Universitat Politecnica de Valencia, Spain

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

#include <slepc-private/dsimpl.h>
#include <slepcblaslapack.h>

PetscErrorCode cutlr_(PetscBLASInt start,PetscBLASInt n,PetscBLASInt blkct, 
        PetscBLASInt *bsizes,PetscBLASInt *ranks,PetscBLASInt *cut,
        PetscBLASInt *lsum,PetscBLASInt *lblks,PetscBLASInt *info)
{
/*  -- Routine written in LAPACK Version 3.0 style -- */
/* *************************************************** */
/*     Written by */
/*     Michael Moldaschl and Wilfried Gansterer */
/*     University of Vienna */
/*     last modification: March 16, 2014 */

/*     Small adaptations of original code written by */
/*     Wilfried Gansterer and Bob Ward, */
/*     Department of Computer Science, University of Tennessee */
/*     see http://dx.doi.org/10.1137/S1064827501399432 */
/* *************************************************** */

/*  Purpose */
/*  ======= */

/*  CUTLR computes the optimal cut in a sequence of BLKCT neighboring */
/*  blocks whose sizes are given by the array BSIZES. */
/*  The sum of all block sizes in the sequence considered is given by N. */
/*  The cut is optimal in the sense that the difference of the sizes of */
/*  the resulting two halves is minimum over all cuts with minimum ranks */
/*  between blocks of the sequence considered. */

/*  Arguments */
/*  ========= */

/*  START  (input) INTEGER */
/*         In the original array KSIZES of the calling routine DIBTDC, */
/*         the position where the sequence considered in this routine starts. */
/*         START >= 1. */

/*  N      (input) INTEGER */
/*         The sum of all the block sizes of the sequence to be cut = */
/*         = sum_{i=1}^{BLKCT} BSIZES( I ). */
/*         N >= 3. */

/*  BLKCT  (input) INTEGER */
/*         The number of blocks in the sequence to be cut. */
/*         BLKCT >= 3. */

/*  BSIZES (input) INTEGER array, dimension (BLKCT) */
/*         The dimensions of the (quadratic) blocks of the sequence to be */
/*         cut. sum_{i=1}^{BLKCT} BSIZES( I ) = N. */

/*  RANKS  (input) INTEGER array, dimension (BLKCT-1) */
/*         The ranks determining the approximations of the off-diagonal */
/*         blocks in the sequence considered. */

/*  CUT    (output) INTEGER */
/*         After the optimum cut has been determined, the position (in the */
/*         overall problem as worked on in DIBTDC !) of the last block in */
/*         the first half of the sequence to be cut. */
/*         START <= CUT <= START+BLKCT-2. */

/*  LSUM   (output) INTEGER */
/*         After the optimum cut has been determined, the sum of the */
/*         block sizes in the first half of the sequence to be cut. */
/*         LSUM < N. */

/*  LBLKS  (output) INTEGER */
/*         After the optimum cut has been determined, the number of the */
/*         blocks in the first half of the sequence to be cut. */
/*         1 <= LBLKS < BLKCT. */

/*  INFO   (output) INTEGER */
/*          = 0:  successful exit. */
/*          < 0:  illegal arguments. */
/*                if INFO = -i, the i-th (input) argument had an illegal */
/*                value. */
/*          > 0:  illegal results. */
/*                if INFO = i, the i-th (output) argument had an illegal */
/*                value. */

/*  Further Details */
/*  =============== */

/*  Based on code written by */
/*     Wilfried Gansterer and Bob Ward, */
/*     Department of Computer Science, University of Tennessee */

/*  ===================================================================== */

  PetscBLASInt i, ksk, kchk, ksum, nhalf, deviat, mindev, minrnk, tmpsum;

  PetscFunctionBegin;
  *info = 0;

  if (start < 1) {
    *info = -1;
  } else if (n < 3) {
    *info = -2;
  } else if (blkct < 3) {
    *info = -3;
  }
  if (*info == 0) {
    ksum = 0;
    kchk = 0;
    for (i = 0; i < blkct; ++i) {
      ksk = bsizes[i];
      ksum += ksk;
      if (ksk < 1) kchk = 1;
    }
    if (ksum != n || kchk == 1) *info = -4;
  }
  if (*info) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONG,"Wrong argument %d in CUTLR",-(*info));

  /* determine smallest rank in the range considered */

  minrnk = n;
  for (i = 0; i < blkct-1; ++i) {
    if (ranks[i] < minrnk) minrnk = ranks[i];
  }

  /* determine best cut among those with smallest rank */

  nhalf = n / 2;
  tmpsum = 0;
  mindev = n;
  for (i = 0; i < blkct; ++i) {
    tmpsum += bsizes[i];
    if (ranks[i] == minrnk) {

      /* determine deviation from "optimal" cut NHALF */

      deviat = tmpsum - nhalf;
      if (deviat<0) deviat = -deviat;

      /* compare to best deviation so far */

      if (deviat < mindev) {
        mindev = deviat;
        *cut = start + i;
        *lblks = i + 1;
        *lsum = tmpsum;
      }
    }
  }

  if (*cut < start || *cut >= start + blkct - 1) {
    *info = 6;
  } else if (*lsum < 1 || *lsum >= n) {
    *info = 7;
  } else if (*lblks < 1 || *lblks >= blkct) {
    *info = 8;
  }
  PetscFunctionReturn(0);
}

