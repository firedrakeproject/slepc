/* dcond.f -- translated by f2c (version 20030320).
   You must link the resulting object file with the libraries:
	-lf2c -lm   (in that order)
*/

/* #include "f2c.h" */

#include "slepceps.h"

typedef PetscTruth logical;
typedef PetscBLASInt integer;
typedef PetscScalar doublereal;
typedef PetscBLASInt ftnlen;

#define abs(x) ((x) >= 0 ? (x) : -(x))
#define min(a,b) ((a) <= (b) ? (a) : (b))
#define max(a,b) ((a) >= (b) ? (a) : (b))

doublereal dcond_(integer *m, doublereal *h__, integer *ldh, integer *ipvt, 
	doublereal *mult, integer *info)
{
    /* System generated locals */
    integer h_dim1, h_offset, i__1, i__2, i__3;
    doublereal ret_val, d__1, d__2;

    /* Local variables */
    static integer i__, j, k;
    static doublereal s;
    static integer jj;
    static doublereal hn;
    static integer jm1;
    static doublereal hin;
    extern doublereal dlange_(char *, integer *, integer *, doublereal *, 
	    integer *, doublereal *, ftnlen), dlanhs_(char *, integer *, 
	    doublereal *, integer *, doublereal *, ftnlen);

/*     .. */
/*     .. Scalar Arguments .. */
/*     .. */
/*     .. Array Arguments .. */
/*     .. */

/*  Purpose */
/*  ======= */

/*  Compute the inf-norm condition number of an upper Hessenberg */
/*  matrix H: */

/*             COND = norm(H)*norm(inv(H)) */

/*  In this application, the code uses Gaussian elimination with partial */
/*  pivoting to compute the inverse explicitly. */

/*  Arguments */
/*  ========= */

/*  M       (input) INTEGER */
/*          M is the order of matrix H. */

/*  H       (input/output) DOUBLE PRECISION array, dimension( LDH,M ) */
/*          On entry, H contains an M by M upper Hessenberg matrix. */
/*          On exit, H contains the inverse of H. */

/*  LDH     (input) INTEGER */
/*          The leading dimension of the array H, LDH >= max(1,M) */

/*  IPVT    (workspace) INTEGER array, dimension(M) */

/*  MULT    (workspace) DOUBLE PRECISION array, dimension(2*M) */

/*  INFO    (output) INTEGER */
/*          On exit, if INFO is set to */
/*             0,  normal return. */
/*             1,  the matrix is singular, condition number is infinity. */

/*  ==================================================================== */

/*     .. Parameters .. */
/*     .. */
/*     .. Local Scalars .. */
/*     .. */
/*     .. External Functions .. */
/*     .. */
/*     .. Intrinsic Functions .. */
/*     .. */
/*     .. Executable Statements .. */

    /* Parameter adjustments */
    h_dim1 = *ldh;
    h_offset = 1 + h_dim1;
    h__ -= h_offset;
    --ipvt;
    --mult;

    /* Function Body */
    *info = 0;

/*     Compute inf-norm of the upper Hessenberg matrix H */

    hn = dlanhs_("Inf", m, &h__[h_offset], ldh, &mult[1], (ftnlen)3);

/*     Compute the inverse of matrix H */

/*     Step 1: Gaussian elimination with partial pivoting: */
/*           (M_{m-1} P_{m-1} .... M_1 P_1 )H = H1 */
/*     where P_i and M_i are permutation and elementray transformation */
/*     matrices, resp. (stored in MULT and IPVT), H1 is an upper */
/*     triangular */

    i__1 = *m - 1;
    for (i__ = 1; i__ <= i__1; ++i__) {
	ipvt[i__] = 0;
	mult[i__] = 0.;
	if (h__[i__ + 1 + i__ * h_dim1] == 0.) {
	    goto L60;
	}

	if ((d__1 = h__[i__ + 1 + i__ * h_dim1], abs(d__1)) <= (d__2 = h__[
		i__ + i__ * h_dim1], abs(d__2))) {
	    goto L40;
	}

	ipvt[i__] = 1;
	i__2 = *m;
	for (j = i__; j <= i__2; ++j) {
	    s = h__[i__ + j * h_dim1];
	    h__[i__ + j * h_dim1] = h__[i__ + 1 + j * h_dim1];
	    h__[i__ + 1 + j * h_dim1] = s;
/* L30: */
	}

L40:

	mult[i__] = h__[i__ + 1 + i__ * h_dim1] / h__[i__ + i__ * h_dim1];
	h__[i__ + 1 + i__ * h_dim1] = 0.;
	i__2 = *m;
	for (j = i__ + 1; j <= i__2; ++j) {
	    h__[i__ + 1 + j * h_dim1] -= mult[i__] * h__[i__ + j * h_dim1];
/* L50: */
	}
L60:
	;
    }

/*     Step 2: compute the inverse of H1 (triangular matrix) */

    i__1 = *m;
    for (j = 1; j <= i__1; ++j) {
	if (h__[j + j * h_dim1] == 0.) {
	    ret_val = 1e8;
	    *info = 1;
	    return ret_val;
	}

	h__[j + j * h_dim1] = 1. / h__[j + j * h_dim1];
	if (j == 1) {
	    goto L100;
	}
	jm1 = j - 1;
	i__2 = jm1;
	for (i__ = 1; i__ <= i__2; ++i__) {
	    s = 0.;
	    i__3 = jm1;
	    for (k = i__; k <= i__3; ++k) {
		s += h__[i__ + k * h_dim1] * h__[k + j * h_dim1];
/* L80: */
	    }
	    h__[i__ + j * h_dim1] = -s * h__[j + j * h_dim1];
/* L90: */
	}

L100:

/* L110: */
	;
    }

/*     Step 3: Compute the inverse of H: */
/*         inv(H) = inv(H1)*(M_{m-1} P_{m-1} ... M_1 P_1) */

    i__1 = *m - 1;
    for (jj = 1; jj <= i__1; ++jj) {
	j = *m - jj;
	if (mult[j] == 0.) {
	    goto L130;
	}
	i__2 = *m;
	for (i__ = 1; i__ <= i__2; ++i__) {
	    h__[i__ + j * h_dim1] -= mult[j] * h__[i__ + (j + 1) * h_dim1];
/* L120: */
	}

L130:
	if (ipvt[j] == 0) {
	    goto L150;
	}
	i__2 = *m;
	for (i__ = 1; i__ <= i__2; ++i__) {
	    s = h__[i__ + j * h_dim1];
	    h__[i__ + j * h_dim1] = h__[i__ + (j + 1) * h_dim1];
	    h__[i__ + (j + 1) * h_dim1] = s;
/* L140: */
	}

L150:
/* L160: */
	;
    }

/*     Compute the inf-norm of the inverse matrix. */

    hin = dlange_("Inf", m, m, &h__[h_offset], ldh, &mult[1], (ftnlen)3);

/*     Compute the condition number */

    ret_val = hn * hin;

    return ret_val;

/*     End of DCOND */
} /* dcond_ */

