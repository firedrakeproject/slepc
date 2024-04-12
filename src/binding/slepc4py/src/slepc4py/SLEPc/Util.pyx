# -----------------------------------------------------------------------------

cdef class Util:

    @classmethod
    def createMatBSE(cls, Mat R, Mat C):
        """
        Create a matrix that can be used to define a structured eigenvalue
        problem of type BSE (Bethe-Salpeter Equation).

        Parameters
        ----------
        R: Mat
           The matrix for the diagonal block (resonant).
        C: Mat
           The matrix for the off-diagonal block (coupling).

        Returns
        -------
        H: Mat
           The matrix with the block form H = [ R C; -C^H -R^T ].
        """
        cdef Mat H = Mat()
        CHKERR( MatCreateBSE(R.mat, C.mat, &H.mat) )
        return H

# -----------------------------------------------------------------------------
