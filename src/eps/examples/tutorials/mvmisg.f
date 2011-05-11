      SUBROUTINE MVMISG( TRANS, N, M, X, LDX, Y, LDY )
*     ..
*     .. Scalar Arguments ..
      INTEGER            LDY, LDX, M, N, TRANS
*     ..
*     .. Array Arguments ..
      DOUBLE PRECISION   Y( LDY, * ), X( LDX, * )
*     ..
*
*  Purpose
*  =======
*
*  Compute
*
*               Y(:,1:M) = op(A)*X(:,1:M)
*
*  where op(A) is A or A' (the transpose of A). The A is the Ising 
*  matrix.
*
*  Arguments
*  =========
*
*  TRANS   (input) INTEGER
*          If TRANS = 0, compute Y(:,1:M) = A*X(:,1:M) 
*          If TRANS = 1, compute Y(:,1:M) = A'*X(:,1:M) 
*           
*  N       (input) INTEGER
*          The order of the matrix A. N has to be an even number.
*
*  M       (input) INTEGERS
*          The number of columns of X to multiply.
*
*  X       (input) DOUBLE PRECISION array, dimension ( LDX, M )
*          X contains the matrix (vectors) X.
*
*  LDX     (input) INTEGER
*          The leading dimension of array X, LDX >= max( 1, N )
*
*  Y       (output) DOUBLE PRECISION array, dimension (LDX, M )
*          contains the product of the matrix op(A) with X.
*
*  LDY     (input) INTEGER
*          The leading dimension of array Y, LDY >= max( 1, N )
*
*  ===================================================================
*
*
*     .. PARAMETERS ..
      DOUBLE PRECISION   PI 
      PARAMETER          ( PI = 3.141592653589793D+00 )
      DOUBLE PRECISION   ALPHA, BETA
      PARAMETER          ( ALPHA = PI/4, BETA = PI/4 ) 
*
*     .. Local Variables .. 
      INTEGER            I, K 
      DOUBLE PRECISION   COSA, COSB, SINA, SINB, TEMP, TEMP1 
*
*     .. Intrinsic functions ..
      INTRINSIC          COS, SIN 
*
      COSA = COS( ALPHA ) 
      SINA = SIN( ALPHA ) 
      COSB = COS( BETA )
      SINB = SIN( BETA ) 
*      
      IF ( TRANS.EQ.0 ) THEN 
*
*     Compute Y(:,1:M) = A*X(:,1:M)

         DO 30 K = 1, M
*
            Y( 1, K ) = COSB*X( 1, K ) - SINB*X( N, K ) 
            DO 10 I = 2, N-1, 2   
               Y( I, K )   =  COSB*X( I, K ) + SINB*X( I+1, K )
               Y( I+1, K ) = -SINB*X( I, K ) + COSB*X( I+1, K )  
   10       CONTINUE
            Y( N, K ) = SINB*X( 1, K ) + COSB*X( N, K ) 
*
            DO 20 I = 1, N, 2
               TEMP        =  COSA*Y( I, K ) + SINA*Y( I+1, K )
               Y( I+1, K ) = -SINA*Y( I, K ) + COSA*Y( I+1, K )  
               Y( I, K )   = TEMP 
   20       CONTINUE  
*
   30    CONTINUE 
*
      ELSE IF ( TRANS.EQ.1 ) THEN 
*
*        Compute Y(:1:M) = A'*X(:,1:M) 
*
         DO 60 K = 1, M 
*
            DO 40 I = 1, N, 2
               Y( I, K )   =  COSA*X( I, K ) - SINA*X( I+1, K )
               Y( I+1, K ) =  SINA*X( I, K ) + COSA*X( I+1, K )  
   40       CONTINUE  
            TEMP  = COSB*Y(1,K) + SINB*Y(N,K) 
            DO 50 I = 2, N-1, 2   
               TEMP1       =  COSB*Y( I, K ) - SINB*Y( I+1, K )
               Y( I+1, K ) =  SINB*Y( I, K ) + COSB*Y( I+1, K )  
               Y( I, K )   =  TEMP1
   50       CONTINUE
            Y( N, K ) = -SINB*Y( 1, K ) + COSB*Y( N, K ) 
            Y( 1, K ) = TEMP 
*
   60    CONTINUE
*
      END IF 
*
      RETURN
*  
*     END OF MVMISG
      END 
