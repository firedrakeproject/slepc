%%
%
%  Solves a generalized eigenvalue problem with SLEPc with shift-and-invert
%  with the matrices loaded from file
%

%%
%  Set the Matlab path and initialize SLEPc
%
path(path,'../../')
if ~exist('PetscInitialize','file')
  PETSC_DIR = getenv('PETSC_DIR');
  if isempty(PETSC_DIR) 
    error('Must set environment variable PETSC_DIR or add the appropriate dir to Matlab path')
  end
  path(path,[PETSC_DIR '/bin/matlab/classes'])
end
SlepcInitialize({'-malloc','-malloc_debug','-malloc_dump'});

%%
%  Load matrices from file
%
SLEPC_DIR = getenv('SLEPC_DIR');
viewer = PetscViewer([SLEPC_DIR '/src/mat/examples/bfw62a.petsc'],Petsc.FILE_MODE_READ);
A = PetscMat();
A.Load(viewer);
viewer.Destroy;
viewer = PetscViewer([SLEPC_DIR '/src/mat/examples/bfw62b.petsc'],Petsc.FILE_MODE_READ);
B = PetscMat();
B.Load(viewer);
viewer.Destroy;

%%
%  Create the eigensolver, pass the matrices and solve the problem
%
eps = SlepcEPS();
eps.SetType('krylovschur');
eps.SetOperators(A,B);
eps.SetProblemType(SlepcEPS.GNHEP);
eps.SetDimensions(6);
eps.SetTolerances(1e-13);
st = eps.GetST();
st.SetType('sinvert');
eps.SetWhichEigenpairs(SlepcEPS.TARGET_MAGNITUDE);
eps.SetTarget(0.0);
eps.SetFromOptions();
eps.Solve();
nconv = eps.GetConverged();
fprintf('           k          ||Ax-kx||/||kx||\n')
fprintf('   ----------------- ------------------\n')
for i=1:nconv
  lambda = eps.GetEigenpair(i);
  relerr = eps.ComputeRelativeError(i);
  if isreal(lambda)
    fprintf('    %14.2f        %12g\n',lambda,relerr)
  else
    fprintf('  %.2f%+.2f      %12g\n',real(lambda),imag(lambda),relerr)
  end
end

%%
%   Free objects and shutdown SLEPc
%
A.Destroy();
B.Destroy();
eps.Destroy();
SlepcFinalize();
