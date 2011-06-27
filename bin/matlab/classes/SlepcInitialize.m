function err = SlepcInitialize(args,argfile,arghelp)
%
%  In order to use the SLEPc MATLAB classes, PETSc must have been configured with
%  specific options. See: help PetscInitialize.
%
%  Add ${PETSC_DIR}/bin/matlab/classes to your MATLAB path, as well as
%  ${SLEPC_DIR}/bin/matlab/classes
%
%  In MATLAB use help Slepc to get started using SLEPc from MATLAB
%

if exist('PetscInitialize')~=2
  error('Must add ${PETSC_DIR}/bin/matlab/classes to your MATLAB path')
end

if ~libisloaded('libslepc')
  SLEPC_DIR = getenv('SLEPC_DIR');
  PETSC_ARCH = getenv('PETSC_ARCH');
  if isempty(SLEPC_DIR)
    disp('Must have environmental variable SLEPC_DIR set')
  end
  if isempty(PETSC_ARCH)
    disp('Must have environmental variable PETSC_ARCH set')
  end
  loadlibrary([SLEPC_DIR '/' PETSC_ARCH '/lib/' 'libslepc'], [SLEPC_DIR '/bin/matlab/classes/slepcmatlabheader.h']);
end

if ~libisloaded('libpetsc')
  PETSC_DIR = getenv('PETSC_DIR');
  PETSC_ARCH = getenv('PETSC_ARCH');
  if isempty(PETSC_DIR)
    disp('Must have environmental variable PETSC_DIR set')
  end
  if isempty(PETSC_ARCH)
    disp('Must have environmental variable PETSC_ARCH set')
  end
  loadlibrary([PETSC_DIR '/' PETSC_ARCH '/lib/' 'libpetsc'], [PETSC_DIR '/bin/matlab/classes/matlabheader.h']);
end

if (nargin == 0)
  args = '';
end
if (nargin < 2) 
  argfile = '';
end
if (nargin < 3)
  arghelp = '';
end
if (ischar(args)) 
  args = {args};
end

% append any options in the options variable
global options
if (length(options) > 0)
  args = [args,options];
  disp('Using additional options')
  disp(options)
end

% first argument should be program name, use matlab for this
arg = ['matlab',args];
%
% If the user forgot to SlepcFinalize() we do it for them, before restarting SLEPc
%
init = calllib('libslepc', 'SlepcInitializedMatlab');
if (init) 
  err = calllib('libslepc', 'SlepcFinalize');PetscCHKERRQ(err);
end
err = calllib('libslepc', 'SlepcInitializeMatlab', length(arg), arg,argfile,arghelp);PetscCHKERRQ(err);


