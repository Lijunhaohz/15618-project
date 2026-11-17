"""Match the CUDA compiler setup used by run_cuda_sample.sh."""
import os
from pathlib import Path
import shutil
import subprocess


def configure_cuda_environment():
    compiler = os.environ.get('CUDACXX') or shutil.which('nvcc')
    if not compiler and os.access('/usr/local/cuda/bin/nvcc', os.X_OK):
        compiler = '/usr/local/cuda/bin/nvcc'
    if not compiler:
        raise RuntimeError('CUDA compiler not found. Set CUDACXX=/path/to/nvcc or add CUDA bin to PATH. '
                           'nvidia-smi alone does not provide the CUDA compiler.')
    resolved = shutil.which(compiler)
    if not resolved:
        raise RuntimeError(f'CUDA compiler is not executable: {compiler}')
    os.environ['CUDACXX'] = resolved
    # Also expose sibling toolkit tools such as ncu when supplied with CUDA.
    os.environ['PATH'] = str(Path(resolved).parent) + os.pathsep + os.environ.get('PATH', '')
    version = subprocess.run([resolved, '--version'], capture_output=True, text=True, check=True).stdout
    if not os.environ.get('CUDAHOSTCXX') and not os.environ.get('CXX'):
        host = shutil.which('g++-11')
        if host and any(release in version for release in ('release 11.7,', 'release 11.8,')):
            os.environ['CUDAHOSTCXX'] = host
    print('CUDA compiler: ' + resolved, flush=True)
    if os.environ.get('CUDAHOSTCXX'):
        print('CUDA host compiler: ' + os.environ['CUDAHOSTCXX'], flush=True)
    return resolved
