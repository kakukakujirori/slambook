# slambook

This is a re-implementation of the code associated with [this book](https://github.com/gaoxiang12/slambook2).

These codes works in macOS environment.

## Note

- In ch12, if cmake cannot find qt5, please type instead:
    ```
    cmake .. -DCMAKE_PREFIX_PATH=$(brew --prefix qt5)
    ```
    Also, pcd_viewer seems not to be supported on MacOS, so use MeshLab instead.
    In order to do so, delete the first 11 lines (until 'DATA ascii'), and change its extension to '.asc'.

- In ch13, it needs openmp, which is not supported in clang. Therefore you need to pass an flag:
    ```
    cmake -DCMAKE_CXX_COMPILER=/usr/local/opt/llvm/bin/clang ..
    ```