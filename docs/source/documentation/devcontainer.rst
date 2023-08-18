Using development containers with Visual Studio Code
====================================================

Scilpy comes pre-equiped with a development container recipe for Visual Studio Code. You can use it to develop Scilpy on any platform (Windows, Linux, Mac) without having to install any dependencies on your host machine. 


Prerequisites
-------------

The only requirement is to have `Docker <https://docs.docker.com/get-docker>`__ and `Visual Studio Code <https://code.visualstudio.com/download>`__ installed, with the `Dev Containers extension <https://marketplace.visualstudio.com/items?itemName=ms-vscode-remote.remote-containers>`__ activated.


Launching the container
-----------------------

1. Clone the Scilpy repository locally with Visual Studio Code and open it.

2. Open the command palette (Ctrl+Shift+P) and select `Remote-Containers: Reopen in Container`.

3. Wait for the container to build and start. You can monitor the progress in the bottom left corner of the window.


Using the container
-------------------

Once the container is running, you can use Visual Studio Code as you would normally. The only difference is that the code is running inside the container, so you can use all the tools and dependencies that are installed in it, such as *ANTs*, *FSL* and *mrtrix*. The container is also equiped with *offscreen rendering* capabilities, so scripts generating figures should work.

.. note::
    The only accessible path from outside the container is the Scilpy repository. If you want to access other files, you will have to mount them in the container. See `this page <https://code.visualstudio.com/remote/advancedcontainers/add-local-file-mount>`__ for more information.

.. note::
    The container is running as a non-root user, so you might have to change the permissions of some files and folders to be able to write to them.
