# Scilpy

**Scilpy** is the main library supporting research and development at the Sherbrooke Connectivity Imaging Lab
([SCIL]).

**Scilpy** mainly comprises tools and utilities to quickly work with diffusion MRI. Most of the tools are based
on or are wrappers of the [DIPY] library, and most of them will eventually be migrated to [DIPY]. Those tools implement the recommended workflows and parameters used in the lab.

The library's structure is mostly aligned on that of [DIPY].

### Dependencies and installation

We highly recommend working in a [Python Virtual Environment]. Dependencies can be installed by running
```
pip install -r requirements.txt
```

Following this, the library and scripts can be installed locally by using

```
python setup.py build_ext --inplace
python setup.py install
python setup.py install_scripts
```
Note that using this technique will make it harder to remove the scripts when changing versions.

[SCIL]:http://scil.dinf.usherbrooke.ca/
[DIPY]:http://dipy.org
[Python Virtual Environment]:https://virtualenv.pypa.io/en/latest/