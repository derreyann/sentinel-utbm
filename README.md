# Installation process


## GDAL Installation

Install GDAL with HDF4 support. This installation process will depend on your system.
GDAL and the python bidings are needed.

### Arch

Using the AUR depository and yay
``` shell
yay -S gdal-hdf4 python-gdal-hdf4
```

## Dependencies installation 

#### Using poetry 

Most dependencies can be installed using poetry.
``` shell
poetry install
```

### Rastario

After installing GDAL and HDF4 support, you can check if you can open files using
``` shell
gdalinfo 'file.hdf'
```

Once you can successfully read HDF files, you need to install rastario from source for it to include HDF4 support : 
``` shell
pip install rasterio==1.3.10 --no-binary :all:
```

## Credentials

You need to setup your own credentials in the config.yaml file.
You need to add the Modis / NASA username and password, as well as your sentinel hub id and secret.