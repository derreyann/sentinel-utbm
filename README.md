# DS50 Project: Sentinel, MODIS and Weather Processing Pipeline

This projects provides an easy way to work with satellite data for Data Science. The data is gathered from their respective APIs, converted then standardized, and outputed into .tiff files as well as Tensors for use in models.

[The full report is available in English here](DS50_REPORT_GROUPE_5.pdf)

Data sources:
- Sentinel from SentinelHub data for analysis
- MODIS from NASA data for target labeling
- Weather data from MeteoStat as feature data


## GDAL Installation

Install GDAL with HDF4 support. This installation process will depend on your system.
GDAL and the python bindings are needed.

### macOS

Using MacPorts
``` zsh
sudo port install gdal +hdf4
```

### Arch

Using the AUR repository and yay
``` shell
yay -S gdal-hdf4 python-gdal-hdf4
```

## Dependencies installation 

#### Using poetry 

Most dependencies can be installed using poetry.
``` shell
poetry install
```

### Rasterio

After installing GDAL and HDF4 support, you can check if you can open files using
``` shell
gdalinfo 'file.hdf'
```

Once you can successfully read HDF files, you need to install rasterio from source for it to include HDF4 support : 
``` shell
pip install rasterio==1.3.10 --no-binary :all:
```

## Credentials

You need to setup your own credentials in the config.yaml file.
You need to add the Modis / NASA username and password, as well as your sentinel hub id and secret.
