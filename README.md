# Earth Observation Product Formats

`Arepyextras Earth Observation Products` is the Aresys Python module to manage external product formats.

List of supported formats:

- **Sentinel-1 SAFE**: SLC and GRD products
- **NovaSAR-1**: SLC, GRD, SCD and SRD products
- **ICEYE**: SLC and GRD products [topsar, stripmap, spotlight]
- **SAOCOM**: SLC and GRD products [topsar, stripmap]
- **EOS-04**: SLC and GRD products [scansar, stripmap]

Heritage missions:

- **ENVISAT/ERS**: SLC and GRD ASAR products

> [!WARNING]  
> To access Heritage missions support, the package must be installed using the optional dependency "heritage". This operation
> will install additional dependencies that will change the current license of this software from **MIT** to **GPL-3.0**.

The package can be installed via pip:

```shell
pip install arepyextras-eo_products
```

with heritage support:

```shell
pip install arepyextras-eo_products[heritage]
```
