.. _l1_generic_func:

:html_theme.sidebar_secondary.remove:

L1 Product: Generic Function Quickstart
=======================================

Each supported external format reader is designed :ref:`following the high-level guideline<l1_prod_howto>` in order to have
a unifying set of common functionalities and objects that help the user accessing the product data.

By design, information is split in product-wise features and channel-wise features. The python object representing the product
is quite generic and it's loosely bound to be protocol-compliant to guarantee a unified starting point for each supported format
from which derive channel-specific data. In particular, channel data are split between metadata and raster data: the first is
managed using a ``read_metadata``-like function while the latter is accessed using a ``read_data``-like.

Opening a product
-----------------

To open an existing product from disk, an ``open_product()`` function is available for the format of choice and can be imported
from its `reader` module. This commonly shared interface can be used as shown in the following example:

The loaded python object is compliant to the ``EOL1ProductProtocol`` meaning that it has a pre-defined minimum set of attributes
and methods that are the same across the module for each implemented L1 reader.

.. code-block:: python

    from arepyextras.eo_products.format_of_choice.l1_products.reader import open_product

    path_to_product = r"path_to_selected_product"
    product = open_product(path_to_product)

    # set of commonly available properties this object
    product.acquisition_time        # returns the acquisition time of the SAR product
    product.data_list               # returns a list of the raster data paths, one for each channel
    product.channels_number         # returns the number of available channels
    product.channels_list           # returns the unique identifier of each channel, may it be its name or number

A method to retrieve channel dependent data should be implemented, returning the absolute path of the metadata and/or raster
files corresponding to the selected channel. In principle, several other files can be associated to a given channel id and
their path can be returned by this method keeping the protocol compliance, but this feature is strictly format-dependent.

The user can then read the metadata and raster data files using the dedicated functions available in the `reader` module, a
``read_metadata()`` and ``read_data()`` like functions (names are not binding). These functions should at least have a
single argument that is the path to the file to be read. The ``read_data()``-like function should have an additional argument
to specify the region of the raster file to be read, to avoid reading the whole image that can cause RAM issues.
Additional format-specific arguments can be added if needed.
