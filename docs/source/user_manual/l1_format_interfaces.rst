.. _l1_prod_howto:

How to implement a new L1 Format
================================

To keep the EO Products module self-consistent as much as possible, new formats implemented should match the standard
implementation used for the already implemented ones, at least in terms of main files, exposed functions and interfaces.

Here is a list of the main conditions that should be matched to properly extend the *arepyextras.eo_products* package:

- The new format should have at least a `reader.py` and `utilities.py` files in the main *l1_products* folder
- Inside the `reader.py` file, an ``open_product()`` function should be available as a main function to read the product
  and return a ``EOL1ProductProtocol``-compliant object
- Inside the `reader.py` file, functions to read metadata and raster file given a Path to the file on disk must be implemented
- The ``read_metadata_function()`` (name is not binding) returned object should be consistent or at least similar to the objects used
  in already implemented format readers, although it's not bound to be compliant with any template or scheme

.. note::

   Internal enum defined in the `eo_products.common` module should be used to match enum-like properties across different
   implementations and formats. Be sure that exposed functions accepting or using enum-like inputs ALWAYS perform a conversion
   to the internal version of that enum type, if present.
