# SPDX-FileCopyrightText: Aresys S.r.l. <info@aresys.it>
# SPDX-License-Identifier: MIT

"""Checking protocol compliance for main products objects for each implemented format"""

import unittest

from arepyextras.eo_products.common.protocols import EOL1ProductProtocol
from arepyextras.eo_products.iceye.l1_products.utilities import ICEYEProduct
from arepyextras.eo_products.novasar.l1_products.utilities import NovaSAR1Product
from arepyextras.eo_products.safe.l1_products.utilities import S1Product
from arepyextras.eo_products.saocom.l1_products.utilities import SAOCOMProduct


class ProductProtocolComplianceTest(unittest.TestCase):
    """Testing ProductFolderManager class"""

    def test_product_compliance_safe(self) -> None:
        """Assessing protocol compliance"""
        assert isinstance(S1Product, EOL1ProductProtocol)

    def test_product_compliance_novasar(self) -> None:
        """Assessing protocol compliance"""
        assert isinstance(NovaSAR1Product, EOL1ProductProtocol)

    def test_product_compliance_iceye(self) -> None:
        """Assessing protocol compliance"""
        assert isinstance(ICEYEProduct, EOL1ProductProtocol)

    def test_product_compliance_saocom(self) -> None:
        """Assessing protocol compliance"""
        assert isinstance(SAOCOMProduct, EOL1ProductProtocol)


if __name__ == "__main__":
    unittest.main()
