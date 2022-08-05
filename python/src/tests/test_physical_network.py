from network import physical_network
import numpy as np


def test_get_valid_dcs_customer():
    np.random.seed(0)
    # Given
    physnet = physical_network.PhysicalNetwork(3, 3, 2, 100, 50)
    # When
    valid_c0 = physnet.get_valid_dcs(0)
    valid_c1 = physnet.get_valid_dcs(1)
    valid_c2 = physnet.get_valid_dcs(2)

    # Then
    print("the matrix")
    print(physnet.dcs_per_customer_array)
    print(valid_c0)
    print(valid_c1)
    print(valid_c2)
    assert (valid_c0 == np.array([1, 2])).all()
    assert (valid_c1 == np.array([1, 2])).all()
    assert (valid_c2 == np.array([0, 2])).all()
