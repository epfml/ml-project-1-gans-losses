import helpers as hlp
import numpy as np

def test_data_processing():
    x = np.array([[1.0, 1.0, 6.0, 3.0],
                  [2.0, 1.0, np.nan, np.nan],
                  [3.0, 1.0, np.nan, 5.0]])
    print(x)
    y = np.array([1.0, 0.0, 1.0])
    x_exp = np.array([[-1.2247448714, -1.2247448714],
                      [0.0, 0.0],
                      [1.2247448714, 1.2247448714]])
    y_exp = y
    x_res, y_res = hlp.preprocess_data(x, y, nan_rate_threshold=0.5, in_place=False)

    assert np.allclose(x_res, x_exp)
    assert np.allclose(y_res, y_exp)