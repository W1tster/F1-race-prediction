F1 RACE POSITION PREDICTION MODEL
==================================

HOW TO USE
----------

Run the test script to predict any race:

    python test_sao_paulo_gp.py

To predict a different race, copy the script and change the location/year on line 52.


REQUIRED FILES
--------------

- f1_model.pth (trained model)
- f1_deep_neural_network.py (model definition)


MODEL PERFORMANCE
-----------------

Mean Absolute Error: 2.53 positions
Within ±1 position: 34.30%
Within ±2 positions: 57.97%
Within ±3 positions: 75.85%


DEPENDENCIES
------------

    pip install torch pandas numpy scikit-learn
