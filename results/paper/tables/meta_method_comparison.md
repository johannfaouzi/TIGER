| Method                               |   Mean accuracy |   Median accuracy |    Std |   Mean rank | Deployable   |
|:-------------------------------------|----------------:|------------------:|-------:|------------:|:-------------|
| Meta-oracle (test-time, upper bound) |          0.8916 |            0.9468 | 0.1214 |      1.8415 | False        |
| Oracle (test-time, upper bound)      |          0.8922 |            0.9446 | 0.1213 |      2.6232 | False        |
| Auto-selected (this paper's rule)    |          0.8838 |            0.9382 | 0.1241 |      4.4190 | True         |
| TabICLv2 (stacked)                   |          0.8789 |            0.9267 | 0.1301 |      4.8944 | True         |
| CAWPE hard-vote                      |          0.8777 |            0.9316 | 0.1247 |      5.8908 | True         |
| Best base learner (CV-picked)        |          0.8778 |            0.9303 | 0.1279 |      5.9789 | True         |
| CAWPE                                |          0.8737 |            0.9253 | 0.1281 |      6.8873 | True         |
| Logistic regression (stacked)        |          0.8721 |            0.9257 | 0.1367 |      6.9261 | True         |
| LightGBM (stacked)                   |          0.8655 |            0.9117 | 0.1382 |      7.6972 | True         |
| Hard majority vote                   |          0.8680 |            0.9159 | 0.1303 |      7.8415 | True         |