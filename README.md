# Project description
his project implements two dynamic spatiotemporal interpolation (DST) methods, i.e., coarse-grained DST (CGDST) and fine-grained DST (FGDST) using both temporal and spatial interpolation results. Different from other hybrid spatiotemporal interpolation methods, they make differences in the contribution of temporal and spatial interpolation results and assign them with different weights. Both CGDST and FGDST treat each missing value differently and fill it by considering the reliability of both temporal and spatial interpolation results in terms of the lengths of its column gap and row gap. CGDST treats each missing value in a continuous missing area equally and all missing values have same lengths of column and row gaps and FGDST goes beyond CGDST and treats each missing value differently based on its temporal distance to the nearest real observed values in both forward and backward directions.

In order to demonstrate the superiority of two dynamic interpolation methods proposed by this work, this project also implements eight interpolation methods, i.e., LT [1], OK [2], RST [3], EST [3], EFIM [4], HST [5], ST-ISE [6] and BiLSTM [7].

[1] Alexandre M. Bayen and Timmy Siauw. Chapter 14 - interpolation. In Alexandre M. Bayen and Timmy Siauw, editors, An Introduction to MATLAB? Programming and Numerical Methods for Engineers, pages 211(223). Academic Press, Boston, 2015.
[2] William A. Mart  nez, Carlos E. Melo, and Oscar O. Melo. Median polish kriging for space{time analysis of precipitation. Spatial Statistics, 19:1(20), 2017.
[3] Yan Li and Linan Wang. Research of spatiotemporal interpolation algorithm based on time series. Computer Science, 41(6A):414{424, 2014.
[4] Cai Bo, Zeyuan Shi, and Jianhui Zhao. Novel spatial and temporal interpolation
algorithms based on extended field intensity model with applications for sparse aqi. Multimedia Tools and Applications, 02 2021.
[5] Siling Peng. Developments of spatio-temporal interpolation methods for meteorological eleents. Master's thesis, Central South University, 2010.
[6] Shifen Cheng, Peng Peng, and Feng Lu. A lightweight ensemble spatiotemporal interpolation model for geospatial data. International Journal of Geographical Information Science, 34:1(24), 02 2020.
[7] Weitian Tong, Lixin Li, Xiaolu Zhou, Andrew Hamilton, and Kai Zhang. Deep learning pm2.5 concentrations with bidirectional lstm rnn. Air Quality, Atmosphere & Health, 12:411(423), 2019.
