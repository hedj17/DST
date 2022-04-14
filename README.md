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

# The implementation details
 this project, the interpolate() function of pandas DataFrame was employed to implement LT. To implement linear interpolation, the parameter method was set to 'linear' and the parameter limit_direction was set to 'both' to fill missing values in both forward and backward direction. This is helpful when the first value or the last value of time series data is missing. OK was implemented by using PyKrige, which is a toolkit for Python and it supports 2D and 3D ordinary and universal kriging. It is worth noticing that OK requires at least three observed real data values to predict missing values and the average values of the observed real values were used if the number of the observed real values is less than three. RST was implemented by filling missing values with LT first and then revised the interpolation results based on OK. More precisely, for each value filled by LT, it is revised by using existing observed real values and the LT interpolation results of other missing values based on OK. EST was implemented by using OrdinaryKriging3D() method of PyKrige. For each missing value, EST fills it by using latitudes and longitudes of all observed values within two hours. Moreover, EST employs the time distance between the missing values and the other observed values within two hours. The time distance is calculated by dividing the time difference by the time step, which is one in the experiments. To increase the contribution of temporal relationship, the value of time difference is magnified 10 times. EFIM considered the effect of all nearby monitoring stations within two hours for interpolation. HST was implemented based on linear regression of the interpolation results of LT and OK. The linear regression of HST is implemented by sklearn.linear_model.LinearRegression. 

For the SES model, ST-ISE selected the optimal historical time window first and then assigned different weights to the samples in the window to calculated the temporal interpolation results. For the spatial interpolation, ST-ISE also considered the effect of all nearby monitoring stations in the datasets. BiLSTM considered three nearest monitoring stations and the observed values four hours before and after the interpolation time. This work selected all stations which did not have missing values four hours before and after the interpolation time first and selected three nearest stations. If there were less three stations, the average values of the existing observed values were used as the interpolation results. ST-ISE and BiLSTM were implemented by PyTorch, which a widely used deep learning library developed by Facebook.

The temporal and spatial interpolation results used by two DST models were estimated by LT and OK, respectively. Moreover, the DST models are also implemented by PyTorch to perform the regression analysis with a purpose of finding an optimal dynamic weight coefficient to balance the contribution of the spatial and temporal interpolation results. More precisely, the regression model was created using torch.nn.Linear and the sigmoid function, which is PyTorch sub-module for neural network. In addition, MSE (Mean Squared Error) was used as loss function and Adam was employed as optimizer to find regression coefficients when the loss function is minimum. To obtain optimal DST models, different small learning rates ranging from 0.001 to 0.002 and different the number of epochs ranging from 500 to 3000 were used to train the models on different datasets.

# The main Python packages used in the project
- numpy                              1.16.4
- pandas                             1.0.1
- pip                                19.1.1
- PyKrige                            1.6.1
- scikit-learn                       0.24.1
- scipy                              1.2.1
- seaborn                            0.9.0
- torch                              1.11.0

# The main experiments of the project
1. e80: the PM2.5 interpolation for Xi'an and Chengdu. The results are used in Table 3 and Table 5 of the paper.
2. e82: the PM10 interpolation for Xi'an and Chengdu. The results are used in Table 4
3. e83: the PM2.5 interpolation for Xi'an and Chengdu. All stations are used and the results are used in Figure 12.
4. e84: the PM2.5 interpolation for Xi'an and Chengdu. All stations except the farthest stations are used and the results are used in Figure 12.
5. e85: the PM10 interpolation for Xi'an and Chengdu. All stations are used and the results are used in Figure 13.
6. e86: the PM10 interpolation for Xi'an and Chengdu. All stations except the farthest stations are used and the results are used in Figure 13.
7. e91, e93, e95, e98: to analyze the length of the column gap for different methods when the gaps with length of one, three, five and eight, respectively. All experiments perform the PM10 interpolation for Xi'an.
8. e99: to analyze the trade-off of DST between its spatial and temporal interpolation results.
9. The experiment uses the same data and the experimental configuration as e80, but the DST models have no constraint on the spatial and temporal interpolation results. The results are used in Table 5.

# The main files and directories of the project
- main.py: the starting point of the project.
- data/: contains the data used in the experiments
- results/: contains the results of the experiments
- LT/lt.py: the implementation of LT
- kriging/krige_infill.py: the implementation of OK
- RST/rst.py: the implementation of RST
- EST/est.py: the implementation of EST
- EFIM/efim.py: the implementation of EFIM
- HST/hst.py: the implementation of HST
- ST-ISE/etise.py: the implementation of ST-ISE
- BiLSTM/bilstm.py: the implementation of BiLSTM
- dst/dst.py: the implementation of DST
- data_extraction_mask.py: the processes of the raw dataset extraction and data mask
- missing_data_features_statistics.py: calculates the features of each missing values

# run project
python main.py
