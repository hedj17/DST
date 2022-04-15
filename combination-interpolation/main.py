from BiLSTM.bilstm import bilstm
from DST.dst import dst
from DataConstants import DataConstants
from EST.est import est
from HST.hst import hst
from EFIM.EFIM import efim
from LT.lt import lt
from RST.rst import rst
from STISE.stise import stise
from error_calculation import error_calc
from kriging.krige_infill import ok
from missing_data_features_statistics import feature_statistic
from missing_data_statistics import missingDataStatistic
import numpy as np



def callInterpolationsMethods(eTag):
    # lt(eTag)
    # print("LT is done!")
    # ok(eTag)
    # print("OK is done!")
    # rst(eTag)
    # print("RST is done!")
    # est(eTag)
    # print("EST is done!")
    # hst(eTag)
    # print("HST is done!")
    #
    # stise(eTag)
    # print("STISE is done!")
    #
    # efim(eTag)
    # print("EFIM is done!")

    bilstm(eTag)
    print("BiLSTM is done!")

    learning_rates = {'xian': 0.001, 'chengdu': 0.001}
    epochs = {'xian': 1000, 'chengdu': 800}
    dst(1, eTag, learning_rates, epochs)
    print("CGDST is done!")

    learning_rates = {'xian': 0.002, 'chengdu': 0.002}
    epochs = {'xian': 2000, 'chengdu': 2000}
    dst(2, eTag, learning_rates, epochs)
    print("FGDST is done!")

    if eTag == 'e99':
        learning_rates = {'xian': 0.001, 'chengdu': 0.001}
        epochs = {'xian': 1000, 'chengdu': 800}
        dst(3, eTag, learning_rates, epochs)
        print("CGDST is done!")

        learning_rates = {'xian': 0.002, 'chengdu': 0.002}
        epochs = {'xian': 2000, 'chengdu': 2000}
        dst(4, eTag, learning_rates, epochs)
        print("FGDST is done!")


def eval_errors(cities, indicator, eTag):
    # methods=['BiLSTM']
    # fname={'BiLSTM':'bilstm'}
    methods = ['LT', 'OK', 'RST', 'EST', 'EFIM', 'HST', 'STISE', 'BiLSTM', 'CGDST', 'FGDST']
    fname = {'LT': 'lt', 'OK': 'ok', 'RST': 'rst', 'EST': 'est', 'HST': 'hst', 'CGDST': 'cgdst', 'FGDST': 'fgdst',
             'EFIM': 'efim', 'STISE': 'stise', 'BiLSTM': 'bilstm'}
    # The following code calculate RMSE, MAE and SMAPE of interpolation results
    for city in cities:
        i = 0
        estimated_values = []
        for method in methods:
            infilled_file = "./results/" + eTag + '/' + city + "-" + indicator + "-" + fname[method] + "-filling.csv"
            filled_data, real_data = error_calc(city, indicator, method, infilled_file, eTag)
            if i == 0:
                estimated_values.append(real_data)
                estimated_values.append(filled_data)
            else:
                estimated_values.append(filled_data)
            i = i + 1

        estimated_res_arr = np.array(estimated_values)
        np.savetxt("./results/" + eTag + '/' + city + "-" + indicator + "-estimatedValues.csv", estimated_res_arr,
                   delimiter=',')


if __name__ == "__main__":
    # e80: the PM2.5 interpolation for Xi'an and Chengdu. The results are used in Table 3 and Table 5 of the paper.
    # e82: the PM10 interpolation for Xi'an and Chengdu. The results are used in Table 4
    # e83: the PM2.5 interpolation for Xi'an and Chengdu. All stations are used and the results are used in Figure 12.
    # e84: the PM2.5 interpolation for Xi'an and Chengdu.
    #      All stations except the farthest stations are used and the results are used in Figure 12.
    # e85: the PM10 interpolation for Xi'an and Chengdu.
    #      All stations are used and the results are used in Figure 13.
    # e86: the PM10 interpolation for Xi'an and Chengdu.
    #      All stations except the farthest stations are used and the results are used in Figure 13.
    # e91, e93, e95, e98: to analyze the length of the column gap for different methods when
    #                     the gaps with length of one, three, five and eight, respectively.
    #                     All experiments perform the PM10 interpolation for Xi'an.
    # e99: to analyze the trade-off of DST between its spatial and temporal interpolation results.
    #      The experiment uses the same data and the experimental configuration as e80,
    #      but the DST models have no constraint on the spatial and temporal interpolation results.
    #      The results are used in Table 5.

    eTag = "e99"
    dc = DataConstants(eTag)
    cities, indicator, selectedStations = dc.getPublicData()

    # data_dir = r"./站点_20200101-20201231/"
    # dataCombine(data_dir, selectedStations, eTag)
    # print("Data combination is done!")

    value_list = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    probability = [0.25, 0.22, 0.17, 0.12, 0.08, 0.06, 0.04, 0.02, 0.02, 0.02]
    # probability = [0.08, 0.08, 0.08, 0.08, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]

    # extract_mask(value_list, probability, eTag)

    # the method extract_mask3 is specially used for e91, e93, e95 and e98
    # extract_mask3(3, eTag)

    print("Data extraction and mask are done!")

    dataType = 'masked'  # dataType = 'raw' or 'masked', which is used to indicate the purpose of feature_statistic
    # the 'feature_statistic' function is used to calculate the features of the masked data
    feature_statistic(dataType, eTag)
    print("Feature statistic of the masked data is done!")

    # the 'missingDataStatistic' function is used to calculate the number of
    # the missing values of the different stations.
    missingDataStatistic(eTag, dataType)
    print("The statistic of the missing data is done!")

    callInterpolationsMethods(eTag)

    eval_errors(cities, indicator, eTag)
    print('Done!!!')
