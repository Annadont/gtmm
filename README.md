# gtmm
GainzTakerMoneyMakerrrrrr USER GUIDE

GTMM CAN PREDICT CRYPTO PRICES IN THE RANGE OF ALL VALUES IN THE "coinbase.csv" file in the "data" folder. Please choose the timestamp values from this file.

TO USE THIS SOFTWARE, PERFORM THE FOLLOWING STEPS:

1. Open "gtmm_test.py"
2. change start_timestamp to a value from the "Timestamp" column of "data/coinbase.csv"
3. change end_timestamp to a value from the "Timestamp" column of "data/coinbase.csv"
4. Make sure you have deleted "X_dat.dat" and "y_dat.dat" from this folder so that there is no saved data.
4. type "python gtmm_test.py" in any terminal window that has a working directory in this folder.

It will learn the data and then predict values. 

To predict against, simply type "python gtmm_test.py" in any terminal window The saved data will carry over.

To change the results slightly, edit the if statement in "calculate_y" in "gtmm_svm.py"
