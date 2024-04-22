import pandas as pd
import matplotlib.pyplot as plt

# Read the CSV file
data = pd.read_csv("ai4i2020_k.csv")

size_limit1=3800
size_limit2=4000

import numpy as np

def z_score_normalize(data):
    mean_val = np.mean(data)
    std_dev = np.std(data)
    normalized_data = [(x - mean_val) / std_dev for x in data]
    print(mean_val,std_dev)
    return normalized_data


def hdf_visualize():
    # Extract the desired columns
    air_temp = data.iloc[:, 3][size_limit1:size_limit2]  # 4th column
    col_5 = data.iloc[:, 4][size_limit1:size_limit2]      # process temp
    col_6 = data.iloc[:, 5][size_limit1:size_limit2]
    col_10 = data.iloc[:, 10][size_limit1:size_limit2]    #hdf
    diff=abs(air_temp-col_5)

    # Plotting
    plt.figure(figsize=(10, 6))
    plt.plot(z_score_normalize(air_temp), label='Air Temp', color='red')
    plt.plot(z_score_normalize(col_5), label='Process Temp', color='orange')
    plt.scatter(range(len(col_6)),z_score_normalize(col_6), label='RPM',color='green')     # will be the class label failuer is 1 and non-failure is 0 (preferaably make it 2) 
    plt.axhline(y=(1380 - 1552.845) / 179.429, color='b', linestyle='--',label='HDF Condition (RPM = 1380)')      # condition for hdf figure 

    m=0
    b=z_score_normalize(list(map(lambda x:1 if x>= 8.6 else 0 ,diff)))
    for i in range(len(col_10)):
        if b[i]<0:
            if m!=1:
                plt.scatter(i,b[i], color='yellow',alpha=1, label='HDF Condition (Diff Temp < 8.6)')
                m+=1
            else:
                plt.scatter(i,b[i], color='yellow',alpha=1)
        else:
            plt.scatter(i,b[i], color='yellow',alpha=0)

    m=0
    b=z_score_normalize(col_10)
    for i in range(len(col_10)):
        if b[i]>0:
            if m!=1:
                plt.scatter(i,b[i], color='black',alpha=1, label='HDF',marker='x')
                m+=1
            else:
                plt.scatter(i,b[i], color='black',alpha=1,marker='x')
        else:
            plt.scatter(i,b[i], color='black',alpha=0,marker='x')


def pwf_visualize():

    col_6 = data.iloc[:, 5][size_limit1:size_limit2]        # rpm
    col_7 = data.iloc[:, 6][size_limit1:size_limit2]        # torque
    col_11 = data.iloc[:, 11][size_limit1:size_limit2]      # pwf
    
    # convert col_6 to rad/s
    col_6 = [x * (2 * np.pi / 60) for x in col_6]
    # Plotting
    plt.figure(figsize=(10, 6))
    plt.scatter(range(len(col_6)),z_score_normalize(col_6), label='RPM',color='orange')     # will be the class label failuer is 1 and non-failure is 0 (preferaably make it 2) 
    plt.plot(z_score_normalize(col_7), label='Torque')

    # plt.axhline(y=(1380 - 1552.845) / 179.429, color='b', linestyle='--',label='PWF Condition (RPM = 1380)')      # condition for hdf figure 
    # m=0
    product= [1 if list(col_6)[i]*list(col_7)[i]>=9000 or list(col_6)[i]*list(col_7)[i]<=3500 else 0 for i in range(len(col_6))]
    # for i in range(20):
    #     print(list(col_6)[i],list(col_7)[i],product[i])
    b=product

    plt.plot(b, label='Product > 9000 or < 3500',color='red')

    m=0
    b=z_score_normalize(col_11)
    for i in range(len(col_11)):
        if b[i]>0:
            if m!=1:
                plt.scatter(i,b[i], color='black',alpha=1, label='PWF',marker='x')
                m+=1
            else:
                plt.scatter(i,b[i], color='black',alpha=1,marker='x')
        else:
            plt.scatter(i,b[i], color='black',alpha=0,marker='x')

def osf_visualize():

    col_6 = data.iloc[:, 7][size_limit1:size_limit2]        # total wear
    col_7 = data.iloc[:, 6][size_limit1:size_limit2]        # torque
    col_11 = data.iloc[:, 12][size_limit1:size_limit2]      # osf
    type= data.iloc[:, 2][size_limit1:size_limit2]          # type

    # Plotting
    plt.figure(figsize=(10, 6))
    plt.scatter(range(len(col_6)),z_score_normalize(col_6), label='Total Wear',color='orange')     # will be the class label failuer is 1 and non-failure is 0 (preferaably make it 2) 
    plt.plot(z_score_normalize(col_7), label='Torque')
    # m=0
    product= [1 if (list(col_6)[i]*list(col_7)[i]>=11000 and list(type)[i]=='L') or (list(col_6)[i]*list(col_7)[i]>=12000 and list(type)[i]=='M') or (list(col_6)[i]*list(col_7)[i]>=13000 and list(type)[i]=='H') else 0 for i in range(len(col_6))]
    b=product
    
    plt.plot(b, label='Product > 11000')

    m=0
    b=z_score_normalize(col_11)
    for i in range(len(col_11)):
        if b[i]>0:
            if m!=1:
                plt.scatter(i,b[i], color='black',alpha=1, label='OSF',marker='x')
                m+=1
            else:
                plt.scatter(i,b[i], color='black',alpha=1,marker='x')
        else:
            plt.scatter(i,b[i], color='black',alpha=0,marker='x')


def twf_visualize():

    col_6 = data.iloc[:, 7][size_limit1:size_limit2]        # total wear
    col_11 = data.iloc[:, 9][size_limit1:size_limit2]      # twf
    # type= data.iloc[:, 2][size_limit1:size_limit2]          # type

    # Plotting
    plt.figure(figsize=(10, 6))
    plt.scatter(range(len(col_6)),z_score_normalize(col_6), label='Total Wear',color='orange')     # will be the class label failuer is 1 and non-failure is 0 (preferaably make it 2) 
    # plt.plot(z_score_normalize(col_7), label='Torque')
    # m=0
    # product= [1 if (list(col_6)[i]*list(col_7)[i]>=11000 and list(type)[i]=='L') or (list(col_6)[i]*list(col_7)[i]>=12000 and list(type)[i]=='M') or (list(col_6)[i]*list(col_7)[i]>=13000 and list(type)[i]=='H') else 0 for i in range(len(col_6))]
    # b=product
    plt.axhline(y=(200 - 106.84) /59.79, color='b', linestyle='--',label='Tool wear time = 200 ')      # condition for hdf figure 
    plt.axhline(y=(240 - 106.84) /59.79, color='b', linestyle='--',label='Tool wear time = 240')      # condition for hdf figure 

    
    # plt.plot(b, label='Product > 11000')

    m=0
    # b=z_score_normalize(col_11)
    b=list(map(lambda x:x/1.7,z_score_normalize(col_11)))
    for i in range(len(col_11)):
        if b[i]>0:
            if m!=1:
                plt.scatter(i,b[i], color='black',alpha=1, label='TWF',marker='x')
                m+=1
            else:
                plt.scatter(i,b[i], color='black',alpha=1,marker='x')
        else:
            plt.scatter(i,b[i], color='black',alpha=0,marker='x')

# pwf_visualize()
# hdf_visualize()
# osf_visualize()
twf_visualize()

# Adding labels and title
plt.xlabel('Index')
plt.ylabel('Normalized Values')
plt.title('Data Visualization')
plt.legend()

# Show plot
plt.show()