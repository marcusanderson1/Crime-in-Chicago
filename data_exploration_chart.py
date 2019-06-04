import sys
sys.path
import db_connection as db
import queries as q
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import math
import pymysql
import datetime 
plt.style.use('seaborn')
pymysql.install_as_MySQLdb()
import seaborn as sns
import geopy.distance

conn = db.open_sql_connection()


query = q.sql
community_area = []
categorical_fields = ['CommunityArea','PrimaryType']
primary_type =  []

def read_data(query, conn):
    '''
    Purpose of this method is to read in the source data from MySQL
    '''
    
    print('Read data started', datetime.datetime.now())
    print(query)
    data = pd.read_sql(query , conn)
    #CREATE DATE INDEX
    data.Date = pd.to_datetime(data.Date,format='%m/%d/%Y %I:%M:%S %p')
    data.index = pd.DatetimeIndex(data.Date)

    #Cast the fields to a float data type
    data['District'] = data.District.astype(float)
    data['Ward'] = data.Ward.astype(float)
    data['Beat'] = data.Beat.astype(float)
    
    print('Read data finished', datetime.datetime.now())
    return data


def time_series_graph_month(data):
    #Plot graph showing monthly totals
    plt.figure(figsize=(15,10))
    data.resample('M').size().plot(legend=False) #Month
    #Chart titles, x axis labels and y axis labels
    plt.title('Number of crimes per month')
    plt.xlabel('Years')
    plt.ylabel('Number of crimes')
    plt.savefig('Charts/time_series_graph_month.png', bbox_inches = 'tight')
    plt.show()

def time_series_graph_year(data):
    #Plot graph showing annual volumes
    plt.figure(figsize=(15,10))
    data.resample('Y').size().plot(legend=False) #Year
    #Chart titles, x axis labels and y axis labels
    plt.title('Number of crimes per Year')
    plt.xlabel('Years')
    plt.ylabel('Number of crimes')
    plt.savefig('Charts/time_series_graph_year.png', bbox_inches = 'tight')
    plt.show()
   
def table_primary_type_volume(data):
    
    #Here we get the crime volumes by primary type across the whole dataset
    data_pimary_type = data.groupby(['PrimaryType']).size().reset_index(name='Volume') #grouping the data by Primary Type and counting volumes
    data_pimary_type['Percent'] = data_pimary_type['Volume'] / len(data)#Calculate the percent
    data_pimary_type = data_pimary_type.sort_values(by='Percent', ascending = False)#Sort the data by highest percentage
    data_pimary_type.to_csv('Data_Exports/table_primary_type_volume.csv')
    
def table_primary_type_volume05to09(data):
    
    years = ['2005','2006','2007','2008','2009']#filter the data using the years in list
    data05to09 = data[data['Year'].isin(years)]#filtering data
    data_pimary_type = data05to09.groupby(['PrimaryType']).size().reset_index(name='Volume')
    data_pimary_type['Percent'] = data_pimary_type['Volume'] / len(data05to09)#Calculate the percent
    data_pimary_type = data_pimary_type.sort_values(by='Percent', ascending = False)#Sort the data by highest percentage
    data_pimary_type.to_csv('Data_Exports/table_primary_type_volume05to09.csv')
    
    
def table_primary_type_volume10to14(data):
    
    years = ['2010','2011','2012','2013','2014']#filter the data using the years in list
    data10to14 = data[data['Year'].isin(years)]#filtering data
    data_pimary_type = data10to14.groupby(['PrimaryType']).size().reset_index(name='Volume')
    data_pimary_type['Percent'] = data_pimary_type['Volume'] / len(data10to14)
    data_pimary_type = data_pimary_type.sort_values(by='Percent', ascending = False)#Sort the data by highest percentage
    data_pimary_type.to_csv('Data_Exports/table_primary_type_volume10to14.csv')
    
def cut_down_data(data):
    #Perform the rest of the analysis on a cut down version of the data
    global primary_type
    global community_area
    years = ['2010','2011','2012','2013','2014']
    data = data[data['Year'].isin(years)] 
    
    #How we get the top ten primary crime types
    data_pimary_type = data.groupby(['PrimaryType']).size().reset_index(name='Volume')
    data_pimary_type['Percent'] = data_pimary_type['Volume'] / len(data)
    data_pimary_type = data_pimary_type.sort_values(by='Percent', ascending = False)
    primary_type = list(data_pimary_type['PrimaryType'].head(10))
    
    #How we get the top 10 areas with crime

    community_area = list(data['CommunityArea'].value_counts()[:10].index)
    
    #Filtering the data to focus on Top 10 Primary Types and Top 10 Community Areas
    data = data[data['PrimaryType'].isin(primary_type)]
    data = data[data['CommunityArea'].isin(community_area)]
    
    
    return data , community_area, primary_type
    


def get_analysis(data):
    global categorical_fields
    #Grouping by Comunity Area and Primary Type to get some analysis
    analysis_data = data.groupby(categorical_fields).size().reset_index(name='Volume')
    return analysis_data


def chart_data(analysis_data, community_area):

    
    primary_type_arr = []
    vol_arr = []
    #loop through list outputting charts
    for area in community_area:
      
        area_data = analysis_data[analysis_data['CommunityArea'] == area]
        
        print ('Area ' + area)
        for index,row in area_data.iterrows():
            
            vol_arr.append(row['Volume'])
            primary_type_arr.append(row['PrimaryType'])
    
        print(primary_type_arr)
        print(vol_arr)        
        
        y_pos = np.arange(len(vol_arr))
        plt.figure(figsize=(11,5))
        plt.bar(y_pos,vol_arr , align='center', alpha=0.5)
        plt.xticks(y_pos,primary_type_arr,rotation = 90)
        plt.ylabel('Volume')
        plt.xlabel('Primary Type')
        plt.title('Community Area '+ area )
        plt.savefig('Community_Area_Charts/BarChart/'+area+'_Community_Area_barchart.png', bbox_inches = 'tight')
        plt.show()
        
        vol_arr = [] #reset list to empty
        primary_type_arr = []
        

def time_series_community_area_graph_year(data10to14, community_area):

    print('--------running---------')
    print ('Count community ' +str(len(community_area)))
    #loop through community area and produce charts
    for area in community_area:
        analysis = data10to14[data10to14['CommunityArea']==area]
        
        plt.figure(figsize=(11,5))
        analysis.resample('Y').size().plot(legend=False)
        plt.title('Number of crimes in area '+area+' per Year')
        plt.xlabel('Years')
        plt.ylabel('Volume of crimes')
        plt.savefig('Community_Area_Charts/LineChart/'+area+'_CommunityArea_linechart.png', bbox_inches = 'tight')
        plt.show() 
        
        del analysis

    
def chart_primary_type(data10to14):
    '''
    Produce one bar chart with all primary type volumes
    '''
    primary_type_arr = []
    vol_arr = []
    
    analysis_primary_type = data10to14.groupby(['PrimaryType']).size().reset_index(name='Volume')
    
    for index,row in analysis_primary_type.iterrows():
            
        vol_arr.append(row['Volume'])
        primary_type_arr.append(row['PrimaryType'])
    
        #print(primary_type_arr)
        #print(vol_arr)        
        
    y_pos = np.arange(len(vol_arr))
    plt.figure(figsize=(11,5))
    plt.bar(y_pos,vol_arr , align='center', alpha=0.5)
    plt.xticks(y_pos,primary_type_arr,rotation = 90)
    
    plt.ylabel('Volume')
    plt.title('Top 10 Primary Type Crimes between 2010 and 2014' )
    plt.savefig('Primary_Type_Charts/Primary_Type_barchart.png', bbox_inches = 'tight')
    plt.show()
    
    vol_arr = [] #reset list to empty
    primary_type_arr = []


def time_series_year_primary_type(data10to14):
    #produce one line chart with a line percrime type
    #create dataframe to get the volumes for year x primary type
    year_primary_type = data10to14.groupby(['Year','PrimaryType']).size().reset_index(name='Volume')
    
    year = list(year_primary_type['Year'].unique())
    
    assault = get_primary_type_volume(year_primary_type,'ASSAULT')
    theft = get_primary_type_volume(year_primary_type,'THEFT')
    narcotics = get_primary_type_volume(year_primary_type,'NARCOTICS')
    battery = get_primary_type_volume(year_primary_type,'BATTERY')
    burglary = get_primary_type_volume(year_primary_type,'BURGLARY')
    criminal_damage = get_primary_type_volume(year_primary_type,'CRIMINAL DAMAGE')
    deceptive_practice = get_primary_type_volume(year_primary_type,'DECEPTIVE PRACTICE')
    motor_vehicle_theft = get_primary_type_volume(year_primary_type,'MOTOR VEHICLE THEFT')
    other_offense = get_primary_type_volume(year_primary_type,'OTHER OFFENSE')
    robbery = get_primary_type_volume(year_primary_type,'ROBBERY')
    
    #plot graph, different colour lines 
    plt.figure(figsize=(15,10))
    plt.plot(year, assault, color='g',label='Assault')
    plt.plot(year, theft, color='orange',label='Theft')
    plt.plot(year, narcotics, color='b',label = 'Narcotics')
    plt.plot(year, battery, color='r',label = 'Battery')
    plt.plot(year, burglary, color='c',label = 'Burglary')
    plt.plot(year, criminal_damage, color='m',label = 'Criminal Damage')
    plt.plot(year, deceptive_practice, color='k',label = 'Deceptive Practice')
    plt.plot(year, motor_vehicle_theft, color='darksalmon',label = 'Motor Vehicle Theft')
    plt.plot(year, other_offense, color='y',label= 'Other Offense')
    plt.plot(year, robbery,label = 'Robbery')
    plt.legend(loc="upper right" , fancybox=True,framealpha=0.5)

    plt.xlabel('Years')
    plt.ylabel('Volume')
    plt.title('Volume of crimes by Primary Type')
    plt.savefig('Primary_Type_Charts/Primary_Type_by_Year_linechart.png', bbox_inches = 'tight')
    plt.show()
    
    
def get_primary_type_volume(year_primary_type,primary_type):
    
    vol_arr = []

    for index , row in year_primary_type[year_primary_type['PrimaryType']==primary_type].iterrows():
        volume = row['Volume']
        vol_arr.append(volume)

    return vol_arr

def rolling_yearly_primary_type(data, community_area,primary_type ):
    
    print(data.shape , len(community_area) , len(primary_type))
    
    years = ['2009','2010','2011','2012','2013','2014'] #include 2009 in order to plot 2010as it is rolling by 365 
    data= data[data['Year'].isin(years)]
    data=data[data['CommunityArea'].isin(community_area)]
    data=data[data['PrimaryType'].isin(primary_type)]
    print(data.shape)
   
    pivot_data = data.pivot_table('ID', aggfunc=np.size, columns='PrimaryType', index=data.index.date, fill_value=0)
    print(pivot_data)
    pivot_data.index = pd.DatetimeIndex(pivot_data.index)
    plo = pivot_data.rolling(365).sum().plot(figsize=(12, 20), subplots=True, layout=(-1, 3), sharex=False, sharey=False)
   
    
    
def heatmap_primary_type_location_desc(data10to14):
    
    #Produce heat map showing primary type by location

    heatmapdata = data10to14[['CommunityArea','PrimaryType']]
    heatmapdata = heatmapdata.groupby(['CommunityArea','PrimaryType']).size().reset_index(name='Volume')
    pivot_heatmap = heatmapdata.pivot('PrimaryType','CommunityArea','Volume')
    fig, ax = plt.subplots(figsize=(8,5)) 
    
    heatmap_display = sns.heatmap(pivot_heatmap,annot=True,fmt='d',cmap='BuPu',linewidth=0.6)
    
    fig = heatmap_display.get_figure()
    fig.savefig('Heat_Maps/PrimaryType_by_CommunityArea.png',dpi=400,bbox_inches="tight")
    
def heatmap_year_community_area(data10to14):
    
    #Produce  heat map year by community area
    
    heatmapdata = data10to14.groupby(['Year','CommunityArea']).size().reset_index(name='Volume')
    pivot_heatmap =  heatmapdata.pivot('Year','CommunityArea','Volume')
    fig, ax = plt.subplots(figsize=(8,3)) 
    heatmap_display = sns.heatmap(pivot_heatmap,annot=True,fmt='d',cmap='BuPu',linewidth=0.6)
    
    fig = heatmap_display.get_figure()
    fig.savefig('Heat_Maps/Year_by_CommunityArea.png',dpi=400,bbox_inches="tight")    

def heatmap_month_community_area(data10to14):
    #Produce  heat map month by community area
    
    months = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec'] # month list, used for plotting on heat map
    
    heatmapdata = data10to14.groupby(['Month','CommunityArea']).size().reset_index(name='Volume')
    pivot_heatmap =  heatmapdata.pivot('Month','CommunityArea','Volume')
    fig, ax = plt.subplots(figsize=(8,8)) 
    ax = sns.heatmap(pivot_heatmap,annot=True,fmt='d',cmap='BuPu',linewidth=0.6, yticklabels=months)   
    
    fig = ax.get_figure()
    fig.savefig('Heat_Maps/Month_by_CommunityArea.png',dpi=400,bbox_inches="tight")

def heatmap_weekday_community_area(data10to14):
    #Produce heat map showing weekday by community area
    
    days = ['Mon','Tue','Wed','Thu','Fri','Sat','Sun'] #weekday list, used for plotting on heat map
    heatmapdata = data10to14.groupby(['Weekday','CommunityArea']).size().reset_index(name='Volume')
    pivot_heatmap =  heatmapdata.pivot('Weekday','CommunityArea','Volume')
    fig, ax = plt.subplots(figsize=(8,5)) 
    heatmap_display = sns.heatmap(pivot_heatmap,annot=True,fmt='d',cmap='BuPu',linewidth=0.6,yticklabels = days )
    
    fig = heatmap_display.get_figure()
    fig.savefig('Heat_Maps/Weekday_by_CommunityArea.png',dpi=400,bbox_inches="tight")

    
def heatmap_hour_community_area(data10to14):
    heatmapdata = data10to14.groupby(['Hour','CommunityArea']).size().reset_index(name='Volume')
    pivot_heatmap =  heatmapdata.pivot('Hour','CommunityArea','Volume')
    fig, ax = plt.subplots(figsize=(10,10)) 
    heatmap_display = sns.heatmap(pivot_heatmap,annot=True,fmt='d',cmap='BuPu',linewidth=0.6)
    
    fig = heatmap_display.get_figure()
    fig.savefig('Heat_Maps/Hour_by_CommunityArea.png',dpi=400,bbox_inches="tight")
    
def heatmap_hour_primary_type(data10to14):
    heatmapdata = data10to14.groupby(['Hour','PrimaryType']).size().reset_index(name='Volume')
    pivot_heatmap =  heatmapdata.pivot('Hour','PrimaryType','Volume')
    fig, ax = plt.subplots(figsize=(10,10)) 
    heatmap_display = sns.heatmap(pivot_heatmap,annot=True,fmt='d',cmap='BuPu',linewidth=0.6)
    
    fig = heatmap_display.get_figure()
    fig.savefig('Heat_Maps/Hour_by_PrimaryType.png',dpi=400,bbox_inches="tight")
    
def heatmap_month_primary_type(data10to14):
    months = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']
    
    heatmapdata = data10to14.groupby(['Month','PrimaryType']).size().reset_index(name='Volume')
    pivot_heatmap =  heatmapdata.pivot('Month','PrimaryType','Volume')
    fig, ax = plt.subplots(figsize=(8,8)) 
    heatmap_display = sns.heatmap(pivot_heatmap,annot=True,fmt='d',cmap='BuPu',linewidth=0.6,yticklabels=months)
    
    
    fig = heatmap_display.get_figure()
    fig.savefig('Heat_Maps/Month_by_PrimaryType.png',dpi=400,bbox_inches="tight")
    
def heatmap_weekday_primary_type(data10to14):
    days = ['Mon','Tue','Wed','Thu','Fri','Sat','Sun']
    heatmapdata = data10to14.groupby(['Weekday','PrimaryType']).size().reset_index(name='Volume')
    pivot_heatmap =  heatmapdata.pivot('Weekday','PrimaryType','Volume')
    fig, ax = plt.subplots(figsize=(8,5)) 
    heatmap_display = sns.heatmap(pivot_heatmap,annot=True,fmt='d',cmap='BuPu',linewidth=0.6,yticklabels = days )

    fig = heatmap_display.get_figure()
    fig.savefig('Heat_Maps/Weekday_by_PrimaryType.png',dpi=400,bbox_inches="tight")


def calculate_mean_geo_locations(data10to14):
    
    geographic_mean_data = data10to14.groupby(['CommunityArea','PrimaryType'])['Latitude','Longitude'].mean().reset_index()
    return geographic_mean_data


def geographical_data_analysis(data , data10to14):
    #here we get the mean lonitude and latitude for the different primary types in their community area
    #the data parameter is a cut down version of the data either 10 to 14 or 15 to 16 - depending on which part
    #is calling this method
    
    #We use this dataset to get the average distances as we are basing the data all on this file
    print(data.PrimaryType.unique())
    
    print('Shape data10to14'+ str(data10to14.shape))
    print ('data ',data.shape)
    geog_avg = calculate_mean_geo_locations(data10to14)
    
    mising_locations=[]
    
    for index, row in data.iterrows():

        longitude = row['Longitude']
        latitude = row['Latitude']
        
        if (math.isnan(longitude) or math.isnan(latitude)):
            #print(row['ID'],row['CommunityArea'],row['PrimaryType'],row['Latitude'], row['Longitude'])
            for index,row_geo in geog_avg.iterrows():
                if (row['CommunityArea'] == row_geo['CommunityArea'] and row['PrimaryType'] == row_geo['PrimaryType']):
                    mising_locations.append([row['ID'],row_geo['CommunityArea'],row_geo['PrimaryType'],row_geo['Latitude'], row_geo['Longitude']])
        
    df = pd.DataFrame(mising_locations,columns=['ID_','CommunityArea_','PrimaryType_','Latitude_','Longitude_'])
    print('df',df.shape)
    data_enriched = pd.merge(data, df, left_on='ID', right_on='ID_',how='outer')   
    print('data_enriched 1 ', data_enriched.shape)
    print('data 2', data.shape)
    
    data_enriched.Longitude.fillna(value=data_enriched.Longitude_,inplace=True)
    data_enriched.Latitude.fillna(value=data_enriched.Latitude_,inplace=True)
    
    print(data_enriched.columns)
    print(data_enriched.isna().sum())
    
    data_enriched.drop(columns=['ID_','CommunityArea_','PrimaryType_','Latitude_','Longitude_'], axis=1,inplace=True)
    print ('geog_ shape', geog_avg.shape)
    print('data_enriched 2',data_enriched.shape)
    
    data_enriched = pd.merge(data_enriched,geog_avg,left_on=['CommunityArea','PrimaryType'],right_on=['CommunityArea','PrimaryType'],how='inner')
    print('data_enriched 3',data_enriched.shape)
    
    return data_enriched
    
def calculate_distance_from_mean(data_enriched):
    geo_distance = []
    
    for index, row in data_enriched.iterrows():
        
        #Coordinates 1 will take the average location
        avg_longitude = row['Longitude_y']
        avg_latitude = row['Latitude_y']
        #Coordinates 2 will take the actual location
        actual_longitude = row['Longitude_x']
        actual_latitude = row['Latitude_x']
        
        
        coordinate1 = (avg_latitude , avg_longitude)
        coordinate2 = (actual_latitude , actual_longitude)
        #print (coordinate1 , coordinate2)
        
        distance = geopy.distance.distance(coordinate1,coordinate2).miles
        
        #allocating the distance from the mean hotspot
        if (distance <= 0.25):
            distance_band = 'QuarterMile'
        elif(distance > 0.25 and  distance <= 0.5):
            distance_band = 'HalfMile'
        elif(distance > 0.5 and  distance <= 0.75):
            distance_band = 'ThreeQuarterMile'
        elif(distance > 0.75 and  distance <= 1.0):
            distance_band = 'OneMile'
        elif(distance > 1.0):
            distance_band = 'GreaterThanMile'
        else:
            distance_band = 'blank'
            
        geo_distance.append([row['ID'] , distance, distance_band])
    
    geo_distance_df = pd.DataFrame(geo_distance, columns=['ID','Distance','DistanceBand'])
    
    data_enriched_dist = pd.merge(data_enriched,geo_distance_df,left_on='ID', right_on='ID', how='inner' )
    
    return data_enriched_dist

           
  
        
        
def calculate_distance_km(mean_cord,current_record_cord):
    #calculate the distance from the average crime hotspot
    return geopy.distance.distance(mean_cord,current_record_cord).km


def calculate_percentage_arrests(data10to14,community_area ):
   #calculate the arrest rates split by community area and primary type
   #these charts will be added to the appendix
    
    arrested = data10to14[data10to14['Arrest']=='True']
    crime_volumes = data10to14.groupby(['CommunityArea','PrimaryType']).size().reset_index(name='Volume')
    arrested_volumes = arrested.groupby(['CommunityArea','PrimaryType',]).size().reset_index(name='Volume')
    
    result = pd.merge(crime_volumes,arrested_volumes,how='inner',on= ['CommunityArea','PrimaryType'])
    result['Percent'] = round(((result.Volume_y / result.Volume_x) * 100),2)
    arrest_percent_data(result, community_area)
    
def arrest_percent_data(analysis_data, community_area):
    
    primary_type_arr = []
    vol_arr = []
    
    for area in community_area:
      
        area_data = analysis_data[analysis_data['CommunityArea'] == area]
        
        print ('Area ' + area)
        for index,row in area_data.iterrows():
            
            vol_arr.append(row['Percent'])
            primary_type_arr.append(row['PrimaryType'])
    
        print(primary_type_arr)
        print(vol_arr)        
        
        y_pos = np.arange(len(vol_arr))
        plt.figure(figsize=(11,5))
        plt.bar(y_pos,vol_arr , align='center', alpha=0.5,color=(0.3, 0.75, 1, 1),  edgecolor='blue')
        plt.xticks(y_pos,primary_type_arr,rotation = 90)
        plt.ylabel('Percent %')
        plt.title('Community Area '+ area )
        plt.xlabel('Primary Types')
        plt.savefig('Community_Area_Charts/BarChartPercent/'+area+'_Community_Area_Arrest_Rate_barchart.png', bbox_inches = 'tight')
        plt.show()
        
        vol_arr = [] #reset list to empty
        primary_type_arr = []  
    
        
def overall_arrest_percentage(data10to14):
    #Here we have an arrest chart for all our 10/14 dataset, to be used in report
    chart_title = 'Overall_Arrest_Rate.png'
    arrested = data10to14[data10to14['Arrest']=='True']
    crime_volumes = data10to14.groupby(['PrimaryType']).size().reset_index(name='Volume')
    arrested_volumes = arrested.groupby(['PrimaryType',]).size().reset_index(name='Volume')
    
    result = pd.merge(crime_volumes,arrested_volumes,how='inner',on= ['PrimaryType'])
    result['Percent'] = round(((result.Volume_y / result.Volume_x) * 100),2)
    
    y_pos = np.arange(len(result['Percent']))
    
    percent_arr = np.array(result['Percent'])
    primary_type_arr = np.array(result['PrimaryType'])
    
    plt.figure(figsize=(11,5))
    plt.bar(y_pos,percent_arr , align='center', alpha=0.5,color=(0.3, 0.75, 1, 1),  edgecolor='blue')
    plt.xticks(y_pos,primary_type_arr,rotation = 90)
    plt.ylabel('Percent %')
    plt.title('Overall Arrest Rate' )
    plt.xlabel('Primary Types')
    plt.savefig('Charts/'+chart_title, bbox_inches = 'tight')
    
       
 