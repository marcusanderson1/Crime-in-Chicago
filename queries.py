#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 31 21:20:18 2019

@author: marcusanderson
"""

#SQL Scripts to be executed read the data into our Python data frame objects
#data read here is used for the data analysis

sql = '''
 
SELECT `ChicagoCrimeDatasetFull_profile`.`ID`,
        `ChicagoCrimeDatasetFull_profile`.`CaseNumber`,
            `ChicagoCrimeDatasetFull_profile`.`Date`,
                case 
    			when month(Date) = 1 then 'January' 
                when month(Date) = 2 then 'February' 
                when month(Date) = 3 then 'March' 
                when month(Date) = 4 then 'April' 
                when month(Date) = 5 then 'May' 
                when month(Date) = 6 then 'June' 
                when month(Date) = 7 then 'July' 
                when month(Date) = 8 then 'August' 
                when month(Date) = 9 then 'September' 
                when month(Date) = 10 then 'October' 
                when month(Date) = 11 then 'November' 
                when month(Date) = 12 then 'December' 
    	end as Month,
        
        hour(Date) as Hour, 
        case 
    			when weekday(Date) = 0 then 'Monday'
    			when weekday(Date) = 1 then 'Tuesday'
                when weekday(Date) = 2 then 'Wednesday'
                when weekday(Date) = 3 then 'Thursday'
                when weekday(Date) = 4 then 'Friday'
                when weekday(Date) = 5 then 'Saturday'
                when weekday(Date) = 6 then 'Sunday'
    	end as Weekday,

        `ChicagoCrimeDatasetFull_profile`.`PrimaryType`,
        `ChicagoFactorizedValues`.`LocationDescription`,
        `ChicagoCrimeDatasetFull_profile`.`Arrest`,
        `ChicagoFactorizedValues`.`Description`,
        `ChicagoCrimeDatasetFull_profile`.`Domestic`,
        CASE WHEN `ChicagoCrimeDatasetFull_profile`.`Beat` = '' THEN NULL ELSE `ChicagoCrimeDatasetFull_profile`.`Beat` END AS Beat,
        CASE WHEN `ChicagoCrimeDatasetFull_profile`.`District` = '' THEN NULL ELSE  `ChicagoCrimeDatasetFull_profile`.`District` END AS District ,
        CASE WHEN `ChicagoCrimeDatasetFull_profile`.`Ward` = '' THEN NULL ELSE `ChicagoCrimeDatasetFull_profile`.`Ward` END AS Ward,
        CommunityAreaLookup.CommunityAreaName as CommunityArea,
        `ChicagoCrimeDatasetFull_profile`.`Year`,
        `ChicagoCrimeDatasetFull_profile`.`Location`,
        `ChicagoCrimeDatasetFull_profile`.`Longitude`,
        `ChicagoCrimeDatasetFull_profile`.`Latitude` ,
        `ChicagoFactorizedValues`.`Block`,
        `ChicagoFactorizedValues`.`IUCR`,
        `ChicagoFactorizedValues`.`FBICode`    
FROM `ml`.`ChicagoCrimeDatasetFull_profile`
left join CommunityAreaLookup on ChicagoCrimeDatasetFull_profile.CommunityArea = CommunityAreaLookup.CommunityAreaCode
left join ChicagoFactorizedValues on ChicagoCrimeDatasetFull_profile.ID = ChicagoFactorizedValues.ID
where Year between 2005 and 2014
    
'''
#SQL query to read in the Test Data for years 15/16

sql_new_data = '''

SELECT `ChicagoCrimeDatasetFull_profile`.`ID`,
        `ChicagoCrimeDatasetFull_profile`.`CaseNumber`,
            `ChicagoCrimeDatasetFull_profile`.`Date`,
                case 
    			when month(Date) = 1 then 'January' 
                when month(Date) = 2 then 'February' 
                when month(Date) = 3 then 'March' 
                when month(Date) = 4 then 'April' 
                when month(Date) = 5 then 'May' 
                when month(Date) = 6 then 'June' 
                when month(Date) = 7 then 'July' 
                when month(Date) = 8 then 'August' 
                when month(Date) = 9 then 'September' 
                when month(Date) = 10 then 'October' 
                when month(Date) = 11 then 'November' 
                when month(Date) = 12 then 'December' 
    	end as Month,
        
        hour(Date) as Hour, 
        case 
    			when weekday(Date) = 0 then 'Monday'
    			when weekday(Date) = 1 then 'Tuesday'
                when weekday(Date) = 2 then 'Wednesday'
                when weekday(Date) = 3 then 'Thursday'
                when weekday(Date) = 4 then 'Friday'
                when weekday(Date) = 5 then 'Saturday'
                when weekday(Date) = 6 then 'Sunday'
    	end as Weekday,

        `ChicagoCrimeDatasetFull_profile`.`PrimaryType`,
        `ChicagoFactorizedValues`.`LocationDescription`,
        `ChicagoCrimeDatasetFull_profile`.`Arrest`,
        `ChicagoFactorizedValues`.`Description`,
        `ChicagoCrimeDatasetFull_profile`.`Domestic`,
        CASE WHEN `ChicagoCrimeDatasetFull_profile`.`Beat` = '' THEN NULL ELSE `ChicagoCrimeDatasetFull_profile`.`Beat` END AS Beat,
        CASE WHEN `ChicagoCrimeDatasetFull_profile`.`District` = '' THEN NULL ELSE  `ChicagoCrimeDatasetFull_profile`.`District` END AS District ,
        CASE WHEN `ChicagoCrimeDatasetFull_profile`.`Ward` = '' THEN NULL ELSE `ChicagoCrimeDatasetFull_profile`.`Ward` END AS Ward,
        CommunityAreaLookup.CommunityAreaName as CommunityArea,
        `ChicagoCrimeDatasetFull_profile`.`Year`,
        `ChicagoCrimeDatasetFull_profile`.`Location`,
        `ChicagoCrimeDatasetFull_profile`.`Longitude`,
        `ChicagoCrimeDatasetFull_profile`.`Latitude` ,
        `ChicagoFactorizedValues`.`Block`,
        `ChicagoFactorizedValues`.`IUCR`,
        `ChicagoFactorizedValues`.`FBICode`  
FROM `ml`.`ChicagoCrimeDatasetFull_profile`
left join CommunityAreaLookup on ChicagoCrimeDatasetFull_profile.CommunityArea = CommunityAreaLookup.CommunityAreaCode
left join ChicagoFactorizedValues on ChicagoCrimeDatasetFull_profile.ID = ChicagoFactorizedValues.ID
where Year between 2015 and 2016
'''


#Used for working out the Pearson Correlation coeficciont

sql_data_pre_processing = '''
SELECT `ChicagoCrimeDatasetFull_profile`.`ID`,
        `ChicagoCrimeDatasetFull_profile`.`CaseNumber`,
            `ChicagoCrimeDatasetFull_profile`.`Date`,
                case 
    			when month(Date) = 1 then 'January' 
                when month(Date) = 2 then 'February' 
                when month(Date) = 3 then 'March' 
                when month(Date) = 4 then 'April' 
                when month(Date) = 5 then 'May' 
                when month(Date) = 6 then 'June' 
                when month(Date) = 7 then 'July' 
                when month(Date) = 8 then 'August' 
                when month(Date) = 9 then 'September' 
                when month(Date) = 10 then 'October' 
                when month(Date) = 11 then 'November' 
                when month(Date) = 12 then 'December' 
    	end as Month,
        
        hour(Date) as Hour, 
        case 
    			when weekday(Date) = 0 then 'Monday'
    			when weekday(Date) = 1 then 'Tuesday'
                when weekday(Date) = 2 then 'Wednesday'
                when weekday(Date) = 3 then 'Thursday'
                when weekday(Date) = 4 then 'Friday'
                when weekday(Date) = 5 then 'Saturday'
                when weekday(Date) = 6 then 'Sunday'
    	end as Weekday,

        `ChicagoCrimeDatasetFull_profile`.`PrimaryType`,
        `ChicagoFactorizedValues`.`LocationDescription`,
        `ChicagoCrimeDatasetFull_profile`.`Arrest`,
        `ChicagoFactorizedValues`.`Description`,
        `ChicagoCrimeDatasetFull_profile`.`Domestic`,
        CASE WHEN `ChicagoCrimeDatasetFull_profile`.`Beat` = '' THEN NULL ELSE `ChicagoCrimeDatasetFull_profile`.`Beat` END AS Beat,
        CASE WHEN `ChicagoCrimeDatasetFull_profile`.`District` = '' THEN NULL ELSE  `ChicagoCrimeDatasetFull_profile`.`District` END AS District ,
        CASE WHEN `ChicagoCrimeDatasetFull_profile`.`Ward` = '' THEN NULL ELSE `ChicagoCrimeDatasetFull_profile`.`Ward` END AS Ward,
        CommunityAreaLookup.CommunityAreaName as CommunityArea,
        `ChicagoCrimeDatasetFull_profile`.`Year`,
        `ChicagoFactorizedValues`.`Location`,
        `ChicagoCrimeDatasetFull_profile`.`Longitude`,
        `ChicagoCrimeDatasetFull_profile`.`Latitude` ,
        `ChicagoFactorizedValues`.`Block`,
        `ChicagoFactorizedValues`.`IUCR`,
        `ChicagoFactorizedValues`.`FBICode`           
FROM `ml`.`ChicagoCrimeDatasetFull_profile`
left join CommunityAreaLookup on ChicagoCrimeDatasetFull_profile.CommunityArea = CommunityAreaLookup.CommunityAreaCode
left join ChicagoFactorizedValues on ChicagoCrimeDatasetFull_profile.ID = ChicagoFactorizedValues.ID
where Year between 2010 and 2016

'''