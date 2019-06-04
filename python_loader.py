import csv
import pymysql 
pymysql.install_as_MySQLdb()

with open('/Users/Admin/Documents/YEAR3/Final project/technical/python/3.6/machine_learning/datasets/Chicago_Crimes_2012_to_2017_profile.csv','rt') as csvfile:
    data_in = csv.reader(csvfile,delimiter='|')
    k= 0
    
    for a in data_in:
        print (a,len(a))
        if (k==10):
            break
        k= k + 1


    conn = pymysql.connect(host='localhost', port=3306, user='ml_admin', passwd='P@ssword1', db='ml')
    cur = conn.cursor()
   

    query = ''' 
        
            INSERT INTO `ml`.`ChicagoCrimeDatasetFull`
            (`ID`,
            `CaseNumber`,
            `Date`,
            `Block`,
            `IUCR`,
            `PrimaryType`,
            `Description`,
            `LocationDescription`,
            `Arrest`,
            `Domestic`,
            `Beat`,
            `District`,
            `Ward`,
            `CommunityArea`,
            `FBICode`,
            `XCoordinate`,
            `YCoordinate`,
            `Year`,
            `UpdatedOn`,
            `Latitude`,
            `Longitude`,
            `Location`
            )
            VALUES (%s, %s , %s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
    '''

    print (query)
    for index , row in enumerate(data_in):
        
        
        if (len(row) == 22 and index != 0):
            #print(row)
            line = row[0],row[1] , row[2] ,row[3], row[4] ,row[5] , row[6], row[7], row[8] , row[9], row[10] , row[11] , row[12] \
            , row[13] ,row[14], row[15] ,row[16] , row[17], row[18], row[19] , row[20] , row[21]
            cur.execute(query,line)
            
        
            conn.commit()
    conn.close()

'''

with open('/machine_learning/scripts/factorize.csv','rt') as csvfile:
    data_in = csv.reader(csvfile,delimiter='|')
    k= 0
    
    for a in data_in:
        print (a,len(a))
        if (k==10):
            break
        k= k + 1

'''


        

#    conn.commit()
#    conn.close()

