import csv
import pymysql 
pymysql.install_as_MySQLdb()

with open('/Users/Admin/Documents/YEAR3/Final project/technical/python/3.6/machine_learning/datasets/factorize.csv','rt') as csvfile:
    data_in = csv.reader(csvfile,delimiter=',')
    k= 0
    
    for a in data_in:
        print (a,len(a))
        if (k==10):
            break
        k= k + 1

    conn = pymysql.connect(host='localhost', port=3306, user='ml_admin', passwd='P@ssword1', db='ml')
    cur = conn.cursor()


    
    query = '''           
        INSERT INTO `ml`.`ChicagoFactorizedValues`
        (`ID`,
        `Block`,
        `IUCR`,
        `Description`,
        `LocationDescription`,
        `FBICode`,
        `Location`)

            VALUES (%s, %s , %s,%s,%s,%s,%s)
    '''

    for index , row in enumerate(data_in):
        
       
       
        if (len(row) == 8 and index != 0):

            line = row[1],row[2] , row[3] ,row[4], row[5] ,row[6] , row[7]
            cur.execute(query,line)    

    conn.commit()
    conn.close()