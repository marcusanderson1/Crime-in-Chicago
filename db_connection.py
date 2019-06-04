
import pymysql
pymysql.install_as_MySQLdb()

#Here we open a database connection 
def open_sql_connection():
    
    conn = pymysql.connect(host='localhost', port=3306, user='ml_admin', passwd='P@ssword1', db='ml')
    
    return conn
'''
def close_sql_connection(conn_cursor):
    conn_cursor.close()
    print('closed')
'''

