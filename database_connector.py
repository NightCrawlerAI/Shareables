import pyodbc
import datetime

def _connect():
    cnxn_str = ("DRIVER={SQL Server};"

                "Server=E360-DB01;"

                "Database=Voltage;"

                "Trusted_Connection=yes;")

    connsql = pyodbc.connect(cnxn_str)
    return connsql

def _close_commit(cursor, connection):
    cursor.close()
    connection.commit()
    connection.close()

def call_sp_asofdate(procedure, date=None):

    if date == None:
        date = datetime.datetime.now()
    connsql = _connect()

    cursql = connsql.cursor()

    sql = "EXEC " + procedure + " @p_as_of_date = '{0}';".format(date.strftime("%Y-%m-%d"))

    cursql.execute(sql)

    _close_commit(cursql, connsql)

def insert(query):
    connsql = _connect()

    cursql = connsql.cursor()

    cursql.execute(query)
    inserted_id = cursql.execute("SELECT SCOPE_IDENTITY()").fetchone()[0]

    _close_commit(cursql, connsql)
    return inserted_id

def query(query):
    connsql = _connect()

    cursql = connsql.cursor()

    cursql.execute(query)

    _close_commit(cursql, connsql)

def query_return(query):
    connsql = _connect()

    cursql = connsql.cursor()

    cursql.execute(query)

    data = cursql.fetchall()

    _close_commit(cursql, connsql)

    return data

class ConstantConn:
    def __init__(self):
        self.connection = _connect()
        self.cursor = self.connection.cursor()
        self.connection.autocommit = False

    def kill(self):
       self.cursor.close()
       self.commit()
       self.connection.close()

    def query(self, query):
        try:
            self.cursor.execute(query)
        except Exception as e:
            raise Exception(e)

    def insert(self, query):
        self.query(query)
        return self.cursor.execute("SELECT SCOPE_IDENTITY()").fetchone()[0]

    def commit(self):
        self.connection.commit()

    def rollback(self):
        self.connection.rollback()

    def query_return(self, query):
        self.cursor.execute(query)
        return self.cursor.fetchall()