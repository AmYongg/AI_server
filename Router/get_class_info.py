import cx_Oracle
import os
import pandas as pd

def get_cls_info(cls_id):
    try:
        LOCATION = r"C:\instanclient_21_3"
        os.environ["PATH"] = LOCATION + ";" + os.environ["PATH"]

        connect = cx_Oracle.connect("hr", "hr", "localhost:1521/")
        cursor = connect.cursor()

        cursor.excute("select class_id from AMP_REGISTER_T", {"class_id":cls_id})
        result = cursor.fetchone()

        df_stud= pd.read_sql("select * from AMP_REGISTER_T INNER JOIN AMP_STUDENT_T ON (AMP_REGISTER_T.CLASS_ID == result) "
                             "WHERE AMP_REGISTER_T.STUD_ID==AMP_STUDENT_T.STUD_ID", {"result":result}, con=connect,)
        df_img=pd.read_sql("select * from AMP_STUD_IMG_T", con = connect)
        df = pd.merge(df_stud, df_img, left_on='STUD_ID', right_on='STUD_ID', how='inner')
        df=df.loc["STUD_NAME", "IMG_SIZE", "IMG_PIXEL"]

    except:
        raise RuntimeError("조회 에러")

